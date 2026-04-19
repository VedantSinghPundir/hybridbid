"""
Stage 2: Post-RTC+B co-optimization fine-tuning (stage2_v2).

Pretrain→finetune: loads Stage 1 v5.9 300k checkpoint, expands action space
to 9D (energy + 5 AS products), trains on Dec 2025–Feb 2026 data only.

Key differences from Stage 1:
  - Action space: 9D (3 mode + 1 energy_mag + 5 AS fractions)
  - Reward: symlog(energy_revenue + timing_bonus + as_revenue) + soc_penalty
  - TTFE: two-phase unfreezing (Phase A frozen → B top layer only; Phase C removed)
  - Fresh critic (Stage 1 Q-values invalid for new reward structure)
  - Smaller replay buffer (60k) matched to post-RTC+B data volume

Changes from v1:
  - target_entropy corrected to 5.0 in SACAgent (9D action space; was log(3) ≈ 1.099)
  - Alpha floor at log(0.05) prevents collapse even if target_entropy undershoots
  - Phase C removed entirely (full TTFE unfreeze eroded pretrained representations)
  - Phase A extended to 40% (was 30%) for more critic maturity before TTFE unfreezing
  - Gumbel τ: 0.8→0.5 in Phase A, holds at 0.5 through Phase B (was 0.5→0.1 globally)
  - Total steps: 120k (was 150k; Phase C consumed 60k wasted steps)
"""
import argparse
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.env.ercot_env import ERCOTBatteryEnv
from src.models.sac import SACAgent
from src.training.config import Stage2Config, Stage2V3aConfig, Stage2V60Config

# AS product names (matches env's projected_action[1:6] ordering)
AS_PRODUCTS = ["regup", "regdn", "rrs", "ecrs", "nsrs"]


def symlog(x: float) -> float:
    """DreamerV3 symmetric log transform (Hafner et al., 2023, arXiv:2301.04104)."""
    return math.copysign(math.log1p(abs(x)), x)


def train_stage2(config: Stage2Config = None, scratch: bool = False):
    if config is None:
        config = Stage2Config()

    v3a = isinstance(config, Stage2V3aConfig)
    v60 = isinstance(config, Stage2V60Config)
    if scratch:
        run_label = "stage2_scratch"
    elif v3a:
        run_label = "stage2_v3a"
    else:
        run_label = "stage2_v2"

    phase_b_step = int(config.total_steps * config.phase_b_start_frac)

    print(f"=== Stage 2: Co-Optimization Fine-Tuning ({run_label}) ===")
    print(f"Initialization:    {'SCRATCH (no pretrained weights)' if scratch else 'Stage 1 v5.9 300k pretrained'}")
    print(f"Data:              {config.train_start} → {config.train_end}")
    print(f"Stage 1 ckpt:      {config.stage1_checkpoint}")
    print(f"Device:            {config.device}")
    print(f"Total steps:       {config.total_steps}")
    print(f"Phase A (frozen):  0 → {phase_b_step - 1}")
    print(f"Phase B (top-1):   {phase_b_step} → {config.total_steps - 1}")
    print(f"τ_gumbel:          {config.tau_gumbel_init} → {config.tau_gumbel_final} (Phase A); "
          f"holds at {config.tau_gumbel_final} (Phase B)")
    print(f"target_entropy:    5.0 (corrected for 9D action space)")
    if v3a:
        print(f"Enriched flat obs: static_dim=32 (14 orig + 18 price features), "
              f"obs_dim=108, TTFE input=12-dim (unchanged)")

    # --- Environment ---
    battery_config = dict(
        p_max=config.p_max, e_max=config.e_max,
        soc_min_frac=config.soc_min_frac, soc_max_frac=config.soc_max_frac,
        soc_initial_frac=config.soc_initial_frac,
        eta_ch=config.eta_ch, eta_dch=config.eta_dch,
        degradation_cost=config.degradation_cost,
    )
    env = ERCOTBatteryEnv(
        data_dir=config.data_dir,
        mode="co_optimize",
        battery_config=battery_config,
        seq_len=config.seq_len,
        date_range=(config.train_start, config.train_end),
        enriched_flat=v3a,
    )

    # --- Agent ---
    agent = SACAgent(
        stage=2,
        device=config.device,
        n_prices=config.n_prices,
        n_prices_flat=getattr(config, "n_prices_flat", None),
        d_model=config.d_model,
        nhead=config.nhead,
        n_layers=config.n_layers,
        seq_len=config.seq_len,
        static_dim=config.static_dim,
        hidden_dim=config.hidden_dim,
        lr_actor=config.lr_actor,
        lr_critic=config.lr_critic,
        lr_ttfe=config.lr_ttfe,
        gamma=config.gamma,
        tau=config.tau,
        buffer_capacity=config.buffer_capacity,
        batch_size=config.batch_size,
        max_grad_norm=config.max_grad_norm,
        tau_gumbel=config.tau_gumbel_init,
    )

    # Weight initialization
    if scratch:
        # Scratch baseline: all weights remain at PyTorch default random init.
        # Actor AS heads use the same initialization scheme as mode/energy heads
        # (no near-zero bias), so the only variable vs Stage 2 v2 is initialization.
        print("Scratch mode: all weights randomly initialized (no Stage 1 loading)")
    else:
        # Load pretrained weights; critic stays fresh (init_from_stage1 does not copy critic)
        if os.path.exists(config.stage1_checkpoint):
            agent.init_from_stage1(config.stage1_checkpoint)
            print(f"Loaded Stage 1 weights from {config.stage1_checkpoint}")
        else:
            print(f"WARNING: Stage 1 checkpoint not found: {config.stage1_checkpoint}")
            print("         Falling back to random initialization.")

    # Confirm target_entropy was set correctly in agent
    print(f"Agent target_entropy: {agent.target_entropy:.4f} (expected 5.0)")

    # Phase A: freeze TTFE
    agent.freeze_ttfe()
    frozen_params = sum(p.numel() for p in agent.ttfe.parameters())
    print(f"Phase A: TTFE frozen ({frozen_params:,} params)")

    # --- Training loop ---
    obs, _ = env.reset()
    episode_reward = 0.0      # symlog-transformed
    episode_raw_reward = 0.0  # pre-symlog total
    episode_energy_rev = 0.0  # pre-symlog energy + timing_bonus
    episode_as_rev = 0.0      # pre-symlog AS revenue
    episode_count = 0
    step = 0
    log_interval = config.log_interval
    save_interval = config.save_every
    t_start = time.time()
    current_phase = "A"

    # Rolling metrics
    recent_rewards = []
    recent_raw_rewards = []
    recent_energy_revs = []
    recent_as_revs = []
    recent_socs = []
    mode_counts = {0: 0, 1: 0, 2: 0}
    # Per-product AS fraction accumulators (rolling window of step-level values)
    as_frac_window = {p: [] for p in AS_PRODUCTS}

    # Enriched feature sanity trackers (v3a only)
    recent_pct_rank_24h = []
    recent_z_24h = []
    recent_da_rt_basis = []

    prev_snapshot = None
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    print(f"Warming up for {config.warmup_steps} steps...")

    while step < config.total_steps:

        # --- Phase transition: A → B ---
        if current_phase == "A" and step >= phase_b_step:
            agent.unfreeze_ttfe_top_layers(n_layers=1, lr=config.lr_ttfe)
            current_phase = "B"
            print(
                f"\n[Phase B] Step {step}: Unfroze top TTFE layer "
                f"(lr={config.lr_ttfe}, grad_scale=0.1)",
                flush=True,
            )
            # Confirm TTFE optimizer lr for each param group
            for i, group in enumerate(agent.ttfe_optimizer.param_groups):
                print(
                    f"  TTFE optimizer param_group {i}: "
                    f"lr={group['lr']}, n_params={len(group['params'])}",
                    flush=True,
                )

        # --- Action ---
        action = agent.select_action(obs)

        # --- Step ---
        next_obs, reward, terminated, truncated, info = env.step(action)

        # Symlog transform: economic component only; SoC penalty passes through unchanged
        soc_penalty = -50.0 if info["soc_violated"] else 0.0
        raw_econ = info["energy_revenue"] + info["timing_bonus"] + info["as_revenue"]
        transformed_reward = symlog(raw_econ) + soc_penalty

        episode_reward += transformed_reward
        episode_raw_reward += reward
        episode_energy_rev += info["energy_revenue"] + info["timing_bonus"]
        episode_as_rev += info["as_revenue"]
        recent_socs.append(info["soc"])
        mode_counts[info["mode"]] += 1

        # Per-product AS fractions from projected action (projected_action[1:6] in MW)
        if "projected_action" in info and len(info["projected_action"]) >= 6:
            as_mw = info["projected_action"][1:6]
            for i, prod in enumerate(AS_PRODUCTS):
                as_frac_window[prod].append(float(as_mw[i]) / config.p_max)
        else:
            for prod in AS_PRODUCTS:
                as_frac_window[prod].append(0.0)

        # Track enriched price features for sanity logging (v3a)
        # static_features layout (enriched_flat): [system(7), time(6), soc(1), price_feats(18)]
        # price_feats: [pct_rank_4h, pct_rank_12h, pct_rank_24h, z_4h, z_12h, z_24h, ...]
        if v3a and "static_features" in obs:
            sf = obs["static_features"]
            if len(sf) >= 32:
                recent_pct_rank_24h.append(float(sf[16]))   # pct_rank_24h
                recent_z_24h.append(float(sf[19]))           # z_24h
                recent_da_rt_basis.append(float(sf[29]))     # da_rt_basis

        # Store symlog-transformed reward in replay buffer
        agent.buffer.add(obs, action, transformed_reward, next_obs, terminated)

        # --- Gumbel temperature schedule ---
        # Phase A: anneal 0.8 → 0.5 linearly
        # Phase B: hold at 0.5 (warm enough for continued exploration; no further cooling)
        if step < phase_b_step:
            agent.tau_gumbel = (
                config.tau_gumbel_init
                - (config.tau_gumbel_init - config.tau_gumbel_final) * (step / phase_b_step)
            )
        else:
            agent.tau_gumbel = config.tau_gumbel_final

        # --- Update ---
        metrics = {}
        if step >= config.warmup_steps:
            if step % 100 == 0:
                prev_snapshot = agent.snapshot_state()

            metrics = agent.update(tau_gumbel=agent.tau_gumbel, phase=current_phase)

            if metrics.get("nan_detected"):
                nan_source = metrics.get("nan_source", "unknown")
                print(
                    f"\nFATAL: NaN detected in {nan_source} at step {step}.",
                    flush=True,
                )
                if prev_snapshot is not None:
                    emergency_path = os.path.join(
                        config.checkpoint_dir,
                        f"emergency_pre_nan_step{step}.pt",
                    )
                    agent.save_emergency_checkpoint(emergency_path, prev_snapshot)
                    print(f"  Emergency checkpoint saved: {emergency_path}")
                return agent, recent_rewards

        obs = next_obs
        step += 1

        if terminated or truncated:
            episode_count += 1
            recent_rewards.append(episode_reward)
            recent_raw_rewards.append(episode_raw_reward)
            recent_energy_revs.append(episode_energy_rev)
            recent_as_revs.append(episode_as_rev)
            episode_reward = 0.0
            episode_raw_reward = 0.0
            episode_energy_rev = 0.0
            episode_as_rev = 0.0
            obs, _ = env.reset()

        # --- Logging ---
        if step % log_interval == 0 and metrics:
            elapsed = time.time() - t_start
            steps_per_sec = step / elapsed if elapsed > 0 else 0

            avg_reward = np.mean(recent_rewards[-10:]) if recent_rewards else 0
            avg_raw_reward = np.mean(recent_raw_rewards[-10:]) if recent_raw_rewards else 0
            avg_energy_rev = np.mean(recent_energy_revs[-10:]) if recent_energy_revs else 0
            avg_as_rev = np.mean(recent_as_revs[-10:]) if recent_as_revs else 0
            avg_soc = np.mean(recent_socs[-288:]) if recent_socs else 0

            total_modes = sum(mode_counts.values())
            if total_modes > 0:
                pct_ch = 100.0 * mode_counts[0] / total_modes
                pct_dc = 100.0 * mode_counts[1] / total_modes
                pct_id = 100.0 * mode_counts[2] / total_modes
            else:
                pct_ch = pct_dc = pct_id = 0.0
            mode_counts = {0: 0, 1: 0, 2: 0}

            # AS fraction means over recent window
            window_size = 1000
            as_means = {
                prod: np.mean(as_frac_window[prod][-window_size:])
                if as_frac_window[prod] else 0.0
                for prod in AS_PRODUCTS
            }

            has_nan = any(
                np.isnan(v) for v in metrics.values() if isinstance(v, float)
            )
            nan_flag = " *** NaN DETECTED ***" if has_nan else ""

            # grad_c warning (applies in Phase B when TTFE is live)
            grad_c_val = metrics.get("critic_grad_norm", 0)
            if current_phase == "B" and grad_c_val > 100:
                print(
                    f"  [WARNING] Phase B grad_c={grad_c_val:.1f} exceeds 100 at step {step}",
                    flush=True,
                )

            # Enriched feature summary (v3a only)
            feat_str = ""
            if v3a and recent_pct_rank_24h:
                avg_pct = np.mean(recent_pct_rank_24h[-200:])
                avg_z   = np.mean(recent_z_24h[-200:])
                avg_basis = np.mean(recent_da_rt_basis[-200:])
                feat_str = (f" | pct24h={avg_pct:.2f} z24h={avg_z:.2f}"
                            f" da_rt={avg_basis:.4f}")

            print(
                f"Step {step:>7d}/{config.total_steps} | "
                f"ph={current_phase} | "
                f"ep={episode_count} | "
                f"critic={metrics.get('critic_loss', 0):.4f} | "
                f"actor={metrics.get('actor_loss', 0):.4f} | "
                f"alpha={metrics.get('alpha', 0):.4f} | "
                f"avg_reward={avg_reward:.1f} | "
                f"avg_energy_rev={avg_energy_rev:.2f} | "
                f"avg_as_rev={avg_as_rev:.2f} | "
                f"avg_soc={avg_soc:.2f} | "
                f"grad_c={metrics.get('critic_grad_norm', 0):.3f} "
                f"[q1={metrics.get('grad_q1', 0):.1f} q2={metrics.get('grad_q2', 0):.1f}] | "
                f"grad_a={metrics.get('actor_grad_norm', 0):.3f} | "
                f"grad_t={metrics.get('ttfe_grad_norm', 0):.3f} "
                f"[proj={metrics.get('grad_ttfe_proj', 0):.1f} attn={metrics.get('grad_ttfe_attn', 0):.1f}] | "
                f"mode=[ch={pct_ch:.0f}% dc={pct_dc:.0f}% id={pct_id:.0f}%] | "
                f"as=[ru={as_means['regup']:.3f} rd={as_means['regdn']:.3f} "
                f"rrs={as_means['rrs']:.3f} ec={as_means['ecrs']:.3f} "
                f"ns={as_means['nsrs']:.3f}] | "
                f"tau_g={agent.tau_gumbel:.3f} | "
                f"{steps_per_sec:.1f} steps/s{feat_str}{nan_flag}",
                flush=True,
            )

            if has_nan:
                print("FATAL: NaN in metrics. Saving emergency checkpoint and stopping.")
                if prev_snapshot is not None:
                    emergency_path = os.path.join(
                        config.checkpoint_dir, f"emergency_step{step}.pt"
                    )
                    agent.save_emergency_checkpoint(emergency_path, prev_snapshot)
                    print(f"  Emergency checkpoint saved: {emergency_path}")
                return agent, []

            # Trim rolling buffers to avoid memory growth
            if len(recent_socs) > 1000:
                recent_socs = recent_socs[-500:]
            for prod in AS_PRODUCTS:
                if len(as_frac_window[prod]) > 5000:
                    as_frac_window[prod] = as_frac_window[prod][-2000:]
            if len(recent_pct_rank_24h) > 2000:
                recent_pct_rank_24h = recent_pct_rank_24h[-1000:]
                recent_z_24h = recent_z_24h[-1000:]
                recent_da_rt_basis = recent_da_rt_basis[-1000:]

        # --- Checkpoint ---
        if step % save_interval == 0:
            ckpt_path = os.path.join(
                config.checkpoint_dir, f"checkpoint_step{step}.pt"
            )
            agent.save_checkpoint(ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}", flush=True)

    # Final checkpoint
    final_path = os.path.join(config.checkpoint_dir, "checkpoint_final.pt")
    agent.save_checkpoint(final_path)

    elapsed = time.time() - t_start
    print(f"\n=== Training Complete ===")
    print(f"Total steps:  {step}")
    print(f"Episodes:     {episode_count}")
    print(f"Time:         {elapsed / 3600:.2f} hours")
    print(f"Final ckpt:   {final_path}")
    if recent_rewards:
        print(f"Last 10 episode avg reward: {np.mean(recent_rewards[-10:]):.2f}")

    return agent, recent_rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2 Training (stage2_v2 / v3a / scratch)")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument(
        "--stage1-checkpoint", type=str, default=None,
        help="Override Stage 1 checkpoint path"
    )
    parser.add_argument(
        "--scratch", action="store_true",
        help="Train from scratch (no pretrained weights). Controls for initialization "
             "only — all other hyperparameters are identical to Stage 2 v2."
    )
    parser.add_argument(
        "--v3a", action="store_true",
        help="Stage 2 v3a: enriched flat obs (18 price features, obs_dim=108). "
             "TTFE input stays 12-dim — loads perfectly from v5.9 300k."
    )
    parser.add_argument(
        "--v60", action="store_true", default=False,
        help="Stage 2 compatible with Stage 1 v6.0 (36-dim TTFE input)."
    )
    args = parser.parse_args()

    if args.v60:
        config = Stage2V60Config()
    elif args.v3a:
        config = Stage2V3aConfig()
    else:
        config = Stage2Config()

    if args.scratch:
        config.checkpoint_dir = "checkpoints/stage2_scratch"
    if args.steps is not None:
        config.total_steps = args.steps
    if args.device is not None:
        config.device = args.device
    if args.log_interval is not None:
        config.log_interval = args.log_interval
    if args.stage1_checkpoint is not None:
        config.stage1_checkpoint = args.stage1_checkpoint

    train_stage2(config, scratch=args.scratch) 

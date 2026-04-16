"""
Stage 2: Post-RTC+B co-optimization fine-tuning (stage2_v1).

Pretrain→finetune: loads Stage 1 v5.9 300k checkpoint, expands action space
to 9D (energy + 5 AS products), trains on Dec 2025–Feb 2026 data only.

Key differences from Stage 1:
  - Action space: 9D (3 mode + 1 energy_mag + 5 AS fractions)
  - Reward: symlog(energy_revenue + timing_bonus + as_revenue) + soc_penalty
  - TTFE: progressive unfreezing (Phase A frozen → B top layer → C all)
  - Fresh critic (Stage 1 Q-values invalid for new reward structure)
  - Smaller replay buffer (60k) matched to post-RTC+B data volume
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
from src.training.config import Stage2Config

# AS product names (matches env's projected_action[1:6] ordering)
AS_PRODUCTS = ["regup", "regdn", "rrs", "ecrs", "nsrs"]


def symlog(x: float) -> float:
    """DreamerV3 symmetric log transform (Hafner et al., 2023, arXiv:2301.04104)."""
    return math.copysign(math.log1p(abs(x)), x)


def train_stage2(config: Stage2Config = None):
    if config is None:
        config = Stage2Config()

    phase_b_step = int(config.total_steps * config.phase_b_start_frac)
    phase_c_step = int(config.total_steps * config.phase_c_start_frac)

    print(f"=== Stage 2: Co-Optimization Fine-Tuning (stage2_v1) ===")
    print(f"Data:              {config.train_start} → {config.train_end}")
    print(f"Stage 1 ckpt:      {config.stage1_checkpoint}")
    print(f"Device:            {config.device}")
    print(f"Total steps:       {config.total_steps}")
    print(f"Phase A (frozen):  0 → {phase_b_step}")
    print(f"Phase B (top-1):   {phase_b_step} → {phase_c_step}")
    print(f"Phase C (all):     {phase_c_step} → {config.total_steps}")
    print(f"τ_gumbel:          {config.tau_gumbel_init} → {config.tau_gumbel_final}")

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
    )

    # --- Agent ---
    agent = SACAgent(
        stage=2,
        device=config.device,
        n_prices=config.n_prices,
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

    # Load pretrained weights; critic stays fresh (init_from_stage1 does not copy critic)
    if os.path.exists(config.stage1_checkpoint):
        agent.init_from_stage1(config.stage1_checkpoint)
        print(f"Loaded Stage 1 weights from {config.stage1_checkpoint}")
    else:
        print(f"WARNING: Stage 1 checkpoint not found: {config.stage1_checkpoint}")
        print("         Training from scratch.")

    # Phase A: freeze TTFE
    agent.freeze_ttfe()
    frozen_params = sum(p.numel() for p in agent.ttfe.parameters())
    print(f"Phase A: TTFE frozen ({frozen_params:,} params)")

    # Gumbel annealing range
    tau_gumbel_range = config.tau_gumbel_init - config.tau_gumbel_final

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

    prev_snapshot = None
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    print(f"Warming up for {config.warmup_steps} steps...")

    while step < config.total_steps:

        # --- Phase transitions ---
        if current_phase == "A" and step >= phase_b_step:
            agent.unfreeze_ttfe_top_layers(n_layers=1, lr=config.lr_ttfe)
            current_phase = "B"
            print(
                f"\n[Phase B] Step {step}: Unfroze top TTFE layer "
                f"(lr={config.lr_ttfe})",
                flush=True,
            )
        elif current_phase == "B" and step >= phase_c_step:
            agent.unfreeze_ttfe_all(lr=config.lr_ttfe)
            current_phase = "C"
            print(
                f"\n[Phase C] Step {step}: Unfroze all TTFE layers "
                f"(lr={config.lr_ttfe}; grad_scale=0.1; critic_clip=0.5)",
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

        # Store symlog-transformed reward in replay buffer
        agent.buffer.add(obs, action, transformed_reward, next_obs, terminated)

        # --- Anneal Gumbel temperature ---
        frac = min(1.0, step / max(config.total_steps, 1))
        agent.tau_gumbel = config.tau_gumbel_init - frac * tau_gumbel_range

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

            # Phase C grad_c warning
            grad_c_val = metrics.get("critic_grad_norm", 0)
            if current_phase == "C" and grad_c_val > 100:
                print(
                    f"  [WARNING] Phase C grad_c={grad_c_val:.1f} exceeds 100 at step {step}",
                    flush=True,
                )

            print(
                f"Step {step:>7d}/{config.total_steps} | "
                f"ph={current_phase} | "
                f"ep={episode_count} | "
                f"critic={metrics.get('critic_loss', 0):.4f} | "
                f"actor={metrics.get('actor_loss', 0):.4f} | "
                f"alpha={metrics.get('alpha', 0):.4f} | "
                f"avg_reward={avg_reward:.1f} | "
                f"avg_raw_reward={avg_raw_reward:.1f} | "
                f"avg_energy_rev={avg_energy_rev:.1f} | "
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
                f"{steps_per_sec:.1f} steps/s{nan_flag}",
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
    parser = argparse.ArgumentParser(description="Stage 2 Training (stage2_v1)")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument(
        "--stage1-checkpoint", type=str, default=None,
        help="Override Stage 1 checkpoint path"
    )
    args = parser.parse_args()

    config = Stage2Config()
    if args.steps is not None:
        config.total_steps = args.steps
    if args.device is not None:
        config.device = args.device
    if args.log_interval is not None:
        config.log_interval = args.log_interval
    if args.stage1_checkpoint is not None:
        config.stage1_checkpoint = args.stage1_checkpoint

    train_stage2(config)

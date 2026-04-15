"""
Stage 1: Energy-only pretraining on pre-RTC+B data.

Full training loop with logging, checkpointing, and numerical stability.
"""

import argparse
import os
import sys
import time

import math

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


def symlog(x: float) -> float:
    """DreamerV3 symmetric logarithmic transform (Hafner et al., 2023, arXiv:2301.04104).

    Compresses large reward magnitudes while preserving sign and ordering.
    symlog(0)=0, symlog(1)≈0.69, symlog(100)≈4.62, symlog(9000)≈9.10.
    Applied to the economic reward component only (NOT to the SoC penalty).
    """
    return math.copysign(math.log1p(abs(x)), x)

from src.env.ercot_env import ERCOTBatteryEnv
from src.models.sac import SACAgent
from src.training.config import Stage1Config


def train_stage1(config: Stage1Config = None):
    if config is None:
        config = Stage1Config()

    print(f"=== Stage 1: Energy-Only Training ===")
    print(f"Data: {config.train_start} to {config.train_end}")
    print(f"Device: {config.device}")
    print(f"Total steps: {config.total_steps}")
    print(f"Max grad norm: {config.max_grad_norm}")
    print(f"LR: actor={config.lr_actor} critic={config.lr_critic} ttfe={config.lr_ttfe}")
    print(f"τ_gumbel: {config.tau_gumbel_init} → {config.tau_gumbel_final}")

    # Create environment
    battery_config = dict(
        p_max=config.p_max, e_max=config.e_max,
        soc_min_frac=config.soc_min_frac, soc_max_frac=config.soc_max_frac,
        soc_initial_frac=config.soc_initial_frac,
        eta_ch=config.eta_ch, eta_dch=config.eta_dch,
        degradation_cost=config.degradation_cost,
    )
    env = ERCOTBatteryEnv(
        data_dir=config.data_dir,
        mode="energy_only",
        battery_config=battery_config,
        seq_len=config.seq_len,
        date_range=(config.train_start, config.train_end),
    )

    # Create SAC agent
    agent = SACAgent(
        stage=1,
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

    # Gumbel temperature annealing schedule
    tau_gumbel_range = config.tau_gumbel_init - config.tau_gumbel_final

    # Training loop
    obs, _ = env.reset()
    episode_reward = 0.0      # symlog-transformed (what the agent trains on)
    episode_raw_reward = 0.0  # pre-symlog (for comparison with v5.7/v5.8)
    episode_count = 0
    step = 0
    log_interval = config.log_interval
    save_interval = config.save_every
    t_start = time.time()

    # Rolling metrics for logging
    recent_rewards = []      # symlog-transformed episode totals
    recent_raw_rewards = []  # pre-symlog episode totals
    recent_socs = []
    mode_counts = {0: 0, 1: 0, 2: 0}  # charge=0, discharge=1, idle=2

    # Last-good-state snapshot for NaN recovery
    prev_snapshot = None

    os.makedirs(config.checkpoint_dir, exist_ok=True)

    print(f"Warming up for {config.warmup_steps} steps...")

    while step < config.total_steps:
        # Select action
        action = agent.select_action(obs)

        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action)

        # Apply symlog to economic reward component; keep SoC penalty at original scale.
        # symlog compresses ERCOT's heavy-tailed price distribution (Cauchy residuals,
        # Uri storm $9k/MWh) without destroying reward ordering. DreamerV3 (Hafner 2023).
        soc_penalty = -50.0 if info["soc_violated"] else 0.0
        raw_econ = info["energy_revenue"] + info["timing_bonus"]
        transformed_reward = symlog(raw_econ) + soc_penalty

        episode_reward += transformed_reward
        episode_raw_reward += reward  # original env reward (pre-symlog) for logging
        recent_socs.append(info["soc"])
        mode_counts[info["mode"]] += 1

        # Store symlog-transformed reward in replay buffer
        agent.buffer.add(obs, action, transformed_reward, next_obs, terminated)

        # Anneal Gumbel temperature
        frac = min(1.0, step / max(config.total_steps, 1))
        agent.tau_gumbel = config.tau_gumbel_init - frac * tau_gumbel_range

        # Update agent
        metrics = {}
        if step >= config.warmup_steps:
            # Snapshot state every 100 steps for NaN recovery
            if step % 100 == 0:
                prev_snapshot = agent.snapshot_state()

            metrics = agent.update(tau_gumbel=agent.tau_gumbel)

            # NaN guard: check if update() detected NaN in parameters
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
                    print(f"  Emergency checkpoint (last good state) saved: {emergency_path}")
                else:
                    print("  No previous snapshot available for emergency save.")
                return agent, recent_rewards

        obs = next_obs
        step += 1

        if terminated or truncated:
            episode_count += 1
            recent_rewards.append(episode_reward)
            recent_raw_rewards.append(episode_raw_reward)
            episode_reward = 0.0
            episode_raw_reward = 0.0
            obs, _ = env.reset()

        # Logging
        if step % log_interval == 0 and metrics:
            elapsed = time.time() - t_start
            steps_per_sec = step / elapsed if elapsed > 0 else 0
            avg_reward = np.mean(recent_rewards[-10:]) if recent_rewards else 0
            avg_raw_reward = np.mean(recent_raw_rewards[-10:]) if recent_raw_rewards else 0
            avg_soc = np.mean(recent_socs[-288:]) if recent_socs else 0

            # Mode distribution over the logging window
            total_modes = sum(mode_counts.values())
            if total_modes > 0:
                mode_pct_charge = 100.0 * mode_counts[0] / total_modes
                mode_pct_discharge = 100.0 * mode_counts[1] / total_modes
                mode_pct_idle = 100.0 * mode_counts[2] / total_modes
            else:
                mode_pct_charge = mode_pct_discharge = mode_pct_idle = 0.0
            mode_counts = {0: 0, 1: 0, 2: 0}  # reset window

            gumbel_temperature = agent.tau_gumbel

            # Check for NaN in metrics values (belt-and-suspenders with param check)
            has_nan = any(
                np.isnan(v) for v in metrics.values() if isinstance(v, float)
            )
            nan_flag = " *** NaN DETECTED ***" if has_nan else ""

            print(
                f"Step {step:>7d}/{config.total_steps} | "
                f"ep={episode_count} | "
                f"critic={metrics.get('critic_loss', 0):.4f} | "
                f"actor={metrics.get('actor_loss', 0):.4f} | "
                f"alpha={metrics.get('alpha', 0):.4f} | "
                f"avg_reward={avg_reward:.1f} | "
                f"avg_raw_reward={avg_raw_reward:.1f} | "
                f"avg_soc={avg_soc:.2f} | "
                f"grad_c={metrics.get('critic_grad_norm', 0):.3f} "
                f"[q1={metrics.get('grad_q1', 0):.1f} q2={metrics.get('grad_q2', 0):.1f}] | "
                f"grad_a={metrics.get('actor_grad_norm', 0):.3f} | "
                f"grad_t={metrics.get('ttfe_grad_norm', 0):.3f} "
                f"[proj={metrics.get('grad_ttfe_proj', 0):.1f} attn={metrics.get('grad_ttfe_attn', 0):.1f}] | "
                f"mode=[ch={mode_pct_charge:.0f}% dc={mode_pct_discharge:.0f}% id={mode_pct_idle:.0f}%] | "
                f"tau_g={gumbel_temperature:.3f} | "
                f"{steps_per_sec:.1f} steps/s{nan_flag}",
                flush=True,
            )

            if has_nan:
                print("FATAL: NaN detected in metrics. Saving emergency checkpoint and stopping.")
                if prev_snapshot is not None:
                    emergency_path = os.path.join(
                        config.checkpoint_dir, f"emergency_step{step}.pt"
                    )
                    agent.save_emergency_checkpoint(emergency_path, prev_snapshot)
                    print(f"  Emergency checkpoint saved: {emergency_path}")
                return agent, []

            # Clear old SoC history to avoid memory growth
            if len(recent_socs) > 1000:
                recent_socs = recent_socs[-500:]

        # Checkpointing
        if step % save_interval == 0:
            ckpt_path = os.path.join(config.checkpoint_dir, f"checkpoint_step{step}.pt")
            agent.save_checkpoint(ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}", flush=True)

    # Final checkpoint
    final_path = os.path.join(config.checkpoint_dir, "checkpoint_final.pt")
    agent.save_checkpoint(final_path)

    elapsed = time.time() - t_start
    print(f"\n=== Training Complete ===")
    print(f"Total steps: {step}")
    print(f"Episodes: {episode_count}")
    print(f"Time: {elapsed/3600:.2f} hours")
    print(f"Final checkpoint: {final_path}")

    if recent_rewards:
        print(f"Last 10 episode avg reward: {np.mean(recent_rewards[-10:]):.2f}")

    return agent, recent_rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1 Training")
    parser.add_argument("--steps", type=int, default=None, help="Override total_steps")
    parser.add_argument("--start", type=str, default=None, help="Override train_start date")
    parser.add_argument("--end", type=str, default=None, help="Override train_end date")
    parser.add_argument("--device", type=str, default=None, help="Override device")
    parser.add_argument("--log-interval", type=int, default=None, help="Override log interval")
    args = parser.parse_args()

    config = Stage1Config()
    if args.steps is not None:
        config.total_steps = args.steps
    if args.start is not None:
        config.train_start = args.start
    if args.end is not None:
        config.train_end = args.end
    if args.device is not None:
        config.device = args.device
    if args.log_interval is not None:
        config.log_interval = args.log_interval

    train_stage1(config)

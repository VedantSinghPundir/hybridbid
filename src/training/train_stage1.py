"""
Stage 1: Energy-only pretraining on pre-RTC+B data.

Minimal working stub — runs end-to-end on a tiny data slice to verify the loop.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.env.ercot_env import ERCOTBatteryEnv
from src.models.sac import SACAgent
from src.training.config import Stage1Config


def train_stage1(config: Stage1Config = None):
    if config is None:
        config = Stage1Config()

    print(f"=== Stage 1: Energy-Only Training ===")
    print(f"Data: {config.train_start} to {config.train_end}")
    print(f"Device: {config.device}")

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
    )

    # Training loop
    obs, _ = env.reset()
    episode_reward = 0.0
    episode_count = 0
    step = 0
    metrics_history = []

    total_steps = min(config.total_steps, 1000)  # Cap for stub
    print(f"Running {total_steps} steps...")

    while step < total_steps:
        # Select action
        action = agent.select_action(obs)

        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward

        # Store transition
        agent.buffer.add(obs, action, reward, next_obs, terminated)

        # Update agent
        if step >= config.warmup_steps:
            metrics = agent.update()
            if metrics and step % 100 == 0:
                metrics_history.append(metrics)
                print(f"  Step {step}: critic_loss={metrics['critic_loss']:.4f}, "
                      f"actor_loss={metrics['actor_loss']:.4f}, "
                      f"alpha={metrics['alpha']:.4f}, "
                      f"SoC={info['soc']:.2f}")

        obs = next_obs
        step += 1

        if terminated:
            episode_count += 1
            print(f"  Episode {episode_count} reward: {episode_reward:.2f}")
            episode_reward = 0.0
            obs, _ = env.reset()

    # Save checkpoint
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(config.checkpoint_dir, "checkpoint.pt")
    agent.save_checkpoint(ckpt_path)
    print(f"Checkpoint saved to {ckpt_path}")

    # Print summary
    if metrics_history:
        final = metrics_history[-1]
        print(f"\nFinal metrics: critic_loss={final['critic_loss']:.4f}, "
              f"actor_loss={final['actor_loss']:.4f}")

    return agent, metrics_history


if __name__ == "__main__":
    train_stage1()

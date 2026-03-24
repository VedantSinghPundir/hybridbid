"""
Stage 2: Co-optimization finetuning on post-RTC+B data.

Minimal working stub with progressive unfreezing schedule.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.env.ercot_env import ERCOTBatteryEnv
from src.models.sac import SACAgent
from src.training.config import Stage2Config


def train_stage2(config: Stage2Config = None):
    if config is None:
        config = Stage2Config()

    print(f"=== Stage 2: Co-Optimization Finetuning ===")
    print(f"Data: {config.train_start} to {config.train_end}")
    print(f"Stage 1 checkpoint: {config.stage1_checkpoint}")
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
        mode="co_optimize",
        battery_config=battery_config,
        seq_len=config.seq_len,
        date_range=(config.train_start, config.train_end),
    )

    # Create SAC agent
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
    )

    # Initialize from Stage 1
    if os.path.exists(config.stage1_checkpoint):
        agent.init_from_stage1(config.stage1_checkpoint)
        print("Initialized from Stage 1 checkpoint")
    else:
        print("WARNING: Stage 1 checkpoint not found, training from scratch")

    # Progressive unfreezing
    total_steps = min(config.total_steps, 1000)  # Cap for stub
    phase1_steps = min(config.phase1_steps, total_steps // 2)
    phase2_steps = total_steps - phase1_steps

    # Phase 1: Freeze TTFE, train heads
    print(f"\n--- Phase 1: Frozen TTFE ({phase1_steps} steps) ---")
    agent.freeze_ttfe()

    obs, _ = env.reset()
    episode_reward = 0.0
    episode_count = 0
    step = 0
    metrics_history = []

    while step < total_steps:
        # Phase transition
        if step == phase1_steps:
            print(f"\n--- Phase 2: Unfreeze top TTFE layers ({phase2_steps} steps) ---")
            agent.unfreeze_ttfe_top_layers(n_layers=1, lr=config.lr_ttfe)

        # Select action
        action = agent.select_action(obs)

        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward

        # Store transition
        agent.buffer.add(obs, action, reward, next_obs, terminated)

        # Update
        if step >= config.warmup_steps:
            metrics = agent.update()
            if metrics and step % 100 == 0:
                metrics_history.append(metrics)
                phase = "P1" if step < phase1_steps else "P2"
                print(f"  [{phase}] Step {step}: critic_loss={metrics['critic_loss']:.4f}, "
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

    if metrics_history:
        final = metrics_history[-1]
        print(f"\nFinal metrics: critic_loss={final['critic_loss']:.4f}, "
              f"actor_loss={final['actor_loss']:.4f}")

    return agent, metrics_history


if __name__ == "__main__":
    train_stage2()

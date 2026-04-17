"""
Stage 1 checkpoint evaluation against TBx and Perfect Foresight baselines.

Runs deterministic rollouts over the Stage 1 test set (2025-10-01 → 2025-12-04)
and reports actual daily revenue in $.

Revenue formula:
  info["energy_revenue"] is in p.u. units (energy_mag × rt_lmp × η × Δt).
  Actual $ = info["energy_revenue"] × P_max (10 MW).

Usage:
  python -m src.evaluation.evaluate_stage1 --checkpoint checkpoints/stage1/checkpoint_step450000.pt
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.env.ercot_env import ERCOTBatteryEnv
from src.models.sac import SACAgent
from src.training.config import Stage1Config, Stage1V60Config

# Baselines from CLAUDE.md (pre-RTC+B, $/day for 10 MW / 20 MWh battery)
TBEX_DAILY = 870.0
PERFECT_FORESIGHT_DAILY = 1519.0

# Stage 1 test set — pre-RTC+B held-out period
TEST_START = "2025-10-01"
TEST_END   = "2025-12-04"


def evaluate(checkpoint_path: str, config: Stage1Config = None, verbose: bool = True):
    if config is None:
        config = Stage1Config()

    enriched = isinstance(config, Stage1V60Config)

    # --- Environment (test set, deterministic) ---
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
        date_range=(TEST_START, TEST_END),
        enriched_obs=enriched,
    )
    n_days = len(env.day_starts)

    # --- Agent ---
    agent = SACAgent(
        stage=1,
        device=config.device,
        n_prices=config.n_prices,
        n_prices_flat=getattr(config, "n_prices_flat", None),
        d_model=config.d_model,
        nhead=config.nhead,
        n_layers=config.n_layers,
        seq_len=config.seq_len,
        static_dim=config.static_dim,
        hidden_dim=config.hidden_dim,
        tau_gumbel=config.tau_gumbel_final,  # fully annealed = deterministic
    )
    agent.load_checkpoint(checkpoint_path)

    if verbose:
        print(f"\n=== Stage 1 Evaluation ===")
        print(f"Checkpoint : {checkpoint_path}")
        print(f"Test period: {TEST_START} → {TEST_END} ({n_days} days)")
        print(f"Device     : {config.device}")

    # --- Rollout ---
    daily_revenues = []
    daily_modes    = []    # list of [ch_frac, dc_frac, id_frac] per day
    daily_socs     = []    # mean SoC per day
    soc_violations = 0

    for day_idx in range(n_days):
        obs, _ = env.reset(options={"day_idx": day_idx})
        day_revenue  = 0.0
        mode_counts  = {0: 0, 1: 0, 2: 0}
        socs         = []
        done         = False

        while not done:
            action = agent.select_action(obs, deterministic=True)
            obs, _reward, terminated, truncated, info = env.step(action)

            # Actual $ revenue = p.u. energy_revenue × P_max
            day_revenue += info["energy_revenue"] * config.p_max
            mode_counts[info["mode"]] += 1
            socs.append(info["soc"])
            if info["soc_violated"]:
                soc_violations += 1
            done = terminated or truncated

        daily_revenues.append(day_revenue)
        total_steps = sum(mode_counts.values())
        daily_modes.append([
            mode_counts[0] / total_steps,
            mode_counts[1] / total_steps,
            mode_counts[2] / total_steps,
        ])
        daily_socs.append(np.mean(socs))

    # --- Aggregate ---
    revenues = np.array(daily_revenues)
    modes    = np.array(daily_modes)   # (n_days, 3)

    avg_daily   = revenues.mean()
    std_daily   = revenues.std()
    median_daily = np.median(revenues)
    best_day    = revenues.max()
    worst_day   = revenues.min()

    capture_rate     = avg_daily / PERFECT_FORESIGHT_DAILY * 100
    tbx_capture_rate = avg_daily / TBEX_DAILY * 100

    if verbose:
        print(f"\n{'─'*55}")
        print(f"  Daily Revenue (actual $)")
        print(f"{'─'*55}")
        print(f"  Mean              ${avg_daily:>8.2f}/day")
        print(f"  Median            ${median_daily:>8.2f}/day")
        print(f"  Std dev           ${std_daily:>8.2f}/day")
        print(f"  Best day          ${best_day:>8.2f}/day")
        print(f"  Worst day         ${worst_day:>8.2f}/day")
        print(f"{'─'*55}")
        print(f"  Baselines")
        print(f"    TBx rule-based  ${TBEX_DAILY:>8.2f}/day   (vs agent: {tbx_capture_rate:+.1f}%)")
        print(f"    Perfect foresight ${PERFECT_FORESIGHT_DAILY:>6.2f}/day")
        print(f"    Capture rate    {capture_rate:>8.1f}%  of perfect foresight")
        print(f"{'─'*55}")
        print(f"  Mode distribution (avg across days)")
        print(f"    Charge          {modes[:,0].mean()*100:>6.1f}%")
        print(f"    Discharge       {modes[:,1].mean()*100:>6.1f}%")
        print(f"    Idle            {modes[:,2].mean()*100:>6.1f}%")
        print(f"  Avg SoC           {np.mean(daily_socs):>6.2f} MWh  "
              f"(range: {env.soc_min:.1f}–{env.soc_max:.1f})")
        print(f"  SoC violations    {soc_violations}")
        print(f"{'─'*55}\n")

    return {
        "avg_daily_revenue": avg_daily,
        "std_daily_revenue": std_daily,
        "median_daily_revenue": median_daily,
        "best_day": best_day,
        "worst_day": worst_day,
        "capture_rate_pct": capture_rate,
        "tbx_ratio_pct": tbx_capture_rate,
        "avg_mode_charge": modes[:,0].mean(),
        "avg_mode_discharge": modes[:,1].mean(),
        "avg_mode_idle": modes[:,2].mean(),
        "soc_violations": soc_violations,
        "n_days": n_days,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/stage1/checkpoint_step450000.pt",
        help="Path to Stage 1 checkpoint",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--v60", action="store_true",
        help="Use Stage1V60Config (enriched obs: 36-dim TTFE + 18 price features, obs_dim=108)",
    )
    args = parser.parse_args()

    cfg = Stage1V60Config() if args.v60 else Stage1Config()
    if args.device:
        cfg.device = args.device

    evaluate(args.checkpoint, cfg)

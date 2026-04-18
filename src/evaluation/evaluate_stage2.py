"""
Stage 2 checkpoint evaluation on post-RTC+B test set (March 2026).

Reports total $/day, energy $/day, AS $/day, and capture rate vs
post-RTC+B TBx baseline ($361/day).

Usage:
  python -m src.evaluation.evaluate_stage2 --checkpoint checkpoints/stage2/checkpoint_step75000.pt
  python -m src.evaluation.evaluate_stage2 --stage1-baseline   # Stage 1 300k on post-RTC+B
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.env.ercot_env import ERCOTBatteryEnv
from src.models.sac import SACAgent
from src.training.config import Stage1Config, Stage2Config, Stage2V3aConfig

# Post-RTC+B baselines ($/day for 10 MW / 20 MWh battery)
TBEX_DAILY_POST = 361.0
PERFECT_FORESIGHT_DAILY_POST = 763.0

# Post-RTC+B test set — held-out month
TEST_START = "2026-03-01"
TEST_END   = "2026-03-31"

AS_PRODUCTS = ["regup", "regdn", "rrs", "ecrs", "nsrs"]


def evaluate(
    checkpoint_path: str,
    stage: int = 2,
    config=None,
    verbose: bool = True,
    enriched_flat: bool = False,
) -> dict:
    """
    Run deterministic rollout on post-RTC+B test set.

    Parameters
    ----------
    checkpoint_path : str
    stage : int
        1 = evaluate Stage 1 checkpoint in co_optimize mode (baseline comparison)
        2 = evaluate Stage 2 checkpoint
    config : Stage1Config or Stage2Config or None
    verbose : bool

    Returns
    -------
    dict with evaluation metrics
    """
    if config is None:
        config = Stage2Config()

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
        date_range=(TEST_START, TEST_END),
        enriched_flat=enriched_flat,
    )
    n_days = len(env.day_starts)

    agent = SACAgent(
        stage=stage,
        device=config.device,
        n_prices=config.n_prices,
        n_prices_flat=getattr(config, "n_prices_flat", None),
        d_model=config.d_model,
        nhead=config.nhead,
        n_layers=config.n_layers,
        seq_len=config.seq_len,
        static_dim=config.static_dim,
        hidden_dim=config.hidden_dim,
        tau_gumbel=0.1,  # fully annealed → deterministic
    )
    # weights_only_mode=True: skip optimizer states (may not match current config,
    # e.g. Phase B checkpoints have partial ttfe_optimizer param groups)
    agent.load_checkpoint(checkpoint_path, weights_only_mode=True)

    if verbose:
        print(f"\n=== Stage {stage} Evaluation (post-RTC+B) ===")
        print(f"Checkpoint : {checkpoint_path}")
        print(f"Test period: {TEST_START} → {TEST_END} ({n_days} days)")
        print(f"Device     : {config.device}")

    daily_totals = []
    daily_energy = []
    daily_as = []
    daily_modes = []
    daily_soc_violations = []
    daily_as_fracs = {p: [] for p in AS_PRODUCTS}

    for day_idx in range(n_days):
        obs, _ = env.reset(options={"day_idx": day_idx})
        day_energy_pU = 0.0  # p.u. (needs ×P_max for actual $)
        day_as_usd = 0.0     # already in actual $
        day_modes = [0, 0, 0]
        day_violations = 0
        day_as_fracs = {p: [] for p in AS_PRODUCTS}
        done = False

        while not done:
            action = agent.select_action(obs, deterministic=True)
            # Stage 1 agent outputs 4D; co_optimize env needs 9D — pad AS dims with zeros
            if stage == 1 and len(action) == 4:
                action = np.concatenate([action, np.zeros(5, dtype=action.dtype)])
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # energy_revenue is in p.u. (energy_mag * lmp * η * Δt); ×P_max = actual $
            # as_revenue is already in actual $ (MW * $/MWh * Δt)
            day_energy_pU += info["energy_revenue"]
            day_as_usd += info["as_revenue"]
            day_modes[info["mode"]] += 1
            if info["soc_violated"]:
                day_violations += 1

            if "projected_action" in info and len(info["projected_action"]) >= 6:
                as_mw = info["projected_action"][1:6]
                for i, prod in enumerate(AS_PRODUCTS):
                    day_as_fracs[prod].append(float(as_mw[i]) / config.p_max)

        day_energy_usd = day_energy_pU * config.p_max
        day_total_usd = day_energy_usd + day_as_usd

        daily_totals.append(day_total_usd)
        daily_energy.append(day_energy_usd)
        daily_as.append(day_as_usd)
        total_steps = sum(day_modes)
        daily_modes.append([m / max(total_steps, 1) for m in day_modes])
        daily_soc_violations.append(day_violations)
        for prod in AS_PRODUCTS:
            daily_as_fracs[prod].append(np.mean(day_as_fracs[prod]) if day_as_fracs[prod] else 0.0)

    mean_total = np.mean(daily_totals)
    median_total = np.median(daily_totals)
    mean_energy = np.mean(daily_energy)
    mean_as = np.mean(daily_as)
    total_violations = sum(daily_soc_violations)
    avg_modes = np.mean(daily_modes, axis=0)

    if verbose:
        sep = "─" * 55
        print(f"\n{sep}")
        print(f"  Daily Revenue (actual $)")
        print(sep)
        print(f"  Mean (total)      $ {mean_total:>8.2f}/day")
        print(f"    Energy + EMA    $ {mean_energy:>8.2f}/day")
        print(f"    AS revenue      $ {mean_as:>8.2f}/day")
        print(f"  Median            $ {median_total:>8.2f}/day")
        print(f"  Std dev           $ {np.std(daily_totals):>8.2f}/day")
        print(f"  Best day          $ {max(daily_totals):>8.2f}/day")
        print(f"  Worst day         $ {min(daily_totals):>8.2f}/day")
        print(sep)
        print(f"  Baselines (post-RTC+B)")
        print(f"    TBx rule-based  $ {TBEX_DAILY_POST:>8.2f}/day   "
              f"(agent: {100*mean_total/TBEX_DAILY_POST:+.1f}%)")
        print(f"    Perfect foresight $ {PERFECT_FORESIGHT_DAILY_POST:.2f}/day")
        print(f"    Capture rate      {100*mean_total/PERFECT_FORESIGHT_DAILY_POST:.1f}%  "
              f"of perfect foresight")
        print(sep)
        print(f"  Mode distribution (avg across days)")
        print(f"    Charge          {100*avg_modes[0]:.1f}%")
        print(f"    Discharge       {100*avg_modes[1]:.1f}%")
        print(f"    Idle            {100*avg_modes[2]:.1f}%")
        print(f"  AS utilization (mean fraction of P_max committed)")
        for prod in AS_PRODUCTS:
            mean_frac = np.mean(daily_as_fracs[prod])
            print(f"    {prod:<6}          {mean_frac:.3f}")
        print(f"  SoC violations    {total_violations}")
        print(sep)

    return {
        "mean_total": mean_total,
        "median_total": median_total,
        "mean_energy": mean_energy,
        "mean_as": mean_as,
        "capture_rate": mean_total / PERFECT_FORESIGHT_DAILY_POST,
        "soc_violations": total_violations,
        "avg_modes": avg_modes.tolist(),
        "as_fracs": {p: float(np.mean(daily_as_fracs[p])) for p in AS_PRODUCTS},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2 Evaluation")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/stage2/checkpoint_final.pt",
        help="Path to Stage 2 checkpoint",
    )
    parser.add_argument(
        "--stage1-baseline",
        action="store_true",
        help="Evaluate Stage 1 v5.9 300k checkpoint on post-RTC+B test set",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--v3a", action="store_true",
        help="Use Stage2V3aConfig (enriched flat obs, static_dim=32, obs_dim=108)",
    )
    args = parser.parse_args()

    cfg = Stage2V3aConfig() if args.v3a else Stage2Config()
    if args.device:
        cfg.device = args.device

    if args.stage1_baseline:
        s1_ckpt = "checkpoints/stage1/checkpoint_step300000.pt"
        print(f"[Stage 1 baseline] Evaluating {s1_ckpt} on post-RTC+B test set")
        evaluate(s1_ckpt, stage=1, config=cfg)
    else:
        evaluate(args.checkpoint, stage=2, config=cfg, enriched_flat=args.v3a)

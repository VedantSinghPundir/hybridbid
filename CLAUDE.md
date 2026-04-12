# CLAUDE.md — TempDRL for ERCOT RTC+B
**Date:** April 7, 2026
**Project:** Temporal-Aware Deep Reinforcement Learning for Battery Storage Bidding in ERCOT's Post-RTC+B Market
**Status:** Paper-spec reset complete (v5). 89 tests passing. Ready to launch Stage 1 training.

---

## CRITICAL INSTRUCTIONS FOR CLAUDE CODE

1. **Work incrementally.** Do NOT attempt to build the full system at once. Build one module, test it, confirm, then proceed.
2. **Two-stage training is the core architectural decision.** Do not mix pre-RTC+B and post-RTC+B data in training. Stage 1 uses pre-RTC+B only, Stage 2 uses post-RTC+B only.
3. **Ask before building.** When encountering ambiguity, stop and report rather than guessing.
4. **This CLAUDE.md is the single source of truth.** All prior versions are obsolete.
5. **Era-aware features (is_post_rtcb, days_since_rtcb, rt_as_available) are REMOVED from the observation space.** The structural break is handled by the two-stage training schedule, not observation flags.
6. **Explore and report first.** Surface findings before building. Never attempt to build the full system at once.

---

## WHAT WE'RE BUILDING

Adapting the TempDRL approach from Li et al. (2024) for ERCOT's post-RTC+B market, with a **two-stage pretrain → finetune** training strategy to handle the RTC+B structural break.

> **Li, J., Wang, C., Zhang, Y., & Wang, H.** "Temporal-Aware Deep Reinforcement Learning for Energy Storage Bidding in Energy and Contingency Reserve Markets." *IEEE Transactions on Energy Markets, Policy and Regulation*, Sept 2024. (arXiv:2402.19110)

**Target users:** Small battery storage operators (5-20 MW) in ERCOT.

---

## ARCHITECTURE — Paper-Spec Reset (April 7, 2026)

The implementation was reset to match Li et al. as closely as possible. Previous deviations (continuous-only action space, weak EMA bonus, reward scaling, price normalization, alpha floor) caused cascading training failures. All compensatory additions have been removed.

### Action Space (Matches Li et al. Eq. 1, 23)
- **3-class Gumbel-Softmax mode:** charge / discharge / idle
- **Continuous magnitude:** energy bid power, squashed Gaussian, scaled to [0, P_max]
- Stage 1 action dim: 4 (3 mode logits + 1 magnitude)
- Stage 2 action dim: 9 (3 mode logits + 1 energy magnitude + 5 AS magnitudes)
- Gumbel temperature anneals from 1.0 → 0.1 over training
- During evaluation: argmax (hard mode selection)

### Observation Space (90-dim, matches Li et al. Eq. 22)
- TTFE output: 64-dim (global average pooling, Eq. 21)
- Raw current prices: 12-dim (RT LMP + 5 RT MCPC + DAM SPP + 5 DAM AS)
- System conditions: 7-dim (load forecast, actual load, wind, solar, 3 ERCOT indicators)
- Cyclical time: 6-dim (hour sin/cos, day-of-week sin/cos, month sin/cos)
- SoC: 1-dim

### TTFE
- Input: 12-dim price vectors, rolling window L=32
- Feature embedding → 2 stacked MHA layers (8 heads each) → global average pooling
- Output: 64-dim feature vector (F'=64)
- Shared across actor and critic, updated via gradient descent from both

### Reward Function (Matches Li et al. Eq. 24–30)
```
r_t = r_S_t    (Stage 1, energy only)
r_t = r_S_t + r_fast_t + r_slow_t + r_delay_t    (Stage 2, joint)
```

Spot reward (Eq. 26):
```
r_S = a_S * ρ_S * (v_dch * η_dch - v_ch / η_ch)
    + β_S * a_S * |ρ_S - ρ̄_S| * (I_dch * v_dch * η_dch + I_ch * v_ch / η_ch)
```
- EMA: ρ̄_S_t = τ_S * ρ̄_S_{t-1} + (1 - τ_S) * ρ_S_t
- I_ch = sgn(ρ̄ - ρ), I_dch = sgn(ρ - ρ̄)
- **τ_S = 0.9, β_S = 10** (paper values, NOT the old 0.95/0.5)
- NO degradation cost in step reward (not in Eq. 30)
- NO reward scaling

### Constraints
- **Feasibility projection:** clips actions to keep SoC within [SoC_min, SoC_max], preserves gradients
- **Episode termination:** -50 penalty when SoC violates energy limits, terminated=True
- Both mechanisms coexist (projection catches most, termination penalizes edge cases)

### SAC Configuration
- SAC v2 (twin Q, no separate V network)
- γ = 0.99, lr = 0.0003 (actor, critic, TTFE)
- τ_ψ = 0.01 (target network smoothing)
- target_entropy = log(3) - 1
- NO alpha_min floor
- Gradient clipping: max_norm=1.0 (only stability addition retained)

### Battery Parameters
- P_max: 10 MW, E_max: 20 MWh
- η_ch = η_dch = 0.95 (paper value)
- SoC limits: 10% – 90% (2.0 – 18.0 MWh)
- Initial SoC: fixed 50% (10.0 MWh), no randomization

### What Was REMOVED in This Reset
- Continuous-only action space (replaced by Gumbel-Softmax mode)
- Reward scaling (×0.001)
- Price normalization (÷100 for TTFE input)
- α_min = 0.01 floor
- Degradation cost in step reward
- randomize_initial_soc
- Old EMA parameters (τ=0.95, β=0.5)

---

## DATA

### Processed Data (on M4, `data/processed/`)
- 75 monthly Parquet files × 3 tables (prices, system, load_forecast)
- 654,048 rows total, 5-minute resolution
- 12-dim TTFE price vector: RT LMP, 5 RT MCPC, DAM SPP, 5 DAM AS
- load_forecast verified real through 2026-03
- Tarball also on Narnia at `~/processed_data.tar.gz`

### Temporal Split
- Stage 1 training: 2020-01-01 → 2023-12-31
- Stage 1 validation: 2024-01-01 → 2025-09-30
- Stage 1 test (pre-RTC+B): 2025-10-01 → 2025-12-04
- Stage 2 (post-RTC+B): 2025-12-05 → present (~30k transitions)

### Known Data Issue
- ECRS product launched June 2023 — NaN before that date
- RT SPP bulk files are 15-min resolution; use NP6-788-CD for true 5-min RT LMP

---

## BASELINES (computed, in `data/results/`)

| Baseline | Pre-RTC+B ($/day) | Post-RTC+B ($/day) |
|----------|-------------------|-------------------|
| TBx (rule-based) | 870 | 361 |
| Perfect Foresight MIP | 1,519 | 763 |

---

## TRAINING HISTORY

| Run | Key Change | Result |
|-----|-----------|--------|
| v1 (Narnia) | First attempt, no stability fixes | NaN crash at 118k steps |
| v2 (M4) | Added grad clipping, reward scale, price norm | SoC pinned at floor, alpha collapse |
| v3 (M4) | Added α_min, EMA bonus (weak: β=0.5), audit fixes | 86% early termination, 41-step avg episodes |
| v4 (M4) | Removed early termination | SoC still pinned at 2.1, "dump and idle" |
| **v5 (pending)** | **Paper-spec reset: Gumbel-Softmax, β=10, termination restored** | **Ready to launch** |

---

## TWO-STAGE TRAINING PLAN

### Stage 1: Pre-RTC+B Pretraining
- 500k steps on 2020–2023 data, energy-only
- Goal: build robust TTFE representations + learn energy arbitrage cycling
- Replay buffer: 1M transitions, batch size: 256
- Checkpoints every 50k steps

### Stage 2: Post-RTC+B Fine-tuning (after Stage 1 succeeds)
- ~30k transitions, batch size: 128
- Action space expands from 4D to 9D (add 5 AS magnitude heads)
- Progressive TTFE unfreezing: frozen → top 1-2 layers at 10× lower LR → optional full unfreeze
- Critics: re-initialized fresh (old Q-values invalid for new reward structure)
- Actor: energy head from Stage 1, AS heads initialized near-zero
- **ERCOT adaptation needed:** AS revenue must be decoupled from v_ch/v_dch (unlike Li et al., ERCOT allows AS availability while idle)

---

## EVALUATION PLAN (3 dimensions)

1. **Baseline comparison:** TBx, Perfect Foresight MIP, vanilla SAC without TTFE
2. **Train-from-scratch comparison:** Fresh agent on post-RTC+B data only vs two-stage agent
3. **Ablation study:** Progressive unfreezing vs full fine-tune vs full freeze; fresh critics vs warm-started; near-zero AS init vs random

---

## CODE TREE

```
hybridbid/
├── CLAUDE.md
├── configs/
│   └── default.yaml
├── data/
│   ├── processed/          # 75 monthly Parquet files × 3 tables
│   ├── raw/                # ERCOT API downloads
│   └── results/            # Baseline results
├── src/
│   ├── env/
│   │   └── ercot_env.py    # Gymnasium env, Eq. 26 reward, Gumbel action interpretation
│   ├── models/
│   │   ├── sac.py          # SAC v2, twin Q, Gumbel-Softmax support
│   │   └── networks.py     # Actor (Gumbel mode + Gaussian magnitude), Critic, TTFE
│   ├── training/
│   │   ├── config.py       # All hyperparameters
│   │   ├── train_stage1.py # Full training loop with Gumbel annealing
│   │   └── train_stage2.py # Fine-tuning (to be updated for Stage 2)
│   ├── evaluation/
│   │   ├── evaluate.py
│   │   └── baselines.py    # TBx + Perfect Foresight MIP
│   └── data/
│       └── preprocessing.py
├── tests/                  # 89 tests passing
├── checkpoints/
│   ├── stage1_v3/          # Archived v3 run
│   └── stage1_v4/          # Archived v4 run
├── logs/
│   ├── stage1_train_v3.log
│   └── stage1_train_v4.log
└── requirements.txt        # STALE — needs updating
```

---

## MACHINES

- **M4 MacBook** (`karthikmattu@100.113.39.50`) — primary dev, project at `~/hybridbid`, Stage 1 training runs here (MPS)
- **MacBook Air** (`karthikmattu@100.99.63.48`) — background tasks
- **Narnia** (`km5503@narnia.gccis.rit.edu`, GPU node 18, CUDA A16) — fallback for longer training runs. Has processed data tarball. Repo needs paper-spec reset pushed before use.
- Connected via Tailscale. SSH M4→Narnia works. SSH M4→Air does NOT.

---

## EXPLICITLY DEFERRED

- 10-point EB/OC bid curve output
- DreamerV3 / model-based RL comparison
- Cross-market transfer learning
- MIP with full AS co-optimization
- LLM context module
- Meta-controller / hybrid routing
- Predict-and-Optimize benchmark
- Era-aware features (replaced by two-stage training)

---

## OPEN ITEMS

- Two papers outside 5-year window (Howard & Ruder 2018, Ash & Adams 2020) — pending instructor confirmation
- Van de Ven et al. (2024) is book chapter / arXiv — check if instructor requires peer-reviewed
- `requirements.txt` is stale (~15 packages listed, ~250+ installed)
- Git: paper-spec reset changes are LOCAL on M4, not committed/pushed
- Narnia repo is out of date — needs push before it can run v5
- Mid-project checkpoint (submitted Apr 5) describes pre-reset architecture — final paper needs updating

---

## KEY REFERENCES

1. **Li et al. (2024)** — arXiv:2402.19110. **Primary implementation reference.** All architecture and reward decisions should trace back to this paper.
2. **ERCOT RTC+B Battery Overview** — ercot.com
3. **ErcotAPI** — Primary data source (covers 2020+)

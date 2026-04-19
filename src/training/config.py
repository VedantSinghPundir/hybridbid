"""
Hyperparameter configuration for TempDRL training.

Values aligned with Li et al. (2024) Table I where specified.
"""

from dataclasses import dataclass, field

import torch


def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class TrainConfig:
    """Base training hyperparameters (Li et al. Table I defaults)."""

    # Data
    data_dir: str = "data/processed"
    seq_len: int = 32
    n_prices: int = 12
    static_dim: int = 14

    # TTFE — Li et al.: N_MHA=2, h=8
    d_model: int = 64
    nhead: int = 8       # paper: 8 heads
    n_layers: int = 2

    # Networks
    hidden_dim: int = 256

    # SAC — Li et al. Table I
    gamma: float = 0.99
    tau: float = 0.005   # τ_ψ target network smoothing (standard SAC default; 0.01 tracks volatile critic too closely under heavy-tailed rewards)

    # Gradient clipping (not in paper, kept as numerical safety)
    max_grad_norm: float = 1.0

    # Gumbel-Softmax temperature annealing
    tau_gumbel_init: float = 1.0   # start temperature
    tau_gumbel_final: float = 0.1  # end temperature

    # Battery — Li et al. uses η=0.95 for both
    p_max: float = 10.0
    e_max: float = 20.0
    soc_min_frac: float = 0.10
    soc_max_frac: float = 0.90
    soc_initial_frac: float = 0.50
    eta_ch: float = 0.95   # paper: 0.95
    eta_dch: float = 0.95  # paper: 0.95
    degradation_cost: float = 2.0  # kept for info tracking, not used in step reward

    # Device — auto-detect CUDA > MPS > CPU
    device: str = field(default_factory=_detect_device)


@dataclass
class Stage1Config(TrainConfig):
    """Stage 1: Energy-only pretraining."""

    # Training
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4   # v5.3: reverted from 1e-4 — lower LR worsened mode collapse in v5.2
    lr_ttfe: float = 3e-4
    buffer_capacity: int = 1_000_000
    batch_size: int = 256
    total_steps: int = 1_000_000  # v5.2: extended from 500k
    warmup_steps: int = 1000
    updates_per_step: int = 1

    # Data range
    train_start: str = "2020-01-01"
    train_end: str = "2023-12-31"

    # Logging / Checkpoint
    log_interval: int = 1_000
    checkpoint_dir: str = "checkpoints/stage1"
    save_every: int = 25_000   # v5.2: finer granularity from 50k


@dataclass
class Stage1V60Config(Stage1Config):
    """Stage 1 v6.0: Enriched observations with DAM LMPs + engineered price features.

    Key changes from v5.9:
      - n_prices=36: TTFE input expanded to 12 orig + 24 hourly DA LMP values
      - n_prices_flat=12: only original 12 dims used in flat obs (no double-count)
      - static_dim=32: 14 original + 18 engineered price features
      - obs_dim: 64 + 12 + 32 = 108 (was 64 + 12 + 14 = 90)
      - checkpoint_dir: checkpoints/stage1_v60
      All training dynamics identical to v5.9 (symlog, SAC v2, τ=0.005, etc.)
    """
    n_prices: int = 36        # TTFE input: 12 orig + 24 DA LMP
    n_prices_flat: int = 12   # flat obs: first 12 dims only (orig prices)
    static_dim: int = 32      # 14 orig + 18 engineered price features
    checkpoint_dir: str = "checkpoints/stage1_v60"
    save_every: int = 50_000  # checkpoints every 50k (same granularity as v5.9)


@dataclass
class Stage2Config(TrainConfig):
    """Stage 2: Post-RTC+B co-optimization fine-tuning (stage2_v2).

    Key changes from v1:
      - target_entropy corrected to 5.0 in SACAgent (9D action space)
      - Alpha floor at 0.05 (log_alpha.clamp_(min=log(0.05)))
      - Phase C removed — two-phase only (A: 0–40%, B: 40–100%)
      - Gumbel τ: 0.8→0.5 in Phase A, hold 0.5 in Phase B
      - Total steps reduced from 150k to 120k (Phase C consumed wasted steps)
    """

    # Training
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_ttfe: float = 3e-5   # 10× lower for TTFE unfreezing phases
    buffer_capacity: int = 60_000
    batch_size: int = 256
    total_steps: int = 120_000
    warmup_steps: int = 5_000
    updates_per_step: int = 1

    # Gumbel temperature — higher start encourages mode exploration while AS heads learn
    tau_gumbel_init: float = 0.8   # Phase A start; anneals to 0.5 at Phase A end
    tau_gumbel_final: float = 0.5  # Phase B holds at 0.5 (no further annealing)

    # Two-phase TTFE unfreezing (Phase C removed — destabilized pretrained representations)
    phase_b_start_frac: float = 0.40   # Phase A: 0–40% frozen TTFE (steps 0–47999)
                                        # Phase B: 40–100% unfreeze top layer (steps 48000–119999)

    # Data range — post-RTC+B train; test = March 2026 (held out)
    train_start: str = "2025-12-05"
    train_end: str = "2026-02-28"

    # Stage 1 best checkpoint (v5.9 300k, $267/day)
    stage1_checkpoint: str = "checkpoints/stage1/checkpoint_step300000.pt"

    # Logging / Checkpoint
    log_interval: int = 1_000
    checkpoint_dir: str = "checkpoints/stage2_v2"
    save_every: int = 5_000


@dataclass
class Stage2V3aConfig(Stage2Config):
    """Stage 2 v3a: Enriched flat observation (18 price features), TTFE unchanged.

    Key changes from v2:
      - static_dim: 14 → 32 (adds 18 engineered price features to flat obs only)
      - TTFE input stays 12-dim — loads perfectly from v5.9 300k checkpoint
      - obs_dim: 90 → 108 (64 TTFE + 12 raw prices + 32 static)
      - Actor fc1 (90→256) cannot load from v5.9 (dim mismatch) — fresh init
      - Actor fc2 + mode/energy heads: copied from v5.9
      - total_steps: 200k (extended for broader search given larger obs space)
      - phase_b_start_frac: 0.40 → Phase B at step 80k
      - checkpoint_dir: checkpoints/stage2_v3a
    """
    static_dim: int = 32         # 14 orig + 18 engineered price features
    total_steps: int = 200_000   # extended from 120k
    checkpoint_dir: str = "checkpoints/stage2_v3a"


@dataclass
class Stage2V60Config(Stage2Config):
    """Stage 2 compatible with Stage 1 v6.0 (36-dim TTFE input).
    
    Loads Stage 1 v6.0 checkpoint and fine-tunes on post-RTC+B data.
    TTFE input stays 36-dim to match v6.0 weights.
    """
    n_prices: int = 36           # match Stage 1 v6.0 TTFE input
    n_prices_flat: int = 12      # flat obs uses only original 12 dims
    static_dim: int = 32         # 14 orig + 18 engineered price features
    total_steps: int = 200_000
    checkpoint_dir: str = "checkpoints/stage2_v60"
    stage1_checkpoint: str = "checkpoints/stage1_v60/checkpoint_final.pt"

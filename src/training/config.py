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

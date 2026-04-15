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
    """Stage 2: Co-optimization finetuning."""

    # Training
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_ttfe: float = 3e-5  # 10x lower for TTFE
    buffer_capacity: int = 50_000
    batch_size: int = 128
    total_steps: int = 30_000
    warmup_steps: int = 500
    updates_per_step: int = 1

    # Progressive unfreezing
    phase1_steps: int = 10_000  # Frozen TTFE
    phase2_steps: int = 20_000  # Unfreeze top layers

    # Data range
    train_start: str = "2025-12-05"
    train_end: str = "2026-01-31"

    # Stage 1 checkpoint
    stage1_checkpoint: str = "checkpoints/stage1/checkpoint.pt"

    # Checkpoint
    checkpoint_dir: str = "checkpoints/stage2"
    save_every: int = 10_000

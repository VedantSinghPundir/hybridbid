"""
Hyperparameter configuration for TempDRL training.
"""

from dataclasses import dataclass


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    # Data
    data_dir: str = "data/processed"
    seq_len: int = 32
    n_prices: int = 12
    static_dim: int = 14

    # TTFE
    d_model: int = 64
    nhead: int = 4
    n_layers: int = 2

    # Networks
    hidden_dim: int = 256

    # SAC
    gamma: float = 0.99
    tau: float = 0.005

    # Battery
    p_max: float = 10.0
    e_max: float = 20.0
    soc_min_frac: float = 0.10
    soc_max_frac: float = 0.90
    soc_initial_frac: float = 0.50
    eta_ch: float = 0.92
    eta_dch: float = 0.92
    degradation_cost: float = 2.0

    # Device
    device: str = "cpu"


@dataclass
class Stage1Config(TrainConfig):
    """Stage 1: Energy-only pretraining."""

    # Training
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_ttfe: float = 3e-4
    buffer_capacity: int = 1_000_000
    batch_size: int = 256
    total_steps: int = 500_000
    warmup_steps: int = 1000
    updates_per_step: int = 1

    # Data range
    train_start: str = "2026-01-07"
    train_end: str = "2026-01-12"

    # Checkpoint
    checkpoint_dir: str = "checkpoints/stage1"
    save_every: int = 50_000


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
    train_start: str = "2026-01-07"
    train_end: str = "2026-01-12"

    # Stage 1 checkpoint
    stage1_checkpoint: str = "checkpoints/stage1/checkpoint.pt"

    # Checkpoint
    checkpoint_dir: str = "checkpoints/stage2"
    save_every: int = 10_000

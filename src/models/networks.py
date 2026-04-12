"""
Actor and Critic networks for SAC.

Actor: Hybrid discrete-continuous policy.
  - Mode selection (charge/discharge/idle) via Gumbel-Softmax
  - Continuous magnitude via squashed Gaussian
  Matches Li et al. (2024) Eq. 1, 23.

Critic: Twin Q-networks for clipped double-Q learning.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_STD_MIN = -20
LOG_STD_MAX = 2

# Mode indices
MODE_CHARGE = 0
MODE_DISCHARGE = 1
MODE_IDLE = 2


class Actor(nn.Module):
    """
    Hybrid discrete-continuous actor for SAC.

    Discrete part: 3-class mode (charge / discharge / idle) via Gumbel-Softmax.
    Continuous part: energy magnitude [squashed Gaussian in (-1,1)].
    Stage 2 extension: additional AS magnitudes (n_as_dims=5).

    Total action dim = 3 + 1 + n_as_dims
      Stage 1: 4  (3 mode + 1 energy mag)
      Stage 2: 9  (3 mode + 1 energy mag + 5 AS mags)

    Input:  90-dim (TTFE 64 + current prices 12 + static 14)
    """

    def __init__(self, obs_dim: int = 90, n_as_dims: int = 0, hidden_dim: int = 256):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_as_dims = n_as_dims
        self.action_dim = 3 + 1 + n_as_dims  # total flattened action

        # Shared trunk
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Mode head: 3 logits (charge / discharge / idle)
        self.mode_head = nn.Linear(hidden_dim, 3)

        # Energy magnitude head
        self.energy_mag_mean_head = nn.Linear(hidden_dim, 1)
        self.energy_mag_log_std_head = nn.Linear(hidden_dim, 1)

        # AS magnitude heads (Stage 2 only)
        if n_as_dims > 0:
            self.as_mag_mean_head = nn.Linear(hidden_dim, n_as_dims)
            self.as_mag_log_std_head = nn.Linear(hidden_dim, n_as_dims)

    def forward(self, obs: torch.Tensor):
        """
        Returns
        -------
        mode_logits      : (batch, 3)
        energy_mag_mean  : (batch, 1)
        energy_mag_log_std : (batch, 1)
        [as_mag_mean     : (batch, n_as_dims)]  — only when n_as_dims > 0
        [as_mag_log_std  : (batch, n_as_dims)]
        """
        h = F.relu(self.fc1(obs))
        h = F.relu(self.fc2(h))

        mode_logits = self.mode_head(h)
        energy_mag_mean = self.energy_mag_mean_head(h)
        energy_mag_log_std = torch.clamp(
            self.energy_mag_log_std_head(h), LOG_STD_MIN, LOG_STD_MAX
        )

        if self.n_as_dims > 0:
            as_mag_mean = self.as_mag_mean_head(h)
            as_mag_log_std = torch.clamp(
                self.as_mag_log_std_head(h), LOG_STD_MIN, LOG_STD_MAX
            )
            return mode_logits, energy_mag_mean, energy_mag_log_std, as_mag_mean, as_mag_log_std

        return mode_logits, energy_mag_mean, energy_mag_log_std

    def sample(self, obs: torch.Tensor, tau: float = 1.0, hard: bool = False):
        """
        Sample action via Gumbel-Softmax (mode) + reparameterization (magnitudes).

        Parameters
        ----------
        tau : float
            Gumbel-Softmax temperature. Start at 1.0, anneal toward 0.1.
        hard : bool
            If True, use straight-through hard one-hot (for deterministic eval).

        Returns
        -------
        action    : (batch, action_dim) — cat([mode_soft/hard, energy_mag, as_mags?])
        log_prob  : (batch, 1) — sum of mode + magnitude log-probs
        det_action : (batch, action_dim) — deterministic: argmax mode + tanh(mean)
        """
        outputs = self.forward(obs)
        mode_logits = outputs[0]
        energy_mag_mean = outputs[1]
        energy_mag_log_std = outputs[2]

        # --- Mode: Gumbel-Softmax ---
        mode_sample = F.gumbel_softmax(mode_logits, tau=tau, hard=hard)  # (batch, 3)

        # Log-prob of mode: E_{q}[log p(mode)] under categorical
        mode_probs = F.softmax(mode_logits, dim=-1)  # (batch, 3)
        log_prob_mode = (mode_sample * torch.log(mode_probs + 1e-8)).sum(dim=-1, keepdim=True)

        # --- Energy magnitude: squashed Gaussian ---
        energy_std = energy_mag_log_std.exp()
        energy_normal = Normal(energy_mag_mean, energy_std)
        x_energy = energy_normal.rsample()
        energy_mag = torch.tanh(x_energy)  # ∈ (-1, 1)

        log_prob_energy = energy_normal.log_prob(x_energy)
        log_prob_energy -= torch.log(1 - energy_mag.pow(2) + 1e-6)
        log_prob_energy = log_prob_energy.sum(dim=-1, keepdim=True)

        log_prob = log_prob_mode + log_prob_energy
        action_parts = [mode_sample, energy_mag]

        # Deterministic counterparts
        hard_mode = F.one_hot(mode_logits.argmax(dim=-1), 3).float()
        det_energy_mag = torch.tanh(energy_mag_mean)
        det_parts = [hard_mode, det_energy_mag]

        # --- AS magnitudes (Stage 2) ---
        if self.n_as_dims > 0:
            as_mag_mean = outputs[3]
            as_mag_log_std = outputs[4]
            as_std = as_mag_log_std.exp()
            as_normal = Normal(as_mag_mean, as_std)
            x_as = as_normal.rsample()
            as_mag = torch.tanh(x_as)  # ∈ (-1, 1), env takes abs

            log_prob_as = as_normal.log_prob(x_as)
            log_prob_as -= torch.log(1 - as_mag.pow(2) + 1e-6)
            log_prob_as = log_prob_as.sum(dim=-1, keepdim=True)

            log_prob = log_prob + log_prob_as
            action_parts.append(as_mag)
            det_parts.append(torch.tanh(as_mag_mean))

        action = torch.cat(action_parts, dim=-1)
        det_action = torch.cat(det_parts, dim=-1)

        return action, log_prob, det_action

    @classmethod
    def init_stage2_from_stage1(cls, stage1_actor: "Actor", n_as_dims: int = 5) -> "Actor":
        """
        Create a Stage 2 actor initialized from a trained Stage 1 actor.

        - All Stage 1 components (trunk, mode head, energy mag heads) copied exactly.
        - AS magnitude heads initialized near-zero (small Gaussian weights, zero bias).
        """
        actor2 = cls(
            obs_dim=stage1_actor.obs_dim,
            n_as_dims=n_as_dims,
            hidden_dim=stage1_actor.fc1.out_features,
        )

        # Copy all Stage 1 components
        actor2.fc1.load_state_dict(stage1_actor.fc1.state_dict())
        actor2.fc2.load_state_dict(stage1_actor.fc2.state_dict())
        actor2.mode_head.load_state_dict(stage1_actor.mode_head.state_dict())
        actor2.energy_mag_mean_head.load_state_dict(stage1_actor.energy_mag_mean_head.state_dict())
        actor2.energy_mag_log_std_head.load_state_dict(stage1_actor.energy_mag_log_std_head.state_dict())

        # AS heads: near-zero initialization
        with torch.no_grad():
            actor2.as_mag_mean_head.weight.normal_(0, 0.01)
            actor2.as_mag_mean_head.bias.zero_()
            actor2.as_mag_log_std_head.weight.normal_(0, 0.01)
            actor2.as_mag_log_std_head.bias.zero_()

        return actor2


class Critic(nn.Module):
    """Single Q-network: Q(obs, action) -> scalar."""

    def __init__(self, obs_dim: int = 90, action_dim: int = 4, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action], dim=-1)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc3(h)


class TwinCritic(nn.Module):
    """Twin Q-networks for clipped double-Q learning."""

    def __init__(self, obs_dim: int = 90, action_dim: int = 4, hidden_dim: int = 256):
        super().__init__()
        self.q1 = Critic(obs_dim, action_dim, hidden_dim)
        self.q2 = Critic(obs_dim, action_dim, hidden_dim)

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        """Returns (Q1, Q2) both as (batch, 1)."""
        return self.q1(obs, action), self.q2(obs, action)

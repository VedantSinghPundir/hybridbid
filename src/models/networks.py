"""
Actor and Critic networks for SAC.

Actor: Squashed Gaussian policy (outputs mean + log_std for reparameterization).
Critic: Twin Q-networks for clipped double-Q learning.
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_STD_MIN = -20
LOG_STD_MAX = 2


class Actor(nn.Module):
    """
    Squashed Gaussian actor for SAC.

    Input:  78-dim (TTFE 64 + system 7 + time 6 + SoC 1)
    Output: action_dim mean and log_std for reparameterized sampling
    """

    def __init__(self, obs_dim: int = 78, action_dim: int = 1, hidden_dim: int = 256):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs: torch.Tensor):
        """Return mean, log_std for the squashed Gaussian."""
        h = F.relu(self.fc1(obs))
        h = F.relu(self.fc2(h))
        mean = self.mean_head(h)
        log_std = self.log_std_head(h)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs: torch.Tensor):
        """
        Sample action via reparameterization trick with tanh squashing.

        Returns
        -------
        action : Tensor (batch, action_dim) in [-1, 1]
        log_prob : Tensor (batch, 1)
        mean : Tensor (batch, action_dim) — deterministic action (tanh of mean)
        """
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        normal = Normal(mean, std)
        # Reparameterization trick
        x_t = normal.rsample()
        action = torch.tanh(x_t)

        # Log-prob with tanh correction
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob, torch.tanh(mean)

    @classmethod
    def init_stage2_from_stage1(cls, stage1_actor: "Actor", action_dim: int = 6) -> "Actor":
        """
        Create a 6D actor initialized from a trained 1D Stage 1 actor.

        - Hidden layers copied exactly
        - Energy dim (row 0) of output copied from Stage 1
        - AS dims (rows 1-5) initialized with small Gaussian weights
        """
        actor2 = cls(obs_dim=stage1_actor.obs_dim, action_dim=action_dim,
                      hidden_dim=stage1_actor.fc1.out_features)

        # Copy hidden layers
        actor2.fc1.load_state_dict(stage1_actor.fc1.state_dict())
        actor2.fc2.load_state_dict(stage1_actor.fc2.state_dict())

        # Initialize mean head
        with torch.no_grad():
            # Energy dim from Stage 1
            actor2.mean_head.weight[0] = stage1_actor.mean_head.weight[0]
            actor2.mean_head.bias[0] = stage1_actor.mean_head.bias[0]
            # AS dims: small Gaussian
            actor2.mean_head.weight[1:].normal_(0, 0.01)
            actor2.mean_head.bias[1:].zero_()

            # Initialize log_std head
            actor2.log_std_head.weight[0] = stage1_actor.log_std_head.weight[0]
            actor2.log_std_head.bias[0] = stage1_actor.log_std_head.bias[0]
            actor2.log_std_head.weight[1:].normal_(0, 0.01)
            actor2.log_std_head.bias[1:].zero_()

        return actor2


class Critic(nn.Module):
    """Single Q-network: Q(obs, action) -> scalar."""

    def __init__(self, obs_dim: int = 78, action_dim: int = 1, hidden_dim: int = 256):
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

    def __init__(self, obs_dim: int = 78, action_dim: int = 1, hidden_dim: int = 256):
        super().__init__()
        self.q1 = Critic(obs_dim, action_dim, hidden_dim)
        self.q2 = Critic(obs_dim, action_dim, hidden_dim)

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        """Returns (Q1, Q2) both as (batch, 1)."""
        return self.q1(obs, action), self.q2(obs, action)

"""
Soft Actor-Critic agent with TTFE encoder.

Two-stage architecture:
  Stage 1: energy-only (4D action: 3 mode + 1 energy_mag), pretrain on pre-RTC+B
  Stage 2: co-optimize (9D action: 3 mode + 1 energy_mag + 5 AS_mags), finetune on post-RTC+B

Action format matches Li et al. (2024) with Gumbel-Softmax discrete mode selection.
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.ttfe import TTFE
from src.models.networks import Actor, TwinCritic
from src.models.replay_buffer import ReplayBuffer


class SACAgent:
    """
    SAC agent encapsulating TTFE + Actor + TwinCritic + target networks.

    Parameters
    ----------
    stage : int
        1 for energy-only, 2 for co-optimize.
    device : str
        'cpu', 'cuda', or 'mps'.
    tau_gumbel : float
        Initial Gumbel-Softmax temperature (anneal from 1.0 → 0.1 during training).
    """

    def __init__(
        self,
        stage: int = 1,
        device: str = "cpu",
        n_prices: int = 12,
        d_model: int = 64,
        nhead: int = 8,
        n_layers: int = 2,
        seq_len: int = 32,
        static_dim: int = 14,
        hidden_dim: int = 256,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        lr_ttfe: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.01,
        buffer_capacity: int = None,
        batch_size: int = None,
        max_grad_norm: float = 1.0,
        tau_gumbel: float = 1.0,
    ):
        self.stage = stage
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.max_grad_norm = max_grad_norm
        self.tau_gumbel = tau_gumbel

        # Action dimensions
        # Stage 1: 3 mode + 1 energy_mag = 4
        # Stage 2: 3 mode + 1 energy_mag + 5 AS_mags = 9
        self.n_as_dims = 0 if stage == 1 else 5
        self.action_dim = 3 + 1 + self.n_as_dims  # 4 or 9
        self.n_continuous = 1 + self.n_as_dims     # 1 or 6 (continuous dims only)

        self.n_prices = n_prices
        self.obs_dim = d_model + n_prices + static_dim  # 64 + 12 + 14 = 90

        # Default buffer/batch sizes per stage
        if buffer_capacity is None:
            buffer_capacity = 1_000_000 if stage == 1 else 50_000
        if batch_size is None:
            batch_size = 256 if stage == 1 else 128
        self.batch_size = batch_size

        # Networks
        self.ttfe = TTFE(n_prices=n_prices, d_model=d_model, nhead=nhead,
                         n_layers=n_layers, seq_len=seq_len).to(device)
        self.actor = Actor(obs_dim=self.obs_dim, n_as_dims=self.n_as_dims,
                           hidden_dim=hidden_dim).to(device)
        self.critic = TwinCritic(obs_dim=self.obs_dim, action_dim=self.action_dim,
                                 hidden_dim=hidden_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic).to(device)

        for p in self.critic_target.parameters():
            p.requires_grad = False

        # Target entropy: log(3) for discrete mode only.
        # Li et al. specifies log(3) - n_continuous, but that yields ~0.099 (near-zero),
        # causing alpha to collapse in ~15k steps and mode diversity to die.
        # Using log(3) ≈ 1.099 forces the policy to maintain spread across all 3 modes.
        self.target_entropy = float(np.log(3))
        self.log_alpha = torch.zeros(1, device=device, requires_grad=True)

        # Optimizers
        self.ttfe_optimizer = torch.optim.Adam(self.ttfe.parameters(), lr=lr_ttfe)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr_actor)

        # Replay buffer
        self.buffer = ReplayBuffer(
            capacity=buffer_capacity,
            seq_len=seq_len,
            n_prices=n_prices,
            static_dim=static_dim,
            action_dim=self.action_dim,
        )

    @property
    def alpha(self):
        return self.log_alpha.exp().detach()

    def _encode_obs(self, price_history: torch.Tensor, static_features: torch.Tensor) -> torch.Tensor:
        """Run TTFE on price history and concatenate with current prices + static features."""
        temporal = self.ttfe(price_history)                 # (batch, d_model)
        current_prices = price_history[:, -1, :]            # (batch, n_prices)
        return torch.cat([temporal, current_prices, static_features], dim=-1)  # (batch, obs_dim)

    @torch.no_grad()
    def select_action(self, obs: dict, deterministic: bool = False) -> np.ndarray:
        """
        Select action given observation dict.

        Parameters
        ----------
        obs : dict with 'price_history' (seq_len, n_prices) and 'static_features' (static_dim,)
        deterministic : bool
            If True, use argmax mode + tanh(mean) magnitude (no sampling noise).

        Returns
        -------
        np.ndarray of shape (action_dim,)
        """
        self.ttfe.eval()
        self.actor.eval()

        ph = torch.tensor(obs["price_history"], dtype=torch.float32, device=self.device).unsqueeze(0)
        sf = torch.tensor(obs["static_features"], dtype=torch.float32, device=self.device).unsqueeze(0)

        encoded = self._encode_obs(ph, sf)

        if deterministic:
            _, _, action = self.actor.sample(encoded, tau=self.tau_gumbel, hard=True)
        else:
            action, _, _ = self.actor.sample(encoded, tau=self.tau_gumbel, hard=False)

        self.ttfe.train()
        self.actor.train()

        return action.squeeze(0).cpu().numpy()

    def update(self, batch: dict = None, tau_gumbel: float = None) -> dict:
        """
        Perform one SAC update step.

        Parameters
        ----------
        tau_gumbel : float, optional
            Override Gumbel temperature for this update. Uses self.tau_gumbel if None.

        Returns dict of losses/metrics.
        """
        if tau_gumbel is None:
            tau_gumbel = self.tau_gumbel

        if batch is None:
            if len(self.buffer) < self.batch_size:
                return {}
            batch = self.buffer.sample(self.batch_size, device=self.device)

        ph = batch["price_history"]
        sf = batch["static_features"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_ph = batch["next_price_history"]
        next_sf = batch["next_static_features"]
        dones = batch["dones"]

        # Encode observations
        obs_encoded = self._encode_obs(ph, sf)
        with torch.no_grad():
            next_obs_encoded = self._encode_obs(next_ph, next_sf)

        # --- Critic update ---
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.sample(
                next_obs_encoded, tau=tau_gumbel, hard=False
            )
            q1_target, q2_target = self.critic_target(next_obs_encoded, next_actions)
            q_target = torch.min(q1_target, q2_target) - self.alpha * next_log_probs
            td_target = rewards + (1.0 - dones) * self.gamma * q_target

        q1, q2 = self.critic(obs_encoded.detach(), actions)
        critic_loss = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_grad_norm = nn.utils.clip_grad_norm_(
            self.critic.parameters(), self.max_grad_norm
        )
        self.critic_optimizer.step()

        # --- Actor update ---
        new_actions, log_probs, _ = self.actor.sample(
            obs_encoded.detach(), tau=tau_gumbel, hard=False
        )
        q1_new, q2_new = self.critic(obs_encoded.detach(), new_actions)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_probs - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_grad_norm = nn.utils.clip_grad_norm_(
            self.actor.parameters(), self.max_grad_norm
        )
        self.actor_optimizer.step()

        # --- TTFE update (via critic gradients) ---
        obs_encoded_ttfe = self._encode_obs(ph, sf)
        q1_ttfe, q2_ttfe = self.critic(obs_encoded_ttfe, actions)
        ttfe_loss = -(q1_ttfe.mean() + q2_ttfe.mean()) * 0.5

        self.ttfe_optimizer.zero_grad()
        ttfe_loss.backward()
        ttfe_grad_norm = nn.utils.clip_grad_norm_(
            self.ttfe.parameters(), self.max_grad_norm
        )
        self.ttfe_optimizer.step()

        # --- Alpha update ---
        alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # --- Soft update target networks ---
        self._soft_update()

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha.item(),
            "q_mean": q_new.mean().item(),
            "critic_grad_norm": critic_grad_norm.item(),
            "actor_grad_norm": actor_grad_norm.item(),
            "ttfe_grad_norm": ttfe_grad_norm.item(),
        }

    def _soft_update(self):
        """Polyak averaging for target networks."""
        for p, p_target in zip(self.critic.parameters(), self.critic_target.parameters()):
            p_target.data.mul_(1.0 - self.tau)
            p_target.data.add_(self.tau * p.data)

    def save_checkpoint(self, path: str):
        """Save all model weights and optimizer states."""
        torch.save({
            "stage": self.stage,
            "tau_gumbel": self.tau_gumbel,
            "ttfe": self.ttfe.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "log_alpha": self.log_alpha.data,
            "ttfe_optimizer": self.ttfe_optimizer.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
        }, path)

    def load_checkpoint(self, path: str):
        """Load all model weights and optimizer states."""
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.ttfe.load_state_dict(ckpt["ttfe"])
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.log_alpha.data.copy_(ckpt["log_alpha"])
        if "tau_gumbel" in ckpt:
            self.tau_gumbel = ckpt["tau_gumbel"]
        self.ttfe_optimizer.load_state_dict(ckpt["ttfe_optimizer"])
        self.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
        self.critic_optimizer.load_state_dict(ckpt["critic_optimizer"])
        self.alpha_optimizer.load_state_dict(ckpt["alpha_optimizer"])

    def init_from_stage1(self, stage1_checkpoint_path: str):
        """
        Initialize Stage 2 agent from Stage 1 checkpoint.

        - TTFE weights copied from Stage 1.
        - Actor: energy + mode components copied; AS heads initialized near-zero.
        - Critics: fresh random initialization.
        - Buffer: empty.
        """
        assert self.stage == 2, "init_from_stage1 only for Stage 2"

        ckpt = torch.load(stage1_checkpoint_path, map_location=self.device, weights_only=True)

        self.ttfe.load_state_dict(ckpt["ttfe"])

        # Build temporary Stage 1 actor to transfer weights
        stage1_actor = Actor(obs_dim=self.actor.obs_dim, n_as_dims=0,
                             hidden_dim=self.actor.fc1.out_features)
        stage1_actor.load_state_dict(ckpt["actor"])

        new_actor = Actor.init_stage2_from_stage1(stage1_actor, n_as_dims=self.n_as_dims)
        self.actor.load_state_dict(new_actor.state_dict())

    def freeze_ttfe(self):
        """Freeze all TTFE parameters (for Stage 2 Phase 1)."""
        for p in self.ttfe.parameters():
            p.requires_grad = False

    def unfreeze_ttfe_top_layers(self, n_layers: int = 1, lr: float = 3e-5):
        """Unfreeze top N transformer layers (for Stage 2 Phase 2)."""
        total_layers = len(self.ttfe.transformer.layers)
        for i in range(total_layers - n_layers, total_layers):
            for p in self.ttfe.transformer.layers[i].parameters():
                p.requires_grad = True

        unfrozen = [p for p in self.ttfe.parameters() if p.requires_grad]
        if unfrozen:
            self.ttfe_optimizer = torch.optim.Adam(unfrozen, lr=lr)

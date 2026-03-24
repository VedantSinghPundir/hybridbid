"""
Replay buffer for SAC training.

Stores transitions with dict observations (price_history + static_features).
Fixed capacity with FIFO eviction and uniform random sampling.
"""

import torch
import numpy as np


class ReplayBuffer:
    """
    Fixed-capacity replay buffer for SAC.

    Observations are dicts with:
      - price_history: (seq_len, n_prices) tensor
      - static_features: (static_dim,) tensor
    """

    def __init__(
        self,
        capacity: int,
        seq_len: int = 32,
        n_prices: int = 12,
        static_dim: int = 14,
        action_dim: int = 1,
    ):
        self.capacity = capacity
        self.size = 0
        self.pos = 0

        # Pre-allocate storage as numpy arrays for memory efficiency
        self.price_history = np.zeros((capacity, seq_len, n_prices), dtype=np.float32)
        self.static_features = np.zeros((capacity, static_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_price_history = np.zeros((capacity, seq_len, n_prices), dtype=np.float32)
        self.next_static_features = np.zeros((capacity, static_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, obs: dict, action: np.ndarray, reward: float,
            next_obs: dict, done: bool):
        """Add a single transition."""
        idx = self.pos

        self.price_history[idx] = obs["price_history"]
        self.static_features[idx] = obs["static_features"]
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_price_history[idx] = next_obs["price_history"]
        self.next_static_features[idx] = next_obs["static_features"]
        self.dones[idx] = float(done)

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: str = "cpu") -> dict:
        """
        Sample a random batch of transitions.

        Returns dict of tensors on the specified device.
        """
        indices = np.random.randint(0, self.size, size=batch_size)

        return {
            "price_history": torch.tensor(self.price_history[indices], device=device),
            "static_features": torch.tensor(self.static_features[indices], device=device),
            "actions": torch.tensor(self.actions[indices], device=device),
            "rewards": torch.tensor(self.rewards[indices], device=device),
            "next_price_history": torch.tensor(self.next_price_history[indices], device=device),
            "next_static_features": torch.tensor(self.next_static_features[indices], device=device),
            "dones": torch.tensor(self.dones[indices], device=device),
        }

    def __len__(self):
        return self.size

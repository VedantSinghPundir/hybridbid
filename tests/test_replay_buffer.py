"""Tests for replay buffer."""

import numpy as np
import pytest
from src.models.replay_buffer import ReplayBuffer


def _make_obs(val=0.0):
    return {
        "price_history": np.full((32, 12), val, dtype=np.float32),
        "static_features": np.full(14, val, dtype=np.float32),
    }


def test_add_and_sample():
    buf = ReplayBuffer(capacity=100, action_dim=1)
    for i in range(50):
        buf.add(_make_obs(float(i)), np.array([float(i)]), float(i),
                _make_obs(float(i + 1)), False)

    assert len(buf) == 50

    batch = buf.sample(16)
    assert batch["price_history"].shape == (16, 32, 12)
    assert batch["static_features"].shape == (16, 14)
    assert batch["actions"].shape == (16, 1)
    assert batch["rewards"].shape == (16, 1)
    assert batch["next_price_history"].shape == (16, 32, 12)
    assert batch["next_static_features"].shape == (16, 14)
    assert batch["dones"].shape == (16, 1)


def test_fifo_overflow():
    buf = ReplayBuffer(capacity=10, action_dim=1)
    for i in range(20):
        buf.add(_make_obs(float(i)), np.array([float(i)]), float(i),
                _make_obs(float(i + 1)), False)

    assert len(buf) == 10
    # The oldest entries (0-9) should have been overwritten
    # Buffer should contain entries 10-19
    batch = buf.sample(10)
    # All reward values should be >= 10
    assert (batch["rewards"] >= 10.0).all()


def test_batch_diversity():
    buf = ReplayBuffer(capacity=1000, action_dim=1)
    for i in range(100):
        buf.add(_make_obs(float(i)), np.array([float(i)]), float(i),
                _make_obs(float(i + 1)), False)

    batch = buf.sample(32)
    # Not all rewards should be the same
    rewards = batch["rewards"].numpy().flatten()
    assert len(set(rewards)) > 1


def test_6d_actions():
    buf = ReplayBuffer(capacity=100, action_dim=6)
    for i in range(20):
        buf.add(_make_obs(), np.random.randn(6).astype(np.float32), 1.0,
                _make_obs(), False)

    batch = buf.sample(8)
    assert batch["actions"].shape == (8, 6)

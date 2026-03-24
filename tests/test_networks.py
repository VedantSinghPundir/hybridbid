"""Tests for Actor and Critic networks."""

import torch
import pytest
from src.models.networks import Actor, TwinCritic


@pytest.fixture
def actor_1d():
    return Actor(obs_dim=78, action_dim=1)


@pytest.fixture
def actor_6d():
    return Actor(obs_dim=78, action_dim=6)


@pytest.fixture
def twin_critic_1d():
    return TwinCritic(obs_dim=78, action_dim=1)


@pytest.fixture
def twin_critic_6d():
    return TwinCritic(obs_dim=78, action_dim=6)


def test_actor_forward_1d(actor_1d):
    obs = torch.randn(8, 78)
    mean, log_std = actor_1d(obs)
    assert mean.shape == (8, 1)
    assert log_std.shape == (8, 1)


def test_actor_forward_6d(actor_6d):
    obs = torch.randn(8, 78)
    mean, log_std = actor_6d(obs)
    assert mean.shape == (8, 6)
    assert log_std.shape == (8, 6)


def test_critic_forward_1d(twin_critic_1d):
    obs = torch.randn(8, 78)
    action = torch.randn(8, 1)
    q1, q2 = twin_critic_1d(obs, action)
    assert q1.shape == (8, 1)
    assert q2.shape == (8, 1)


def test_critic_forward_6d(twin_critic_6d):
    obs = torch.randn(8, 78)
    action = torch.randn(8, 6)
    q1, q2 = twin_critic_6d(obs, action)
    assert q1.shape == (8, 1)
    assert q2.shape == (8, 1)


def test_stage2_init_from_stage1(actor_1d):
    # Run a forward pass to populate weights
    obs = torch.randn(4, 78)
    actor_1d(obs)

    actor_6d = Actor.init_stage2_from_stage1(actor_1d, action_dim=6)

    # Hidden layers should match
    assert torch.allclose(actor_6d.fc1.weight, actor_1d.fc1.weight)
    assert torch.allclose(actor_6d.fc2.weight, actor_1d.fc2.weight)

    # Energy dim (row 0) should match Stage 1
    assert torch.allclose(actor_6d.mean_head.weight[0], actor_1d.mean_head.weight[0])
    assert torch.allclose(actor_6d.mean_head.bias[0], actor_1d.mean_head.bias[0])

    # AS dims (rows 1-5) should be near-zero
    assert actor_6d.mean_head.weight[1:].abs().max() < 0.1
    assert actor_6d.mean_head.bias[1:].abs().max() < 1e-6

    # Same for log_std
    assert torch.allclose(actor_6d.log_std_head.weight[0], actor_1d.log_std_head.weight[0])
    assert actor_6d.log_std_head.weight[1:].abs().max() < 0.1
    assert actor_6d.log_std_head.bias[1:].abs().max() < 1e-6

    # Forward pass should work
    out_mean, out_log_std = actor_6d(obs)
    assert out_mean.shape == (4, 6)


def test_action_sampling(actor_1d, actor_6d):
    obs = torch.randn(8, 78)

    # 1D sampling
    action, log_prob, det_action = actor_1d.sample(obs)
    assert action.shape == (8, 1)
    assert log_prob.shape == (8, 1)
    assert det_action.shape == (8, 1)
    assert (action >= -1).all() and (action <= 1).all()

    # 6D sampling
    action, log_prob, det_action = actor_6d.sample(obs)
    assert action.shape == (8, 6)
    assert log_prob.shape == (8, 1)
    assert det_action.shape == (8, 6)
    assert (action >= -1).all() and (action <= 1).all()

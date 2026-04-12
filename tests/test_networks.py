"""Tests for Actor and Critic networks."""

import torch
import pytest
from src.models.networks import Actor, TwinCritic, MODE_CHARGE, MODE_DISCHARGE, MODE_IDLE


@pytest.fixture
def actor_stage1():
    return Actor(obs_dim=90, n_as_dims=0)


@pytest.fixture
def actor_stage2():
    return Actor(obs_dim=90, n_as_dims=5)


@pytest.fixture
def twin_critic_stage1():
    return TwinCritic(obs_dim=90, action_dim=4)


@pytest.fixture
def twin_critic_stage2():
    return TwinCritic(obs_dim=90, action_dim=9)


class TestActorForward:
    def test_stage1_output_shapes(self, actor_stage1):
        obs = torch.randn(8, 90)
        mode_logits, energy_mean, energy_log_std = actor_stage1(obs)
        assert mode_logits.shape == (8, 3)
        assert energy_mean.shape == (8, 1)
        assert energy_log_std.shape == (8, 1)

    def test_stage2_output_shapes(self, actor_stage2):
        obs = torch.randn(8, 90)
        mode_logits, energy_mean, energy_log_std, as_mean, as_log_std = actor_stage2(obs)
        assert mode_logits.shape == (8, 3)
        assert energy_mean.shape == (8, 1)
        assert energy_log_std.shape == (8, 1)
        assert as_mean.shape == (8, 5)
        assert as_log_std.shape == (8, 5)

    def test_action_dim_attributes(self, actor_stage1, actor_stage2):
        assert actor_stage1.action_dim == 4   # 3 mode + 1 energy
        assert actor_stage2.action_dim == 9   # 3 mode + 1 energy + 5 AS


class TestActorSampling:
    def test_stage1_sample_shapes(self, actor_stage1):
        obs = torch.randn(8, 90)
        action, log_prob, det_action = actor_stage1.sample(obs, tau=1.0)
        assert action.shape == (8, 4)
        assert log_prob.shape == (8, 1)
        assert det_action.shape == (8, 4)

    def test_stage2_sample_shapes(self, actor_stage2):
        obs = torch.randn(8, 90)
        action, log_prob, det_action = actor_stage2.sample(obs, tau=1.0)
        assert action.shape == (8, 9)
        assert log_prob.shape == (8, 1)
        assert det_action.shape == (8, 9)

    def test_mode_soft_sums_to_one(self, actor_stage1):
        obs = torch.randn(16, 90)
        action, _, _ = actor_stage1.sample(obs, tau=1.0, hard=False)
        mode_soft = action[:, :3]
        # Gumbel-Softmax: each row sums to ~1 and values are non-negative
        assert (mode_soft >= 0).all(), "Soft mode values should be non-negative"
        torch.testing.assert_close(
            mode_soft.sum(dim=-1), torch.ones(16), atol=1e-5, rtol=1e-5
        )

    def test_hard_mode_is_one_hot(self, actor_stage1):
        obs = torch.randn(16, 90)
        action, _, _ = actor_stage1.sample(obs, tau=1.0, hard=True)
        mode_hard = action[:, :3]
        # Hard mode: each row is one-hot (one 1.0, two 0.0s)
        assert (mode_hard.sum(dim=-1) == 1).all()
        assert ((mode_hard == 0) | (mode_hard == 1)).all()

    def test_deterministic_is_consistent(self, actor_stage1):
        obs = torch.randn(4, 90)
        _, _, det1 = actor_stage1.sample(obs, tau=1.0, hard=True)
        _, _, det2 = actor_stage1.sample(obs, tau=1.0, hard=True)
        torch.testing.assert_close(det1, det2)

    def test_energy_magnitude_in_range(self, actor_stage1):
        obs = torch.randn(32, 90)
        action, _, _ = actor_stage1.sample(obs, tau=1.0)
        energy_mag = action[:, 3]
        assert (energy_mag >= -1).all() and (energy_mag <= 1).all()

    def test_all_modes_sampled(self, actor_stage1):
        # Over many samples, all 3 modes should appear
        obs = torch.randn(256, 90)
        action, _, _ = actor_stage1.sample(obs, tau=1.0, hard=True)
        modes = action[:, :3].argmax(dim=-1)
        assert modes.unique().numel() == 3, "All 3 modes should be sampled over 256 draws"


class TestCriticForward:
    def test_stage1_critic(self, twin_critic_stage1):
        obs = torch.randn(8, 90)
        action = torch.randn(8, 4)
        q1, q2 = twin_critic_stage1(obs, action)
        assert q1.shape == (8, 1)
        assert q2.shape == (8, 1)

    def test_stage2_critic(self, twin_critic_stage2):
        obs = torch.randn(8, 90)
        action = torch.randn(8, 9)
        q1, q2 = twin_critic_stage2(obs, action)
        assert q1.shape == (8, 1)
        assert q2.shape == (8, 1)


class TestStage2InitFromStage1:
    def test_trunk_and_energy_heads_copied(self, actor_stage1):
        obs = torch.randn(4, 90)
        actor_stage1(obs)  # populate weights

        actor2 = Actor.init_stage2_from_stage1(actor_stage1, n_as_dims=5)

        # Shared trunk
        assert torch.allclose(actor2.fc1.weight, actor_stage1.fc1.weight)
        assert torch.allclose(actor2.fc2.weight, actor_stage1.fc2.weight)

        # Mode head
        assert torch.allclose(actor2.mode_head.weight, actor_stage1.mode_head.weight)

        # Energy magnitude heads
        assert torch.allclose(
            actor2.energy_mag_mean_head.weight, actor_stage1.energy_mag_mean_head.weight
        )
        assert torch.allclose(
            actor2.energy_mag_log_std_head.weight, actor_stage1.energy_mag_log_std_head.weight
        )

    def test_as_heads_near_zero(self, actor_stage1):
        actor2 = Actor.init_stage2_from_stage1(actor_stage1, n_as_dims=5)

        assert actor2.as_mag_mean_head.weight.abs().max() < 0.1
        assert actor2.as_mag_mean_head.bias.abs().max() < 1e-6
        assert actor2.as_mag_log_std_head.weight.abs().max() < 0.1
        assert actor2.as_mag_log_std_head.bias.abs().max() < 1e-6

    def test_stage2_forward_works(self, actor_stage1):
        actor2 = Actor.init_stage2_from_stage1(actor_stage1, n_as_dims=5)
        obs = torch.randn(4, 90)
        action, log_prob, det = actor2.sample(obs)
        assert action.shape == (4, 9)
        assert log_prob.shape == (4, 1)

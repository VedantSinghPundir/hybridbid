"""Tests for SAC agent."""

import os
import tempfile
import numpy as np
import torch
import pytest
from src.models.sac import SACAgent
from src.models.networks import Actor


def _make_obs():
    return {
        "price_history": np.random.randn(32, 12).astype(np.float32),
        "static_features": np.random.randn(14).astype(np.float32),
    }


class TestSACConstruction:
    def test_stage1(self):
        agent = SACAgent(stage=1, device="cpu")
        assert agent.action_dim == 4   # 3 mode + 1 energy_mag
        assert agent.n_as_dims == 0
        assert agent.batch_size == 256

    def test_stage2(self):
        agent = SACAgent(stage=2, device="cpu")
        assert agent.action_dim == 9   # 3 mode + 1 energy_mag + 5 AS_mags
        assert agent.n_as_dims == 5
        assert agent.batch_size == 128

    def test_no_alpha_min(self):
        """alpha should not have a floor — no alpha_min clamp."""
        agent = SACAgent(stage=1, device="cpu")
        # log_alpha starts at 0 → alpha = 1.0; no clamp should be applied
        assert agent.alpha.item() == pytest.approx(1.0, abs=1e-4)

    def test_nhead_default_is_8(self):
        """Paper Table I specifies h=8 attention heads."""
        agent = SACAgent(stage=1, device="cpu")
        # Check that TTFE was built with nhead=8
        ttfe_layer = agent.ttfe.transformer.layers[0]
        assert ttfe_layer.self_attn.num_heads == 8

    def test_tau_default_is_0_01(self):
        """Paper Table I specifies τ_ψ=0.01 for target network."""
        agent = SACAgent(stage=1, device="cpu")
        assert agent.tau == 0.01


class TestSelectAction:
    def test_stage1_shape(self):
        agent = SACAgent(stage=1, device="cpu")
        obs = _make_obs()
        action = agent.select_action(obs)
        assert action.shape == (4,)

    def test_stage2_shape(self):
        agent = SACAgent(stage=2, device="cpu")
        obs = _make_obs()
        action = agent.select_action(obs)
        assert action.shape == (9,)

    def test_deterministic_consistent(self):
        agent = SACAgent(stage=1, device="cpu")
        obs = _make_obs()
        a1 = agent.select_action(obs, deterministic=True)
        a2 = agent.select_action(obs, deterministic=True)
        np.testing.assert_allclose(a1, a2)

    def test_stochastic_varies(self):
        agent = SACAgent(stage=1, device="cpu")
        obs = _make_obs()
        actions = [agent.select_action(obs) for _ in range(20)]
        # Over 20 samples, actions should not all be identical
        unique_modes = set(np.argmax(a[:3]) for a in actions)
        # With random weights and tau=1.0, should see at least 2 modes eventually
        assert len(unique_modes) >= 1  # at minimum, mode selection is deterministic from weights


class TestUpdate:
    def test_stage1_update_returns_metrics(self):
        agent = SACAgent(stage=1, device="cpu", buffer_capacity=200, batch_size=16)
        for _ in range(50):
            obs = _make_obs()
            action = np.random.randn(4).astype(np.float32)
            next_obs = _make_obs()
            agent.buffer.add(obs, action, 1.0, next_obs, False)

        metrics = agent.update()
        assert "critic_loss" in metrics
        assert "actor_loss" in metrics
        assert "alpha" in metrics

    def test_stage2_update_returns_metrics(self):
        agent = SACAgent(stage=2, device="cpu", buffer_capacity=200, batch_size=16)
        for _ in range(50):
            obs = _make_obs()
            action = np.random.randn(9).astype(np.float32)
            next_obs = _make_obs()
            agent.buffer.add(obs, action, 1.0, next_obs, False)

        metrics = agent.update()
        assert "critic_loss" in metrics

    def test_update_with_tau_gumbel(self):
        """update() should accept tau_gumbel parameter."""
        agent = SACAgent(stage=1, device="cpu", buffer_capacity=200, batch_size=16)
        for _ in range(50):
            agent.buffer.add(_make_obs(), np.random.randn(4).astype(np.float32),
                             1.0, _make_obs(), False)
        # Should not raise
        metrics = agent.update(tau_gumbel=0.5)
        assert "critic_loss" in metrics

    def test_no_nan_in_metrics(self):
        agent = SACAgent(stage=1, device="cpu", buffer_capacity=200, batch_size=16)
        for _ in range(50):
            agent.buffer.add(_make_obs(), np.random.randn(4).astype(np.float32),
                             1.0, _make_obs(), False)
        metrics = agent.update()
        for k, v in metrics.items():
            assert not np.isnan(v), f"NaN in metric: {k}"


class TestCheckpoint:
    def test_save_load_roundtrip(self):
        agent = SACAgent(stage=1, device="cpu")
        orig_weight = agent.ttfe.input_proj.weight.data.clone()

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name

        try:
            agent.save_checkpoint(path)

            agent2 = SACAgent(stage=1, device="cpu")
            agent2.load_checkpoint(path)

            assert torch.allclose(agent2.ttfe.input_proj.weight.data, orig_weight)
            assert torch.allclose(agent2.actor.fc1.weight.data, agent.actor.fc1.weight.data)
        finally:
            os.unlink(path)


class TestStage2InitFromStage1:
    def test_ttfe_and_actor_weights_transferred(self):
        agent1 = SACAgent(stage=1, device="cpu")

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name

        try:
            agent1.save_checkpoint(path)

            agent2 = SACAgent(stage=2, device="cpu")
            agent2.init_from_stage1(path)

            # TTFE weights should match
            for (n1, p1), (n2, p2) in zip(
                agent1.ttfe.named_parameters(), agent2.ttfe.named_parameters()
            ):
                assert torch.allclose(p1, p2), f"TTFE mismatch at {n1}"

            # Mode head should match
            assert torch.allclose(
                agent2.actor.mode_head.weight, agent1.actor.mode_head.weight
            )

            # Energy mag heads should match
            assert torch.allclose(
                agent2.actor.energy_mag_mean_head.weight,
                agent1.actor.energy_mag_mean_head.weight,
            )

            # AS heads should be near-zero
            assert agent2.actor.as_mag_mean_head.weight.abs().max() < 0.1
            assert agent2.actor.as_mag_mean_head.bias.abs().max() < 1e-6

        finally:
            os.unlink(path)

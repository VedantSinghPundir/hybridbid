"""Tests for SAC agent."""

import os
import tempfile
import numpy as np
import torch
import pytest
from src.models.sac import SACAgent


def _make_obs():
    return {
        "price_history": np.random.randn(32, 12).astype(np.float32),
        "static_features": np.random.randn(14).astype(np.float32),
    }


class TestSACConstruction:
    def test_stage1(self):
        agent = SACAgent(stage=1, device="cpu")
        assert agent.action_dim == 1
        assert agent.batch_size == 256

    def test_stage2(self):
        agent = SACAgent(stage=2, device="cpu")
        assert agent.action_dim == 6
        assert agent.batch_size == 128


class TestSelectAction:
    def test_stage1(self):
        agent = SACAgent(stage=1, device="cpu")
        obs = _make_obs()
        action = agent.select_action(obs)
        assert action.shape == (1,)
        assert -1 <= action[0] <= 1

    def test_stage2(self):
        agent = SACAgent(stage=2, device="cpu")
        obs = _make_obs()
        action = agent.select_action(obs)
        assert action.shape == (6,)

    def test_deterministic(self):
        agent = SACAgent(stage=1, device="cpu")
        obs = _make_obs()
        a1 = agent.select_action(obs, deterministic=True)
        a2 = agent.select_action(obs, deterministic=True)
        np.testing.assert_allclose(a1, a2)


class TestUpdate:
    def test_stage1_update(self):
        agent = SACAgent(stage=1, device="cpu", buffer_capacity=200, batch_size=16)
        # Fill buffer
        for _ in range(50):
            obs = _make_obs()
            action = np.random.randn(1).astype(np.float32)
            next_obs = _make_obs()
            agent.buffer.add(obs, action, 1.0, next_obs, False)

        metrics = agent.update()
        assert "critic_loss" in metrics
        assert "actor_loss" in metrics
        assert "alpha" in metrics

    def test_stage2_update(self):
        agent = SACAgent(stage=2, device="cpu", buffer_capacity=200, batch_size=16)
        for _ in range(50):
            obs = _make_obs()
            action = np.random.randn(6).astype(np.float32)
            next_obs = _make_obs()
            agent.buffer.add(obs, action, 1.0, next_obs, False)

        metrics = agent.update()
        assert "critic_loss" in metrics


class TestCheckpoint:
    def test_save_load_roundtrip(self):
        agent = SACAgent(stage=1, device="cpu")
        # Get original TTFE weight
        orig_weight = agent.ttfe.input_proj.weight.data.clone()

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name

        try:
            agent.save_checkpoint(path)

            # Create fresh agent and load
            agent2 = SACAgent(stage=1, device="cpu")
            agent2.load_checkpoint(path)

            assert torch.allclose(agent2.ttfe.input_proj.weight.data, orig_weight)
            assert torch.allclose(agent2.actor.fc1.weight.data, agent.actor.fc1.weight.data)
        finally:
            os.unlink(path)


class TestStage2InitFromStage1:
    def test_init_from_stage1(self):
        # Train Stage 1
        agent1 = SACAgent(stage=1, device="cpu")

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name

        try:
            agent1.save_checkpoint(path)

            # Init Stage 2 from Stage 1
            agent2 = SACAgent(stage=2, device="cpu")
            agent2.init_from_stage1(path)

            # TTFE weights should match
            for (n1, p1), (n2, p2) in zip(
                agent1.ttfe.named_parameters(), agent2.ttfe.named_parameters()
            ):
                assert torch.allclose(p1, p2), f"TTFE mismatch at {n1}"

            # Actor energy dim should match
            assert torch.allclose(
                agent2.actor.mean_head.weight[0],
                agent1.actor.mean_head.weight[0],
            )

            # AS dims should be near-zero
            assert agent2.actor.mean_head.weight[1:].abs().max() < 0.1
        finally:
            os.unlink(path)

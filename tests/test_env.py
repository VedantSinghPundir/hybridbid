"""Tests for ERCOT Battery Gymnasium Environment."""

import numpy as np
import pytest
from src.env.ercot_env import ERCOTBatteryEnv

DATA_DIR = "data/processed"


def _make_env(mode="energy_only", date_range=("2026-01-07", "2026-01-12")):
    """Create env with a date range that has clean data."""
    return ERCOTBatteryEnv(
        data_dir=DATA_DIR,
        mode=mode,
        date_range=date_range,
    )


class TestEnvCreation:
    def test_energy_only(self):
        env = _make_env("energy_only")
        assert env.action_space.shape == (1,)

    def test_co_optimize(self):
        env = _make_env("co_optimize")
        assert env.action_space.shape == (6,)


class TestReset:
    def test_energy_only_obs_shapes(self):
        env = _make_env("energy_only")
        obs, info = env.reset()
        assert obs["price_history"].shape == (32, 12)
        assert obs["static_features"].shape == (14,)

    def test_co_optimize_obs_shapes(self):
        env = _make_env("co_optimize")
        obs, info = env.reset()
        assert obs["price_history"].shape == (32, 12)
        assert obs["static_features"].shape == (14,)


class TestStep:
    def test_energy_only_step(self):
        env = _make_env("energy_only")
        obs, _ = env.reset()
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        assert next_obs["price_history"].shape == (32, 12)
        assert next_obs["static_features"].shape == (14,)
        assert isinstance(reward, float)
        assert not terminated

    def test_co_optimize_step(self):
        env = _make_env("co_optimize")
        obs, _ = env.reset()
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        assert next_obs["price_history"].shape == (32, 12)
        assert isinstance(reward, float)


class TestFullEpisode:
    def test_energy_only_full_day(self):
        env = _make_env("energy_only")
        obs, _ = env.reset()
        total_reward = 0.0
        socs = []

        for step in range(288):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            socs.append(info["soc"])

            if step < 287:
                assert not terminated
            else:
                assert terminated

        # SoC should stay within bounds
        socs = np.array(socs)
        soc_min = env.soc_min_frac * env.e_max
        soc_max = env.soc_max_frac * env.e_max
        assert (socs >= soc_min - 0.01).all(), f"SoC violated min: {socs.min()}"
        assert (socs <= soc_max + 0.01).all(), f"SoC violated max: {socs.max()}"

    def test_co_optimize_full_day(self):
        env = _make_env("co_optimize")
        obs, _ = env.reset()

        for step in range(288):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

        assert terminated


class TestRewardNonZero:
    def test_nonzero_action_nonzero_reward(self):
        env = _make_env("energy_only")
        obs, _ = env.reset()
        # Take a big action
        action = np.array([0.8], dtype=np.float32)
        _, reward, _, _, info = env.step(action)
        # With non-zero LMP and non-zero action, reward should be non-zero
        # (degradation cost alone makes it non-zero)
        assert reward != 0.0 or info["p_net"] == 0.0


class TestMCPCColumns:
    def test_energy_only_pre_rtcb_zeros(self):
        """Check that RT MCPC price columns are usable (filled with 0 where NaN)."""
        env = _make_env("energy_only")
        obs, _ = env.reset()
        # RT MCPC columns are indices 1-5 in price_history
        # For post-RTC+B data they should have non-zero values
        rt_mcpc = obs["price_history"][:, 1:6]
        # At least some should be non-zero for post-RTC+B dates
        assert np.isfinite(rt_mcpc).all()

    def test_co_optimize_has_mcpc(self):
        """Post-RTC+B dates should have non-zero RT MCPC values."""
        env = _make_env("co_optimize")
        obs, _ = env.reset()
        rt_mcpc = obs["price_history"][:, 1:6]
        assert np.isfinite(rt_mcpc).all()
        # At least some non-zero MCPC values expected
        assert rt_mcpc.sum() > 0

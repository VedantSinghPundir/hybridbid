"""Tests for ERCOT Battery Gymnasium Environment."""

import numpy as np
import pytest
from src.env.ercot_env import ERCOTBatteryEnv, MODE_CHARGE, MODE_DISCHARGE, MODE_IDLE

DATA_DIR = "data/processed"


def _make_env(mode="energy_only", date_range=("2026-01-07", "2026-01-12")):
    """Create env with a date range that has clean data."""
    return ERCOTBatteryEnv(
        data_dir=DATA_DIR,
        mode=mode,
        date_range=date_range,
    )


def _discharge_action():
    """Hard discharge action: [0,1,0, magnitude=0.5]"""
    a = np.zeros(4, dtype=np.float32)
    a[MODE_DISCHARGE] = 1.0  # index 1
    a[3] = 0.5
    return a


def _charge_action():
    """Hard charge action: [1,0,0, magnitude=0.5]"""
    a = np.zeros(4, dtype=np.float32)
    a[MODE_CHARGE] = 1.0  # index 0
    a[3] = 0.5
    return a


def _idle_action():
    """Hard idle action: [0,0,1, magnitude=0.0]"""
    a = np.zeros(4, dtype=np.float32)
    a[MODE_IDLE] = 1.0  # index 2
    a[3] = 0.0
    return a


class TestEnvCreation:
    def test_energy_only_action_space(self):
        env = _make_env("energy_only")
        assert env.action_space.shape == (4,)

    def test_co_optimize_action_space(self):
        env = _make_env("co_optimize")
        assert env.action_space.shape == (9,)


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

    def test_initial_soc_fixed_at_50pct(self):
        env = _make_env("energy_only")
        obs, _ = env.reset()
        assert abs(env.soc - 0.50 * env.e_max) < 1e-6


class TestStep:
    def test_energy_only_step(self):
        env = _make_env("energy_only")
        obs, _ = env.reset()
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        assert next_obs["price_history"].shape == (32, 12)
        assert next_obs["static_features"].shape == (14,)
        assert isinstance(reward, float)

    def test_co_optimize_step(self):
        env = _make_env("co_optimize")
        obs, _ = env.reset()
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        assert next_obs["price_history"].shape == (32, 12)
        assert isinstance(reward, float)

    def test_discharge_action_reduces_soc(self):
        env = _make_env("energy_only")
        env.reset()
        soc_before = env.soc
        _, _, _, _, info = env.step(_discharge_action())
        # Discharging should reduce SoC (unless SoC is already at min)
        if soc_before > env.soc_min:
            assert env.soc <= soc_before

    def test_charge_action_increases_soc(self):
        env = _make_env("energy_only")
        env.reset()
        soc_before = env.soc
        _, _, _, _, info = env.step(_charge_action())
        # Charging should increase SoC (unless SoC is already at max)
        if soc_before < env.soc_max:
            assert env.soc >= soc_before

    def test_idle_action_does_not_change_soc(self):
        env = _make_env("energy_only")
        env.reset()
        soc_before = env.soc
        _, _, _, _, info = env.step(_idle_action())
        assert abs(env.soc - soc_before) < 1e-6

    def test_info_contains_required_fields(self):
        env = _make_env("energy_only")
        env.reset()
        _, _, _, _, info = env.step(_discharge_action())
        for key in ["energy_revenue", "timing_bonus", "soc", "p_net", "mode", "ema_price"]:
            assert key in info, f"Missing info key: {key}"


class TestReward:
    def test_energy_term_positive_for_good_discharge(self):
        """Discharging when LMP > 0 should yield positive energy_revenue."""
        env = _make_env("energy_only")
        env.reset()
        # Step: take discharge action
        _, _, _, _, info = env.step(_discharge_action())
        rt_lmp = env.price_data[env._day_start_idx, 0]
        if rt_lmp > 0 and info["p_net"] > 0:
            assert info["energy_revenue"] > 0

    def test_no_degradation_in_reward(self):
        """Reward should not include degradation cost (removed per paper Eq. 30)."""
        env = _make_env("energy_only")
        env.reset()
        _, reward, _, _, info = env.step(_discharge_action())
        # Reward = energy_term + timing_bonus — no separate degradation term
        expected = info["energy_revenue"] + info["timing_bonus"]
        assert abs(reward - expected) < 1e-4, (
            f"reward={reward:.4f} but energy_rev+timing_bonus={expected:.4f}; "
            "degradation cost should not appear in step reward"
        )

    def test_ema_arbitrage_bonus_sign(self):
        """Discharging when price > EMA should yield positive timing_bonus."""
        env = _make_env("energy_only")
        env.reset()
        # Prime the EMA to be lower than current LMP
        env.ema_price = 0.0
        # Now step with discharge
        _, _, _, _, info = env.step(_discharge_action())
        rt_lmp = env.price_data[env._day_start_idx, 0]
        if rt_lmp > 0 and info["p_net"] > 0:
            assert info["timing_bonus"] >= 0

    def test_timing_bonus_uses_beta_10(self):
        """Verify β_S=10 is active (timing_bonus should be substantial)."""
        from src.env.ercot_env import BETA_ARB
        assert BETA_ARB == 10.0

    def test_ema_tau_is_0_9(self):
        """Verify τ_S=0.9 is used."""
        from src.env.ercot_env import EMA_TAU
        assert EMA_TAU == 0.9


class TestSoCTermination:
    def test_soc_violation_terminates(self):
        """Draining SoC to zero should trigger termination with -50 penalty."""
        env = _make_env("energy_only")
        env.reset()
        # Artificially set SoC near minimum, then discharge hard
        env.soc = env.soc_min + 0.001  # barely above minimum
        action = np.zeros(4, dtype=np.float32)
        action[MODE_DISCHARGE] = 1.0  # discharge
        action[3] = 1.0               # full power
        _, reward, terminated, _, info = env.step(action)
        assert terminated, "SoC violation should set terminated=True"
        assert info["soc_violated"], "soc_violated flag should be True"
        # Penalty (-50) was applied: reward = energy_term + timing_bonus - 50
        expected = info["energy_revenue"] + info["timing_bonus"] - 50.0
        assert abs(reward - expected) < 1e-3, (
            f"reward={reward:.3f}, expected {expected:.3f} (energy+timing-50)"
        )

    def test_soc_stays_in_bounds_after_violation(self):
        """Feasibility projection should prevent actual SoC overflow even on termination."""
        env = _make_env("energy_only")
        env.reset()
        env.soc = env.soc_min + 0.001
        action = np.zeros(4, dtype=np.float32)
        action[MODE_DISCHARGE] = 1.0
        action[3] = 1.0
        env.step(action)
        assert env.soc >= env.soc_min - 0.01
        assert env.soc <= env.soc_max + 0.01

    def test_no_termination_on_normal_operation(self):
        """Normal idle action should not trigger termination."""
        env = _make_env("energy_only")
        env.reset()
        _, _, terminated, _, _ = env.step(_idle_action())
        assert not terminated


class TestFullEpisode:
    def test_energy_only_full_day(self):
        env = _make_env("energy_only")
        obs, _ = env.reset()
        total_reward = 0.0
        socs = []
        done = False

        for _ in range(288):
            if done:
                break
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            socs.append(info["soc"])
            done = terminated or truncated

        assert done  # episode must end (either truncated or terminated)

        # SoC must stay within bounds (feasibility projection ensures this)
        socs = np.array(socs)
        soc_min = env.soc_min_frac * env.e_max
        soc_max = env.soc_max_frac * env.e_max
        assert (socs >= soc_min - 0.01).all(), f"SoC violated min: {socs.min()}"
        assert (socs <= soc_max + 0.01).all(), f"SoC violated max: {socs.max()}"

    def test_co_optimize_full_day(self):
        env = _make_env("co_optimize")
        obs, _ = env.reset()
        done = False
        for _ in range(288):
            if done:
                break
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        assert done


class TestMCPCColumns:
    def test_energy_only_prices_finite(self):
        env = _make_env("energy_only")
        obs, _ = env.reset()
        rt_mcpc = obs["price_history"][:, 1:6]
        assert np.isfinite(rt_mcpc).all()

    def test_co_optimize_has_mcpc(self):
        env = _make_env("co_optimize")
        obs, _ = env.reset()
        rt_mcpc = obs["price_history"][:, 1:6]
        assert np.isfinite(rt_mcpc).all()
        assert rt_mcpc.sum() > 0

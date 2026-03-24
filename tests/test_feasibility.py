"""Tests for feasibility projection."""

import torch
import pytest
from src.models.feasibility import project_energy_only, project_co_optimize, DELTA_T_HOURS


P_MAX = 10.0
E_MAX = 20.0
SOC_MIN = 0.10
SOC_MAX = 0.90
ETA_CH = 0.92
ETA_DCH = 0.92

KWARGS = dict(p_max=P_MAX, e_max=E_MAX, soc_min_frac=SOC_MIN,
              soc_max_frac=SOC_MAX, eta_ch=ETA_CH, eta_dch=ETA_DCH)


class TestEnergyOnly:
    def test_power_clipping(self):
        action = torch.tensor(15.0)
        soc = torch.tensor(10.0)  # 50% SoC
        out = project_energy_only(action, soc, **KWARGS)
        assert out <= P_MAX

    def test_soc_min_respected(self):
        """At SoC_min, can't discharge."""
        soc = torch.tensor(E_MAX * SOC_MIN)  # exactly at min
        action = torch.tensor(10.0)  # try to discharge
        out = project_energy_only(action, soc, **KWARGS)
        # Discharge should be zero or very small
        assert out <= 0.01

    def test_soc_max_respected(self):
        """At SoC_max, can't charge."""
        soc = torch.tensor(E_MAX * SOC_MAX)  # exactly at max
        action = torch.tensor(-10.0)  # try to charge
        out = project_energy_only(action, soc, **KWARGS)
        assert out >= -0.01

    def test_mid_soc_no_clipping(self):
        """At 50% SoC, moderate actions pass through."""
        soc = torch.tensor(10.0)
        action = torch.tensor(5.0)
        out = project_energy_only(action, soc, **KWARGS)
        assert torch.isclose(out, action, atol=0.01)

    def test_batch(self):
        actions = torch.tensor([5.0, -5.0, 15.0, -15.0])
        socs = torch.tensor([10.0, 10.0, 10.0, 10.0])
        out = project_energy_only(actions, socs, **KWARGS)
        assert out.shape == (4,)
        assert (out <= P_MAX).all()
        assert (out >= -P_MAX).all()


class TestCoOptimize:
    def test_as_nonnegative(self):
        action = torch.tensor([5.0, -1.0, -2.0, -3.0, -4.0, -5.0])
        soc = torch.tensor(10.0)
        out = project_co_optimize(action, soc, **KWARGS)
        assert (out[1:] >= -1e-6).all()

    def test_joint_upward_capacity(self):
        """p_discharge + regup + rrs + ecrs <= P_max."""
        action = torch.tensor([8.0, 5.0, 0.0, 5.0, 5.0, 0.0])
        soc = torch.tensor(10.0)
        out = project_co_optimize(action, soc, **KWARGS)
        p_dch = torch.clamp(out[0], min=0)
        assert p_dch + out[1] + out[3] + out[4] <= P_MAX + 0.01

    def test_joint_downward_capacity(self):
        """p_charge + regdn <= P_max."""
        action = torch.tensor([-8.0, 0.0, 8.0, 0.0, 0.0, 0.0])
        soc = torch.tensor(10.0)
        out = project_co_optimize(action, soc, **KWARGS)
        p_ch = torch.clamp(-out[0], min=0)
        assert p_ch + out[2] <= P_MAX + 0.01

    def test_soc_duration_nsrs(self):
        """10 MW battery at 50% SoC can't offer 10 MW NSRS (needs 40 MWh, only 8 available)."""
        action = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 10.0])
        soc = torch.tensor(10.0)  # 50% of 20 MWh
        out = project_co_optimize(action, soc, **KWARGS)
        # Available SoC above min = 10 - 2 = 8 MWh
        # NSRS needs 4 MWh/MW, so max NSRS = 8/4 = 2 MW
        assert out[5] <= 2.01

    def test_soc_at_min(self):
        """At SoC_min, no discharge or upward AS possible."""
        action = torch.tensor([5.0, 5.0, 0.0, 5.0, 5.0, 5.0])
        soc = torch.tensor(E_MAX * SOC_MIN)
        out = project_co_optimize(action, soc, **KWARGS)
        assert out[0] <= 0.01  # can't discharge
        # AS offers requiring SoC should be ~0
        assert out[1] <= 0.01
        assert out[3] <= 0.01

    def test_soc_at_max(self):
        """At SoC_max, can't charge."""
        action = torch.tensor([-10.0, 0.0, 5.0, 0.0, 0.0, 0.0])
        soc = torch.tensor(E_MAX * SOC_MAX)
        out = project_co_optimize(action, soc, **KWARGS)
        assert out[0] >= -0.01

    def test_p_max_fully_allocated(self):
        """When P_max is fully used for energy, AS upward should be ~0."""
        action = torch.tensor([10.0, 5.0, 0.0, 5.0, 5.0, 0.0])
        soc = torch.tensor(10.0)
        out = project_co_optimize(action, soc, **KWARGS)
        p_dch = torch.clamp(out[0], min=0)
        upward_as = out[1] + out[3] + out[4]
        assert p_dch + upward_as <= P_MAX + 0.01

    def test_batch(self):
        actions = torch.randn(8, 6)
        socs = torch.full((8,), 10.0)
        out = project_co_optimize(actions, socs, **KWARGS)
        assert out.shape == (8, 6)

    def test_differentiable(self):
        action = torch.randn(4, 6, requires_grad=True)
        soc = torch.tensor([10.0, 10.0, 10.0, 10.0])
        out = project_co_optimize(action, soc, **KWARGS)
        loss = out.sum()
        loss.backward()
        assert action.grad is not None

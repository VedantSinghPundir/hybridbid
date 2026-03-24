"""
Differentiable feasibility projection for battery actions.

All operations use torch tensors for gradient flow through the projection.
Critics see projected (feasible) actions, not raw actor outputs.
"""

import torch


# AS duration requirements: MWh per MW of capacity offered
AS_DURATION_MWH_PER_MW = torch.tensor([
    0.5,  # RegUp (30-min sustain)
    0.5,  # RegDn (30-min sustain)
    0.5,  # RRS (30-min sustain)
    1.0,  # ECRS (1-hour sustain)
    4.0,  # NSRS (4-hour sustain)
])

DELTA_T_HOURS = 5.0 / 60.0  # 5-minute interval


def project_energy_only(
    action: torch.Tensor,
    soc: torch.Tensor,
    p_max: float,
    e_max: float,
    soc_min_frac: float = 0.10,
    soc_max_frac: float = 0.90,
    eta_ch: float = 0.92,
    eta_dch: float = 0.92,
) -> torch.Tensor:
    """
    Project a scalar energy action to respect battery constraints.

    Parameters
    ----------
    action : Tensor (...,) or (..., 1)
        Net power in [-P_max, P_max]. Positive = discharge, negative = charge.
    soc : Tensor (...)
        Current SoC in MWh.
    p_max : float
        Maximum power [MW].
    e_max, soc_min_frac, soc_max_frac, eta_ch, eta_dch : float
        Battery parameters.

    Returns
    -------
    Tensor same shape as action — projected net power.
    """
    soc_min = soc_min_frac * e_max
    soc_max = soc_max_frac * e_max
    dt = DELTA_T_HOURS

    # Clip to power limits
    p_net = torch.clamp(action, -p_max, p_max)

    # Max charge rate (negative p_net = charging): limited by room to fill
    max_charge_power = torch.clamp((soc_max - soc) / (eta_ch * dt), min=0.0, max=p_max)

    # Max discharge rate (positive p_net): limited by energy available
    max_discharge_power = torch.clamp((soc - soc_min) * eta_dch / dt, min=0.0, max=p_max)

    # Apply SoC-aware limits
    p_net = torch.clamp(p_net, -max_charge_power, max_discharge_power)

    return p_net


def project_co_optimize(
    action: torch.Tensor,
    soc: torch.Tensor,
    p_max: float,
    e_max: float,
    soc_min_frac: float = 0.10,
    soc_max_frac: float = 0.90,
    eta_ch: float = 0.92,
    eta_dch: float = 0.92,
) -> torch.Tensor:
    """
    Project a 6D co-optimization action to respect all constraints.

    Parameters
    ----------
    action : Tensor (..., 6)
        [p_net, regup, regdn, rrs, ecrs, nsrs]
    soc : Tensor (...)
        Current SoC in MWh.

    Returns
    -------
    Tensor (..., 6) — projected feasible action.
    """
    soc_min = soc_min_frac * e_max
    soc_max = soc_max_frac * e_max
    dt = DELTA_T_HOURS

    p_net = action[..., 0]
    as_offers = action[..., 1:6]  # regup, regdn, rrs, ecrs, nsrs

    # Clip AS offers to >= 0
    as_offers = torch.clamp(as_offers, min=0.0)

    # Clip p_net to power limits
    p_net = torch.clamp(p_net, -p_max, p_max)

    # Decompose p_net into charge/discharge
    p_discharge = torch.clamp(p_net, min=0.0)
    p_charge = torch.clamp(-p_net, min=0.0)

    regup = as_offers[..., 0]
    regdn = as_offers[..., 1]
    rrs = as_offers[..., 2]
    ecrs = as_offers[..., 3]
    nsrs = as_offers[..., 4]

    # --- Joint upward capacity: p_discharge + regup + rrs + ecrs <= P_max ---
    upward_total = p_discharge + regup + rrs + ecrs
    upward_excess = torch.clamp(upward_total - p_max, min=0.0)
    upward_as = regup + rrs + ecrs
    # Scale down upward AS proportionally when violated
    scale_up = torch.where(
        upward_as > 1e-8,
        torch.clamp(1.0 - upward_excess / upward_as, min=0.0),
        torch.ones_like(upward_as),
    )
    regup = regup * scale_up
    rrs = rrs * scale_up
    ecrs = ecrs * scale_up

    # --- Joint downward capacity: p_charge + regdn <= P_max ---
    downward_total = p_charge + regdn
    downward_excess = torch.clamp(downward_total - p_max, min=0.0)
    scale_dn = torch.where(
        regdn > 1e-8,
        torch.clamp(1.0 - downward_excess / regdn, min=0.0),
        torch.ones_like(regdn),
    )
    regdn = regdn * scale_dn

    # --- SoC duration requirements ---
    # Available SoC above minimum
    soc_available = torch.clamp(soc - soc_min, min=0.0)

    dur = AS_DURATION_MWH_PER_MW.to(action.device)
    as_stack = torch.stack([regup, regdn, rrs, ecrs, nsrs], dim=-1)
    required_soc = (as_stack * dur).sum(dim=-1)

    # If required > available, scale down proportionally
    soc_scale = torch.where(
        required_soc > 1e-8,
        torch.clamp(soc_available / required_soc, min=0.0, max=1.0),
        torch.ones_like(required_soc),
    )
    as_stack = as_stack * soc_scale.unsqueeze(-1)

    regup = as_stack[..., 0]
    regdn = as_stack[..., 1]
    rrs = as_stack[..., 2]
    ecrs = as_stack[..., 3]
    nsrs = as_stack[..., 4]

    # --- Apply SoC bounds for energy (same as Stage 1) ---
    max_charge_power = torch.clamp((soc_max - soc) / (eta_ch * dt), min=0.0, max=p_max)
    max_discharge_power = torch.clamp((soc - soc_min) * eta_dch / dt, min=0.0, max=p_max)
    p_net = torch.clamp(p_net, -max_charge_power, max_discharge_power)

    # Reassemble
    result = torch.stack([p_net, regup, regdn, rrs, ecrs, nsrs], dim=-1)
    return result

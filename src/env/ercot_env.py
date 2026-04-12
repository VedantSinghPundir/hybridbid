"""
ERCOT Battery Bidding Gymnasium Environment.

Supports two modes:
  - energy_only (Stage 1): 4D action [mode(3) + energy_mag(1)]
  - co_optimize (Stage 2): 9D action [mode(3) + energy_mag(1) + as_mags(5)]

Each episode = 1 operating day (288 five-minute steps).

Reward follows Li et al. (2024) Eq. 26-30:
  - Spot market reward with EMA arbitrage shaping (τ_S=0.9, β_S=10)
  - No degradation cost in step reward (not in paper's Eq. 30)
  - Episode terminates with -50 penalty when SoC would violate limits
"""

import glob
import os
from typing import Optional

import gymnasium as gym
import numpy as np
import pandas as pd
import yaml

from src.models.feasibility import project_energy_only, project_co_optimize
from src.utils.battery_sim import BatteryParams


# Battery defaults — Li et al. Table I uses η=0.95
DEFAULT_BATTERY = dict(
    p_max=10.0, e_max=20.0,
    soc_min_frac=0.10, soc_max_frac=0.90, soc_initial_frac=0.50,
    eta_ch=0.95, eta_dch=0.95,
    degradation_cost=2.0,
)

DELTA_T_HOURS = 5.0 / 60.0
STEPS_PER_DAY = 288
SEQ_LEN = 32

# Li et al. Eq. 26 reward parameters
EMA_TAU = 0.9     # τ_S: EMA smoothing factor
BETA_ARB = 10.0   # β_S: EMA arbitrage bonus coefficient
SOC_PENALTY = 50.0  # penalty for SoC limit violation

# Action layout (mode one-hot always first 3 dims)
MODE_CHARGE = 0
MODE_DISCHARGE = 1
MODE_IDLE = 2

# Price vector column ordering (12 dims)
PRICE_COLS = [
    "rt_lmp",
    "rt_mcpc_regup", "rt_mcpc_regdn", "rt_mcpc_rrs", "rt_mcpc_ecrs", "rt_mcpc_nsrs",
    "dam_spp",
    "dam_as_regup", "dam_as_regdn", "dam_as_rrs", "dam_as_ecrs", "dam_as_nsrs",
]
N_PRICES = len(PRICE_COLS)  # 12

# System condition columns (7 dims)
SYSTEM_COLS = [
    "total_load_mw", "load_forecast_mw",
    "wind_actual_mw", "wind_forecast_mw",
    "solar_actual_mw", "solar_forecast_mw",
    "net_load_mw",
]


class ERCOTBatteryEnv(gym.Env):
    """
    Gymnasium environment for ERCOT battery bidding.

    Observation: dict with price_history (seq_len, 12) and static_features (14,)
    Action:
      energy_only:  Box(-1, 1, (4,))  — [mode(3), energy_mag(1)]
      co_optimize:  Box(-1, 1, (9,))  — [mode(3), energy_mag(1), as_mags(5)]
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        data_dir: str,
        mode: str = "energy_only",
        battery_config: Optional[dict] = None,
        seq_len: int = SEQ_LEN,
        date_range: Optional[tuple] = None,
    ):
        """
        Parameters
        ----------
        data_dir : str
            Path to processed/ directory containing energy_prices/, as_prices/, system_conditions/
        mode : str
            'energy_only' or 'co_optimize'
        battery_config : dict, optional
            Battery parameters. Uses defaults if not provided.
        seq_len : int
            TTFE lookback window length.
        date_range : tuple of (start_date, end_date) strings, optional
            Filter data to this date range.
        """
        super().__init__()
        assert mode in ("energy_only", "co_optimize")
        self.mode = mode
        self.seq_len = seq_len

        # Battery config
        bc = {**DEFAULT_BATTERY, **(battery_config or {})}
        self.p_max = bc["p_max"]
        self.e_max = bc["e_max"]
        self.soc_min_frac = bc["soc_min_frac"]
        self.soc_max_frac = bc["soc_max_frac"]
        self.soc_initial_frac = bc["soc_initial_frac"]
        self.eta_ch = bc["eta_ch"]
        self.eta_dch = bc["eta_dch"]
        self.degradation_cost = bc["degradation_cost"]

        self.soc_min = self.soc_min_frac * self.e_max
        self.soc_max = self.soc_max_frac * self.e_max

        # Action/observation spaces
        # Stage 1: [mode(3) + energy_mag(1)] = 4D
        # Stage 2: [mode(3) + energy_mag(1) + as_mags(5)] = 9D
        action_dim = 4 if mode == "energy_only" else 9
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Dict({
            "price_history": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(seq_len, N_PRICES), dtype=np.float32
            ),
            "static_features": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32
            ),
        })

        # Load and merge data
        self._load_data(data_dir, date_range)

        # State
        self.soc = self.soc_initial_frac * self.e_max
        self.current_step = 0
        self.current_day_idx = 0
        self.day_starts = []
        self._build_day_index()
        self.ema_price = 0.0

    def _load_data(self, data_dir: str, date_range: Optional[tuple]):
        """Load and merge all three Parquet tables."""
        ep_dir = os.path.join(data_dir, "energy_prices")
        ap_dir = os.path.join(data_dir, "as_prices")
        sc_dir = os.path.join(data_dir, "system_conditions")

        ep = self._read_parquets(ep_dir)
        ap = self._read_parquets(ap_dir)
        sc = self._read_parquets(sc_dir)

        # Drop is_post_rtcb — not used in observation
        for df in [ep, ap, sc]:
            if "is_post_rtcb" in df.columns:
                df.drop(columns=["is_post_rtcb"], inplace=True)

        # Merge on index
        merged = ep.join(ap, how="outer").join(sc, how="outer")

        # Filter date range
        if date_range:
            start, end = date_range
            merged = merged.loc[start:end]

        # Fill NaN: prices with 0, system conditions with forward fill then 0
        merged[PRICE_COLS] = merged[PRICE_COLS].fillna(0.0)
        merged[SYSTEM_COLS] = merged[SYSTEM_COLS].ffill().fillna(0.0)

        if len(merged) < self.seq_len:
            raise ValueError(f"Not enough data: {len(merged)} rows < seq_len {self.seq_len}")

        self.data = merged
        self.timestamps = merged.index

        self.price_data = merged[PRICE_COLS].values.astype(np.float32)
        self.system_data = merged[SYSTEM_COLS].values.astype(np.float32)

        # Normalize system conditions for observation
        self._system_scales = np.array([
            50000, 50000,  # load
            15000, 15000,  # wind
            10000, 10000,  # solar
            40000,         # net load
        ], dtype=np.float32)

    def _read_parquets(self, directory: str) -> pd.DataFrame:
        """Read all Parquet files in a directory and concatenate."""
        files = sorted(glob.glob(os.path.join(directory, "*.parquet")))
        if not files:
            raise FileNotFoundError(f"No Parquet files in {directory}")
        dfs = [pd.read_parquet(f) for f in files]
        return pd.concat(dfs).sort_index()

    def _build_day_index(self):
        """Build index of day start positions in the data array."""
        dates = pd.Series(self.timestamps.date).unique()
        self.day_starts = []
        for d in dates:
            day_mask = self.timestamps.date == d
            day_indices = np.where(day_mask)[0]
            if len(day_indices) >= STEPS_PER_DAY:
                first_idx = day_indices[0]
                if first_idx >= self.seq_len:
                    self.day_starts.append(first_idx)

        if not self.day_starts:
            raise ValueError("No complete days with sufficient lookback found in data")

    def _get_time_features(self, idx: int) -> np.ndarray:
        """Compute 6 cyclical time features for a given data index."""
        ts = self.timestamps[idx]
        if hasattr(ts, 'tz_convert'):
            ts_local = ts.tz_convert("US/Central")
        else:
            ts_local = ts

        hour = ts_local.hour + ts_local.minute / 60.0
        dow = ts_local.dayofweek
        month = ts_local.month

        features = np.array([
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * dow / 7),
            np.cos(2 * np.pi * dow / 7),
            np.sin(2 * np.pi * month / 12),
            np.cos(2 * np.pi * month / 12),
        ], dtype=np.float32)
        return features

    def _get_observation(self, idx: int) -> dict:
        """Build observation dict for current step."""
        start = idx - self.seq_len + 1
        price_history = self.price_data[start:idx + 1].copy()

        system = self.system_data[idx] / self._system_scales
        time_feats = self._get_time_features(idx)
        soc_frac = np.array([self.soc / self.e_max], dtype=np.float32)
        static_features = np.concatenate([system, time_feats, soc_frac])  # (14,)

        return {
            "price_history": price_history,
            "static_features": static_features,
        }

    def reset(self, seed=None, options=None):
        """Reset environment to start of next day."""
        super().reset(seed=seed)

        if options and "day_idx" in options:
            self.current_day_idx = options["day_idx"]

        self.current_day_idx = self.current_day_idx % len(self.day_starts)
        self.current_step = 0
        self.soc = self.soc_initial_frac * self.e_max  # fixed at 50% per paper

        self._day_start_idx = self.day_starts[self.current_day_idx]

        # Initialize EMA with first RT LMP of the day
        self.ema_price = float(self.price_data[self._day_start_idx, 0])

        obs = self._get_observation(self._day_start_idx)

        self.current_day_idx += 1

        return obs, {}

    def _parse_action(self, action: np.ndarray):
        """
        Parse action array into (mode, energy_mag, [as_mags]).

        Action layout:
          [:3]  mode one-hot (or soft) — argmax gives hard mode
          [3]   energy magnitude ∈ (-1, 1) — abs gives [0,1]
          [4:]  AS magnitudes ∈ (-1, 1) — abs gives [0,1] (co_optimize only)
        """
        mode = int(np.argmax(action[:3]))
        energy_mag = float(np.abs(action[3]))  # always non-negative
        as_mags = None
        if self.mode == "co_optimize":
            as_mags = np.abs(action[4:9])  # [0, 1] each
        return mode, energy_mag, as_mags

    def step(self, action: np.ndarray):
        """
        Execute one 5-minute step.

        Parameters
        ----------
        action : np.ndarray, shape (4,) for energy_only or (9,) for co_optimize

        Returns
        -------
        observation, reward, terminated, truncated, info
        """
        import torch

        data_idx = self._day_start_idx + self.current_step
        dt = DELTA_T_HOURS

        mode, energy_mag, as_mags = self._parse_action(action)

        # --- Compute intended p_net from mode + magnitude ---
        a_S_t = energy_mag * self.p_max  # bid power in MW
        if mode == MODE_DISCHARGE:
            v_dch, v_ch = 1.0, 0.0
            p_net_raw = a_S_t
        elif mode == MODE_CHARGE:
            v_dch, v_ch = 0.0, 1.0
            p_net_raw = -a_S_t
        else:  # IDLE
            v_dch, v_ch = 0.0, 0.0
            p_net_raw = 0.0

        # --- Check SoC violation BEFORE projection (Li et al. termination) ---
        if p_net_raw >= 0:  # discharging or idle
            soc_new_raw = self.soc - (p_net_raw / self.eta_dch) * dt
        else:  # charging
            soc_new_raw = self.soc + abs(p_net_raw) * self.eta_ch * dt

        soc_violated = soc_new_raw < self.soc_min or soc_new_raw > self.soc_max

        # --- Apply feasibility projection (safety net regardless of violation) ---
        if self.mode == "energy_only":
            p_net_t = torch.tensor(p_net_raw, dtype=torch.float32)
            soc_t = torch.tensor(self.soc, dtype=torch.float32)
            p_net_proj = project_energy_only(
                p_net_t, soc_t,
                p_max=self.p_max, e_max=self.e_max,
                soc_min_frac=self.soc_min_frac, soc_max_frac=self.soc_max_frac,
                eta_ch=self.eta_ch, eta_dch=self.eta_dch,
            ).item()

            projected_action = np.array([p_net_proj], dtype=np.float32)

        else:  # co_optimize
            as_phys = as_mags * self.p_max  # MW
            co_action = np.concatenate([[p_net_raw], as_phys])
            action_t = torch.tensor(co_action, dtype=torch.float32)
            soc_t = torch.tensor(self.soc, dtype=torch.float32)
            proj_t = project_co_optimize(
                action_t, soc_t,
                p_max=self.p_max, e_max=self.e_max,
                soc_min_frac=self.soc_min_frac, soc_max_frac=self.soc_max_frac,
                eta_ch=self.eta_ch, eta_dch=self.eta_dch,
            )
            proj = proj_t.detach().numpy()
            p_net_proj = proj[0]
            projected_action = proj

        # --- Update SoC from projected action ---
        if p_net_proj >= 0:  # discharging
            energy_out = p_net_proj / self.eta_dch * dt
            self.soc -= energy_out
        else:  # charging
            energy_in = abs(p_net_proj) * self.eta_ch * dt
            self.soc += energy_in

        # Safety clamp
        self.soc = np.clip(self.soc, self.soc_min, self.soc_max)

        # --- Reward: Li et al. Eq. 26 ---
        rt_lmp = float(self.price_data[data_idx, 0])

        # EMA update: ρ̄_t = τ * ρ̄_{t-1} + (1-τ) * ρ_t
        self.ema_price = EMA_TAU * self.ema_price + (1.0 - EMA_TAU) * rt_lmp

        # Direction indicators (Li et al. Eq. 25)
        # I_dch = sgn(ρ - ρ̄): 1 when price above EMA (good to discharge)
        # I_ch  = sgn(ρ̄ - ρ): 1 when price below EMA (good to charge)
        I_dch = np.sign(rt_lmp - self.ema_price)
        I_ch = np.sign(self.ema_price - rt_lmp)

        price_dev = abs(rt_lmp - self.ema_price)

        # Spot market reward (Eq. 26)
        # r_S = a_S * ρ * (v_dch * η_dch - v_ch / η_ch) * Δt
        #     + β_S * a_S * |ρ - ρ̄| * (I_dch * v_dch * η_dch + I_ch * v_ch / η_ch) * Δt
        #
        # a_S = energy_mag ∈ [0, 1] p.u. — NOT MW (Li et al. Eq. 26 uses normalized bid)
        # Δt = 5/60 converts $/h rate to $ per interval
        #
        # Checkpoint: energy_mag=1.0, rt_lmp=$100/MWh →
        #   energy_term = 1.0 × 100 × 0.95 × (5/60) = $7.92  (not $950)
        energy_term = energy_mag * rt_lmp * (v_dch * self.eta_dch - v_ch / self.eta_ch) * dt
        timing_bonus = (
            BETA_ARB * energy_mag * price_dev
            * (I_dch * v_dch * self.eta_dch + I_ch * v_ch / self.eta_ch) * dt
        )
        reward = energy_term + timing_bonus

        # AS revenue (Stage 2 only — not in Stage 1 Eq. 30)
        as_rev = 0.0
        if self.mode == "co_optimize":
            for i, mcpc_idx in enumerate([1, 2, 3, 4, 5]):
                as_rev += projected_action[i + 1] * float(self.price_data[data_idx, mcpc_idx]) * dt
            reward += as_rev

        # SoC violation: penalty + termination (Li et al. Section III.D)
        terminated = False
        if soc_violated:
            reward -= SOC_PENALTY
            terminated = True

        # Advance step
        self.current_step += 1
        truncated = self.current_step >= STEPS_PER_DAY

        # Build next observation
        if not truncated:
            next_data_idx = self._day_start_idx + self.current_step
            obs = self._get_observation(next_data_idx)
        else:
            obs = self._get_observation(data_idx)

        info = {
            "energy_revenue": float(energy_term),
            "timing_bonus": float(timing_bonus),
            "as_revenue": float(as_rev),
            "soc": float(self.soc),
            "p_net": float(p_net_proj),
            "mode": mode,
            "a_S_t": float(a_S_t),
            "ema_price": float(self.ema_price),
            "soc_violated": soc_violated,
            "raw_action": action.copy(),
            "projected_action": projected_action,
        }

        return obs, float(reward), terminated, truncated, info

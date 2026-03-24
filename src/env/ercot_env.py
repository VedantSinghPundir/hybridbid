"""
ERCOT Battery Bidding Gymnasium Environment.

Supports two modes:
  - energy_only (Stage 1): 1D action, energy arbitrage only
  - co_optimize (Stage 2): 6D action, energy + ancillary services

Each episode = 1 operating day (288 five-minute steps).
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


# Battery defaults
DEFAULT_BATTERY = dict(
    p_max=10.0, e_max=20.0,
    soc_min_frac=0.10, soc_max_frac=0.90, soc_initial_frac=0.50,
    eta_ch=0.92, eta_dch=0.92,
    degradation_cost=2.0,
)

DELTA_T_HOURS = 5.0 / 60.0
STEPS_PER_DAY = 288
SEQ_LEN = 32

# Price vector column ordering (12 dims)
# CLAUDE.md says 11-dim but lists 12 items (1+5+1+5). Using all 12.
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

    Observation: dict with price_history (seq_len, 11) and static_features (14,)
    Action: Box in [-1, 1], dim 1 (energy_only) or 6 (co_optimize)
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
        action_dim = 1 if mode == "energy_only" else 6
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
        self.day_starts = []  # indices into self.data where each day begins
        self._build_day_index()

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

        # Drop rows where we don't have enough history for the first window
        if len(merged) < self.seq_len:
            raise ValueError(f"Not enough data: {len(merged)} rows < seq_len {self.seq_len}")

        self.data = merged
        self.timestamps = merged.index

        # Pre-extract arrays for fast access
        self.price_data = merged[PRICE_COLS].values.astype(np.float32)  # (T, 11)
        self.system_data = merged[SYSTEM_COLS].values.astype(np.float32)  # (T, 7)

        # Normalize system conditions for observation (scale to ~[-1, 1] range)
        # Use simple division by typical magnitudes
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
                # Only include full days where we have enough lookback
                first_idx = day_indices[0]
                if first_idx >= self.seq_len:
                    self.day_starts.append(first_idx)

        if not self.day_starts:
            raise ValueError("No complete days with sufficient lookback found in data")

    def _get_time_features(self, idx: int) -> np.ndarray:
        """Compute 6 cyclical time features for a given data index."""
        ts = self.timestamps[idx]
        # Convert to CPT for ERCOT hour
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
        # Price history window: (seq_len, 11)
        start = idx - self.seq_len + 1
        price_history = self.price_data[start:idx + 1].copy()  # (seq_len, 11)

        # Static features: system(7) + time(6) + soc(1) = 14
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
        # else use current day_idx (sequential)

        self.current_day_idx = self.current_day_idx % len(self.day_starts)
        self.current_step = 0
        self.soc = self.soc_initial_frac * self.e_max

        # Set data index to start of this day
        self._day_start_idx = self.day_starts[self.current_day_idx]

        obs = self._get_observation(self._day_start_idx)

        # Advance to next day for next episode
        self.current_day_idx += 1

        return obs, {}

    def step(self, action: np.ndarray):
        """
        Execute one 5-minute step.

        Parameters
        ----------
        action : np.ndarray in [-1, 1], shape (1,) or (6,)

        Returns
        -------
        observation, reward, terminated, truncated, info
        """
        import torch

        data_idx = self._day_start_idx + self.current_step
        dt = DELTA_T_HOURS

        # Scale action from [-1, 1] to physical units
        action = np.clip(action, -1.0, 1.0)

        if self.mode == "energy_only":
            # Scale to [-P_max, P_max]
            p_net_raw = float(action[0]) * self.p_max

            # Apply feasibility projection
            p_net_t = torch.tensor(p_net_raw, dtype=torch.float32)
            soc_t = torch.tensor(self.soc, dtype=torch.float32)
            p_net_proj = project_energy_only(
                p_net_t, soc_t,
                p_max=self.p_max, e_max=self.e_max,
                soc_min_frac=self.soc_min_frac, soc_max_frac=self.soc_max_frac,
                eta_ch=self.eta_ch, eta_dch=self.eta_dch,
            ).item()

            projected_action = np.array([p_net_proj], dtype=np.float32)

            # Compute SoC change
            if p_net_proj >= 0:  # discharging
                energy_out = p_net_proj / self.eta_dch * dt
                self.soc -= energy_out
            else:  # charging
                energy_in = abs(p_net_proj) * self.eta_ch * dt
                self.soc += energy_in

            # Revenue: energy only
            rt_lmp = self.price_data[data_idx, 0]  # rt_lmp
            energy_rev = p_net_proj * rt_lmp * dt
            as_rev = 0.0
            throughput = abs(p_net_proj) * dt
            deg_cost = self.degradation_cost * throughput
            reward = energy_rev - deg_cost

            info = {
                "energy_revenue": energy_rev,
                "as_revenue": 0.0,
                "degradation_cost": deg_cost,
                "soc": self.soc,
                "p_net": p_net_proj,
                "raw_action": action.copy(),
                "projected_action": projected_action,
            }

        else:  # co_optimize
            # Scale: p_net in [-P_max, P_max], AS in [0, P_max]
            scaled = np.zeros(6, dtype=np.float32)
            scaled[0] = float(action[0]) * self.p_max  # p_net
            scaled[1:] = (np.clip(action[1:], 0, 1)) * self.p_max  # AS offers (only positive)

            action_t = torch.tensor(scaled, dtype=torch.float32)
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

            # SoC update
            if p_net_proj >= 0:
                energy_out = p_net_proj / self.eta_dch * dt
                self.soc -= energy_out
            else:
                energy_in = abs(p_net_proj) * self.eta_ch * dt
                self.soc += energy_in

            # Revenue
            rt_lmp = self.price_data[data_idx, 0]
            energy_rev = p_net_proj * rt_lmp * dt

            # AS revenue: capacity * MCPC * dt
            # RT MCPC indices: 1=regup, 2=regdn, 3=rrs, 4=ecrs, 5=nsrs
            as_rev = 0.0
            for i, mcpc_idx in enumerate([1, 2, 3, 4, 5]):
                as_rev += proj[i + 1] * self.price_data[data_idx, mcpc_idx] * dt

            throughput = abs(p_net_proj) * dt
            deg_cost = self.degradation_cost * throughput
            reward = energy_rev + as_rev - deg_cost

            info = {
                "energy_revenue": energy_rev,
                "as_revenue": as_rev,
                "degradation_cost": deg_cost,
                "soc": self.soc,
                "p_net": p_net_proj,
                "raw_action": action.copy(),
                "projected_action": projected_action,
            }

        # Clamp SoC (safety)
        self.soc = np.clip(self.soc, self.soc_min, self.soc_max)

        # Advance step
        self.current_step += 1

        # Check episode end (288 steps = 1 day)
        terminated = self.current_step >= STEPS_PER_DAY
        truncated = False

        # Build next observation
        if not terminated:
            next_data_idx = self._day_start_idx + self.current_step
            obs = self._get_observation(next_data_idx)
        else:
            obs = self._get_observation(data_idx)  # terminal obs

        return obs, float(reward), terminated, truncated, info

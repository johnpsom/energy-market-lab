"""Weather features — Layer 4.

Reads the long `weather` table, aggregates the Greek sites by role (demand centres vs
wind vs solar regions), and derives load/renewable drivers. Physics (turbine & panel power
curves) is encoded explicitly so the model doesn't have to relearn non-linear response from
scarce data. All features use *forecast* weather → leakage-safe.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..db import read_sql

# Site roles. Weights approximate the demand distribution; wind/solar sites are where the
# resource actually is. Refine later with installed-capacity-weighted aggregation.
DEMAND_WEIGHTS = {"athens": 0.45, "thessaloniki": 0.25, "peloponnese": 0.12,
                  "crete": 0.10, "cyclades": 0.08}
WIND_SITES = ["cyclades", "crete", "peloponnese"]
SOLAR_SITES = ["peloponnese", "crete", "athens", "thessaloniki"]

# Base temperatures for degree-hours (°C).
HDD_BASE, CDD_BASE = 18.0, 22.0
# Turbine power curve (m/s @100 m hub height).
CUT_IN, RATED, CUT_OUT = 3.0, 12.0, 25.0


def load_weather_wide() -> pd.DataFrame:
    """Load the weather table pivoted to columns '<location>__<variable>', ts index (naive local)."""
    long = read_sql("select ts, location, variable, value from weather")
    if long.empty:
        return pd.DataFrame()
    long["ts"] = pd.to_datetime(long["ts"])
    wide = long.pivot_table(index="ts", columns=["location", "variable"], values="value")
    wide.columns = [f"{loc}__{var}" for loc, var in wide.columns]
    return wide.sort_index()


def _wind_power_curve(v: pd.Series) -> pd.Series:
    """Map 100 m wind speed to a normalized capacity factor [0, 1] via a turbine power curve."""
    cf = pd.Series(0.0, index=v.index)
    ramp = (v >= CUT_IN) & (v < RATED)
    cf[ramp] = (v[ramp] ** 3 - CUT_IN ** 3) / (RATED ** 3 - CUT_IN ** 3)
    cf[(v >= RATED) & (v < CUT_OUT)] = 1.0
    return cf.clip(0, 1)


def build(wide: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return weather features indexed by hourly ts. Pass a preloaded wide frame or let it load."""
    if wide is None:
        wide = load_weather_wide()
    if wide.empty:
        return pd.DataFrame()

    def col(loc, var):
        return wide.get(f"{loc}__{var}")

    def agg(sites, var, weights=None):
        cols = {s: col(s, var) for s in sites if col(s, var) is not None}
        if not cols:
            return pd.Series(np.nan, index=wide.index)
        frame = pd.DataFrame(cols)
        if weights:
            w = pd.Series({s: weights[s] for s in frame.columns})
            return (frame * w).sum(axis=1) / w.sum()
        return frame.mean(axis=1)

    f = pd.DataFrame(index=wide.index)

    # --- demand-side (temperature) ---
    temp_demand = agg(list(DEMAND_WEIGHTS), "temperature_2m", DEMAND_WEIGHTS)
    f["temp_demand"] = temp_demand
    f["hdd"] = (HDD_BASE - temp_demand).clip(lower=0)   # heating degree-hours
    f["cdd"] = (temp_demand - CDD_BASE).clip(lower=0)   # cooling degree-hours

    # --- wind supply ---
    wind_speed = agg(WIND_SITES, "wind_speed_100m")
    f["wind_speed_agg"] = wind_speed
    f["wind_power_proxy"] = _wind_power_curve(wind_speed)
    f["wind_ramp_1h"] = f["wind_power_proxy"].diff(1)
    f["wind_ramp_3h"] = f["wind_power_proxy"].diff(3)
    f["wind_persistence_24h"] = f["wind_power_proxy"].rolling(24, min_periods=6).std()

    # --- solar supply ---
    rad = agg(SOLAR_SITES, "shortwave_radiation")
    temp_solar = agg(SOLAR_SITES, "temperature_2m")
    f["solar_rad_agg"] = rad
    # panels lose ~0.4 %/°C above 25 °C cell temp (ambient proxy).
    derate = (1 - 0.004 * (temp_solar - 25)).clip(lower=0.6, upper=1.0)
    f["solar_power_proxy"] = (rad / 1000.0).clip(lower=0) * derate
    f["cloud_index"] = agg(SOLAR_SITES, "cloud_cover")

    # --- context ---
    f["precip_agg"] = agg(list(DEMAND_WEIGHTS), "precipitation")
    return f

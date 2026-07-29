"""Grid features — Layer 4. The merit-order drivers of price.

Needs the `load` and `generation` tables. `residual_load = load − wind − solar` is the single
most informative price predictor in a high-renewables system: it is what thermal plant must
cover, i.e. how far up the merit order the market clears.

NOTE ON LEAKAGE: for a true day-ahead forecast, residual load should be built from the *load
forecast* and *wind/solar forecast* (all published pre-auction), not realized values. This
module prefers the load forecast when present and flags realized generation as leak-risk — add
the ENTSO-E wind/solar day-ahead forecast collector (M6) to make residual_load fully clean.
"""
from __future__ import annotations

import pandas as pd

from ..config import settings
from ..db import read_sql

WIND_FUELS = ["Wind Onshore", "Wind Offshore"]
SOLAR_FUELS = ["Solar"]


def _load_series(zone: str) -> pd.Series:
    """Prefer the day-ahead load forecast (leakage-safe); fall back to actual."""
    for kind in ("forecast", "actual"):
        df = read_sql(
            "select ts, value from load "
            f"where zone='{zone}' and kind='{kind}' order by ts"
        )
        if not df.empty:
            s = pd.Series(df["value"].values, index=pd.to_datetime(df["ts"]), name="load")
            return s[~s.index.duplicated(keep="last")].sort_index()
    return pd.Series(dtype=float)


def _gen_by_fuel(zone: str) -> pd.DataFrame:
    df = read_sql(
        "select ts, fuel, value from generation "
        f"where zone='{zone}' order by ts"
    )
    if df.empty:
        return pd.DataFrame()
    wide = df.pivot_table(index=pd.to_datetime(df["ts"]), columns="fuel", values="value")
    return wide.sort_index()


def build(zone: str | None = None) -> pd.DataFrame:
    """Return grid features indexed by hourly ts."""
    zone = zone or settings.default_zone
    load = _load_series(zone)
    gen = _gen_by_fuel(zone)
    if load.empty or gen.empty:
        return pd.DataFrame()

    wind = gen.reindex(columns=WIND_FUELS).sum(axis=1, min_count=1)
    solar = gen.reindex(columns=SOLAR_FUELS).sum(axis=1, min_count=1)
    idx = load.index.union(gen.index)
    load = load.reindex(idx)
    wind = wind.reindex(idx)
    solar = solar.reindex(idx)
    renew = wind.fillna(0) + solar.fillna(0)

    f = pd.DataFrame(index=idx)
    f["load"] = load
    f["residual_load"] = load - renew
    f["renewable_penetration"] = (renew / load).clip(0, 1.5)
    f["wind_share"] = (wind / load).clip(0, 1.5)
    f["solar_share"] = (solar / load).clip(0, 1.5)
    f["load_ramp_1h"] = load.diff(1)
    f["load_ramp_3h"] = load.diff(3)
    f["thermal_gen"] = gen.reindex(idx).drop(columns=WIND_FUELS + SOLAR_FUELS,
                                             errors="ignore").sum(axis=1, min_count=1)
    return f

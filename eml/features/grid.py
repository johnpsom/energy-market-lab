"""Grid features — Layer 4. The merit-order drivers of price.

Needs the `load` and `generation` tables. `residual_load = load − wind − solar` is the single
most informative price predictor in a high-renewables system: it is what thermal plant must
cover, i.e. how far up the merit order the market clears.

LEAKAGE: a day-ahead forecast must use pre-auction information. So residual load is built from the
day-ahead **load forecast** and the day-ahead **wind/solar forecast** (source priority:
entsoe_fc → weather_fc → actual fallback), not realized generation. Thermal generation (a realized,
leak-risk feature) is kept only as context and is naturally absent in the forecast horizon.
"""
from __future__ import annotations

import pandas as pd

from ..config import settings
from ..db import read_sql

WIND_FUELS = ["Wind Onshore", "Wind Offshore"]
SOLAR_FUELS = ["Solar"]
# source preference for renewables: ENTSO-E day-ahead forecast, then weather-derived, then actual
RES_SOURCE_PRIORITY = {"entsoe_fc": 0, "weather_fc": 1, "entsoe": 2}


def _load_series(zone: str) -> pd.Series:
    """Prefer the day-ahead load forecast (leakage-safe); fall back to actual."""
    for kind in ("forecast", "actual"):
        df = read_sql(
            "select ts, value from load "
            f"where zone='{zone}' and kind='{kind}' order by ts"
        )
        if not df.empty:
            s = pd.Series(df["value"].values, index=pd.to_datetime(df["ts"])).sort_index()
            return s.resample("h").mean().rename("load").dropna()   # 15-min -> hourly
    return pd.Series(dtype=float)


def _res_series(zone: str, fuels: list[str]) -> pd.Series:
    """Forecast-preferred hourly renewable output for `fuels`: use the day-ahead forecast where it
    exists, fall back to weather-derived, then to realized generation — per timestamp."""
    fl = "','".join(fuels)
    df = read_sql("select ts, value, source from generation "
                  f"where zone='{zone}' and fuel in ('{fl}')")
    if df.empty:
        return pd.Series(dtype=float)
    df["ts"] = pd.to_datetime(df["ts"])
    df["prio"] = df["source"].map(RES_SOURCE_PRIORITY).fillna(9)
    # per hourly timestamp, keep the highest-priority source, then sum the fuels within it
    out = None
    for _, g in df.groupby("prio"):
        s = g.pivot_table(index="ts", columns="source", values="value", aggfunc="sum")
        s = s.sum(axis=1, min_count=1).resample("h").mean()
        out = s if out is None else out.combine_first(s)
    return out.dropna()


def _thermal_actual(zone: str) -> pd.Series:
    """Realized thermal generation (context only): actual generation minus wind/solar."""
    fl = "','".join(WIND_FUELS + SOLAR_FUELS)
    df = read_sql("select ts, fuel, value from generation "
                  f"where zone='{zone}' and source='entsoe' and fuel not in ('{fl}')")
    if df.empty:
        return pd.Series(dtype=float)
    wide = df.pivot_table(index=pd.to_datetime(df["ts"]), columns="fuel", values="value")
    return wide.resample("h").mean().sum(axis=1, min_count=1)


def build(zone: str | None = None) -> pd.DataFrame:
    """Return grid features indexed by hourly ts."""
    zone = zone or settings.default_zone
    load = _load_series(zone)
    wind = _res_series(zone, WIND_FUELS)
    solar = _res_series(zone, SOLAR_FUELS)
    if load.empty or (wind.empty and solar.empty):
        return pd.DataFrame()

    idx = load.index.union(wind.index).union(solar.index)
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
    f["thermal_gen"] = _thermal_actual(zone).reindex(idx)
    return f

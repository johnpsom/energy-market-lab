"""Cross-border features — Layer 4. Neighbor price levels + spreads (roadmap 1).

Greece is a net importer, so the day-ahead price in coupled neighbors (Italy-South, Bulgaria) and
the GR-neighbor spread strongly shape the GR price. Coupled markets clear simultaneously, so a
neighbor's price for delivery day D is NOT known before GR gate closure — hence all neighbor
features are LAGGED (D-1, D-7), which capture the persistent level and spread. Values are carried
forward across the forecast horizon (like the price AR features).
"""
from __future__ import annotations

import pandas as pd

from ..db import read_sql
from . import price as price_feat

NEIGHBORS = {"IT_SUD": "it", "BG": "bg"}


def _neighbor_series(zone: str) -> pd.Series:
    df = read_sql(f"select ts, value from prices where product='day_ahead' and zone='{zone}' order by ts")
    if df.empty:
        return pd.Series(dtype=float)
    s = pd.Series(df["value"].values, index=pd.to_datetime(df["ts"])).sort_index()
    return s.resample("h").mean().dropna()      # 15-min -> hourly, matches GR


def build(index: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    gr = price_feat.load_price_series()
    series = {pfx: _neighbor_series(z) for z, pfx in NEIGHBORS.items()}
    if gr.empty or all(s.empty for s in series.values()):
        return pd.DataFrame()

    # continuous hourly index extended ~8 days ahead so lags exist across the forecast horizon
    starts = [gr.index.min()] + [s.index.min() for s in series.values() if not s.empty]
    ends = [gr.index.max()] + [s.index.max() for s in series.values() if not s.empty]
    full = pd.date_range(min(starts), max(ends) + pd.Timedelta(days=8), freq="h")
    gr = gr.reindex(full)
    it = series["it"].reindex(full)
    bg = series["bg"].reindex(full)

    f = pd.DataFrame(index=full)
    f["it_price_lag24"] = it.shift(24)
    f["it_price_lag168"] = it.shift(168)
    f["bg_price_lag24"] = bg.shift(24)
    f["bg_price_lag168"] = bg.shift(168)
    f["gr_it_spread_lag24"] = gr.shift(24) - it.shift(24)
    f["gr_bg_spread_lag24"] = gr.shift(24) - bg.shift(24)
    # cheapest available import source yesterday — the effective price cap
    f["neighbor_min_lag24"] = pd.concat([it.shift(24), bg.shift(24)], axis=1).min(axis=1)
    f = f.ffill()
    return f if index is None else f.reindex(index)

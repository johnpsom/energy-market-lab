"""Price features — Layer 4. Autoregressive structure of the DAM price.

Needs the `prices` table (product='day_ahead'). Lags are ≥24 h so they respect the leakage
rule: when forecasting delivery day D (auction closes ~12:00 D-1), the same-hour price from
D-1 and earlier is known. Downstream alignment is the model's job; these are the raw signals.
"""
from __future__ import annotations

import pandas as pd

from ..config import settings
from ..db import read_sql


def load_price_series(zone: str | None = None) -> pd.Series:
    """Day-ahead price as an hourly Series (EUR/MWh), ts index (naive local)."""
    zone = zone or settings.default_zone
    df = read_sql(
        "select ts, value from prices "
        f"where product='day_ahead' and zone='{zone}' order by ts"
    )
    if df.empty:
        return pd.Series(dtype=float)
    s = pd.Series(df["value"].values, index=pd.to_datetime(df["ts"]), name="dam_price")
    return s[~s.index.duplicated(keep="last")].sort_index()


def build(price: pd.Series | None = None, zone: str | None = None) -> pd.DataFrame:
    """Return price features indexed by hourly ts."""
    if price is None:
        price = load_price_series(zone)
    if price.empty:
        return pd.DataFrame()

    f = pd.DataFrame(index=price.index)
    f["price_lag_24h"] = price.shift(24)
    f["price_lag_48h"] = price.shift(48)
    f["price_lag_168h"] = price.shift(168)
    # rolling stats shifted by 24 h so nothing from the target day leaks in
    f["price_roll_mean_24h"] = price.rolling(24, min_periods=12).mean().shift(24)
    f["price_roll_mean_168h"] = price.rolling(168, min_periods=48).mean().shift(24)
    f["price_roll_std_24h"] = price.rolling(24, min_periods=12).std().shift(24)      # volatility
    f["price_roll_skew_168h"] = price.rolling(168, min_periods=48).skew().shift(24)  # spike regime
    f["price_momentum_24h"] = (price - price.shift(24)).shift(24)
    f["price_accel"] = f["price_momentum_24h"].diff(24)
    return f

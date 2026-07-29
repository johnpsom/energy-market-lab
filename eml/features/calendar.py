"""Calendar features — deterministic demand/price cycles. All features are leakage-safe
(known arbitrarily far ahead)."""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import holidays as _holidays
except ImportError:  # pragma: no cover
    _holidays = None

PEAK_START, PEAK_END = 8, 20  # HEnEx peak block, local hours [08:00, 20:00)


def _cyc(values: pd.Series, period: int, name: str) -> pd.DataFrame:
    """Sine/cosine encoding so the wrap-around is continuous (23:00 ≈ 00:00)."""
    ang = 2 * np.pi * values / period
    return pd.DataFrame({f"{name}_sin": np.sin(ang), f"{name}_cos": np.cos(ang)},
                        index=values.index)


def build(index: pd.DatetimeIndex, country: str = "GR") -> pd.DataFrame:
    """Return calendar features for an hourly DatetimeIndex (local time)."""
    idx = pd.DatetimeIndex(index)
    df = pd.DataFrame(index=idx)
    df["hour"] = idx.hour
    df["dayofweek"] = idx.dayofweek
    df["month"] = idx.month
    df["day_of_year"] = idx.dayofyear
    df["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    df["is_peak"] = ((idx.hour >= PEAK_START) & (idx.hour < PEAK_END)
                     & (idx.dayofweek < 5)).astype(int)
    # meteorological season: 0 winter (DJF), 1 spring, 2 summer, 3 autumn
    df["season"] = (idx.month % 12 // 3)

    # Greek public holidays, including moveable Orthodox Easter.
    if _holidays is not None:
        years = range(idx.year.min(), idx.year.max() + 1)
        cal = _holidays.country_holidays(country, years=list(years))
        hol_dates = set(cal.keys())
        dates = idx.normalize().date
        is_hol = np.array([d in hol_dates for d in dates], dtype=int)
        df["is_holiday"] = is_hol
        # bridge day: a workday with a holiday on one side and a weekend on the other
        prev_hol = np.array([(d - pd.Timedelta(days=1)).date() in hol_dates for d in idx])
        next_hol = np.array([(d + pd.Timedelta(days=1)).date() in hol_dates for d in idx])
        weekday = idx.dayofweek < 5
        df["is_bridge_day"] = (weekday & (is_hol == 0)
                               & ((idx.dayofweek == 0) & prev_hol      # Mon after holiday Sun
                                  | (idx.dayofweek == 4) & next_hol     # Fri before holiday Sat
                                  | prev_hol | next_hol)).astype(int)
    else:  # pragma: no cover
        df["is_holiday"] = 0
        df["is_bridge_day"] = 0

    for col, period in (("hour", 24), ("dayofweek", 7), ("day_of_year", 365)):
        df = df.join(_cyc(df[col], period, col))
    return df

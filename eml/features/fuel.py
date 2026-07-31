"""Fuel & carbon features — Layer 4. The price-LEVEL drivers.

Reads daily TTF gas, EUA carbon, and Brent, and derives the short-run marginal cost (SRMC) of a
gas plant — the level around which the power price clears in most hours. Crucially these features
contain NO power price, so they are valid model INPUTS (unlike spark/dark spreads, which embed the
power price and are outputs, not inputs).

Leakage: daily settlements are lagged one trading day (the prior close is known before the D-1
day-ahead auction) then forward-filled to every hour.
"""
from __future__ import annotations

import pandas as pd

from ..db import read_sql

# Gas CCGT assumptions for SRMC (EUR/MWh_e).
ETA_GAS = 0.50            # electrical efficiency
EF_GAS_TH = 0.20          # tCO2 per MWh_thermal of gas


def load_fuel_wide() -> pd.DataFrame:
    """Daily wide frame: index date, columns per commodity (gas_ttf, carbon_eua, brent)."""
    df = read_sql("select ts, commodity, value from fuel")
    if df.empty:
        return pd.DataFrame()
    df["ts"] = pd.to_datetime(df["ts"])
    wide = df.pivot_table(index="ts", columns="commodity", values="value")
    return wide.sort_index()


def build(index: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """Return hourly fuel features. If `index` is given, reindex onto it (the feature matrix)."""
    wide = load_fuel_wide()
    if wide.empty:
        return pd.DataFrame()

    # lag one trading day so the value used for delivery day D is knowable at gate closure D-1
    daily = wide.shift(1)
    daily = daily.ffill()

    f = pd.DataFrame(index=daily.index)
    gas = daily.get("gas_ttf")
    carbon = daily.get("carbon_eua")
    if gas is not None:
        f["gas_price"] = gas
        f["gas_mom_7d"] = gas.diff(5)                     # ~1 trading week
    if carbon is not None:
        f["carbon_price"] = carbon
    if gas is not None and carbon is not None:
        # short-run marginal cost of a CCGT (EUR/MWh_e): fuel + carbon
        f["gas_srmc"] = (gas + carbon * EF_GAS_TH) / ETA_GAS
    if daily.get("brent") is not None:
        f["brent_price"] = daily["brent"]

    # daily -> hourly, aligned to the requested (naive local) index
    if index is None:
        return f
    idx = pd.DatetimeIndex(index)
    hourly = f.reindex(f.index.union(idx)).ffill().reindex(idx)
    return hourly

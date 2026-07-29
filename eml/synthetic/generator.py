"""Transparent fundamental model: real weather -> load, generation, DAM price.

The relationships are deliberately simple and physical so that (a) the synthetic price is a
genuine function of the real weather features, and (b) a well-trained model + SHAP should
recover the drivers we put in (residual load, wind, solar, temperature, hour). Every row is
written with source='synthetic'; purge these and re-pull with the ENTSO-E collector to go live.

This is a BRIDGE, not a forecast target of record — it exists to exercise the pipeline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import settings
from ..db import upsert
from ..features import calendar as cal_feat
from ..features import weather as wx_feat

# Installed capacity (GR-ish, MW).
WIND_CAP = 5000.0
SOLAR_CAP = 6500.0
THERMAL_REF = 7000.0            # scale for the merit-order curve

# Typical normalized daily load shape (24 values, 0..1), double-peaked (morning + evening).
_HOUR_SHAPE = np.array([
    0.55, 0.50, 0.47, 0.46, 0.48, 0.55, 0.66, 0.78, 0.86, 0.90, 0.92, 0.93,
    0.94, 0.92, 0.90, 0.89, 0.90, 0.93, 0.97, 1.00, 0.98, 0.88, 0.74, 0.62,
])
BASE_LOAD = 3900.0
LOAD_AMPLITUDE = 2600.0


def _synthesize(feat: pd.DataFrame, cal: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    idx = feat.index
    n = len(idx)

    # --- LOAD (MW): daily shape * calendar factors + weather (cooling/heating) response ---
    shape = _HOUR_SHAPE[cal["hour"].to_numpy()]
    weekend = np.where(cal["is_weekend"].to_numpy() == 1, 0.88, 1.0)
    holiday = np.where(cal["is_holiday"].to_numpy() == 1, 0.85, 1.0)
    load = BASE_LOAD + LOAD_AMPLITUDE * shape
    load *= weekend * holiday
    load += 95.0 * feat["cdd"].to_numpy()      # cooling MW per cooling degree-hour
    load += 60.0 * feat["hdd"].to_numpy()      # heating
    load += rng.normal(0, 90, n)               # unexplained variation
    load = np.clip(load, 2200, None)

    # --- RENEWABLE GENERATION (MW) from the physical power-curve proxies ---
    wind = feat["wind_power_proxy"].to_numpy() * WIND_CAP * rng.normal(1.0, 0.05, n)
    solar = feat["solar_power_proxy"].to_numpy() * SOLAR_CAP * rng.normal(1.0, 0.05, n)
    wind = np.clip(wind, 0, WIND_CAP)
    solar = np.clip(solar, 0, SOLAR_CAP)
    renew = wind + solar

    # --- RESIDUAL LOAD -> merit-order PRICE (EUR/MWh) ---
    residual = np.clip(load - renew, 200, None)
    x = (residual - 2800.0) / 5000.0    # ~0 at soft floor, ~1.0 in tight hours
    gas_level = 58.0 + 12.0 * np.sin(np.arange(n) / (24 * 30) * 2 * np.pi)  # slow fuel drift
    price = gas_level + 65.0 * x + 55.0 * np.tanh(1.8 * x)                  # convex S-curve
    price += 620.0 * np.maximum(x - 0.6, 0.0) ** 1.7                        # scarcity spikes
    price += rng.normal(0, 6, n)
    # negative-price regime: renewables flood a low-residual system
    pen = renew / np.clip(load, 1, None)
    flood = (pen > 0.85) & (residual < 2600)
    price = np.where(flood, price - rng.uniform(20, 90, n), price)
    price = np.clip(price, -120, 500)

    # --- allocate residual across thermal/hydro so generation ~ load ---
    hydro = np.clip(0.12 * residual, 0, 1600)
    gas = 0.60 * (residual - hydro)
    coal = 0.28 * (residual - hydro)
    net_other = residual - hydro - gas - coal   # imports/other, absorbs remainder

    return pd.DataFrame({
        "ts": idx,
        "load": load,
        "Wind Onshore": wind,
        "Solar": solar,
        "Hydro Water Reservoir": hydro,
        "Fossil Gas": gas,
        "Fossil Hard coal": coal,
        "Other": net_other,
        "price": price,
    })


def generate(start: str, end: str, seed: int = 42) -> dict:
    """Build weather features over [start, end], synthesize fundamentals, write to warehouse.
    Returns a summary dict."""
    zone = settings.default_zone
    rng = np.random.default_rng(seed)

    wide = wx_feat.load_weather_wide()
    wide = wide.loc[(wide.index >= pd.Timestamp(start)) & (wide.index < pd.Timestamp(end))]
    if wide.empty:
        raise SystemExit(f"No weather in warehouse for [{start}, {end}). "
                         "Run scripts/pull_history.py first.")
    feat = wx_feat.build(wide)
    cal = cal_feat.build(feat.index)
    syn = _synthesize(feat, cal, rng).dropna()

    # --- write PRICES ---
    prices = syn[["ts", "price"]].rename(columns={"price": "value"})
    prices["zone"], prices["product"], prices["source"] = zone, "day_ahead", "synthetic"
    upsert("prices", prices)

    # --- write LOAD (forecast + actual, forecast carries a small error) ---
    load_actual = syn[["ts", "load"]].rename(columns={"load": "value"})
    load_actual["zone"], load_actual["kind"], load_actual["source"] = zone, "actual", "synthetic"
    load_fc = load_actual.copy()
    load_fc["kind"] = "forecast"
    load_fc["value"] = load_fc["value"] * rng.normal(1.0, 0.02, len(load_fc))
    upsert("load", load_actual)
    upsert("load", load_fc)

    # --- write GENERATION (long format per fuel) ---
    fuels = ["Wind Onshore", "Solar", "Hydro Water Reservoir", "Fossil Gas",
             "Fossil Hard coal", "Other"]
    gen = syn.melt(id_vars="ts", value_vars=fuels, var_name="fuel", value_name="value")
    gen["zone"], gen["source"] = zone, "synthetic"
    upsert("generation", gen)

    return {
        "rows": len(syn),
        "span": (str(syn["ts"].min()), str(syn["ts"].max())),
        "price_stats": {
            "mean": round(float(syn["price"].mean()), 1),
            "min": round(float(syn["price"].min()), 1),
            "max": round(float(syn["price"].max()), 1),
            "pct_negative": round(float((syn["price"] < 0).mean() * 100), 2),
            "pct_over_150": round(float((syn["price"] > 150).mean() * 100), 2),
        },
    }

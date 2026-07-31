"""Forecast verification — score frozen forecasts against realized prices.

Reads the forecasts frozen in the `forecasts` table (by `price_forecast.train`, or by
`scripts/forecast.py` in live operation), joins them to the realized DAM prices, and computes
the track-record scorecard: point accuracy (MAE/RMSE/bias), interval honesty (P10-P90 coverage),
quantile quality (pinball), skill vs a naive persistence baseline, spike-probability calibration
(Brier + reliability), and the day-by-day series the dashboard plots.

This is the prospective, out-of-sample test that matters most: the forecast was frozen before
the outcome existed, so scoring it here cannot leak.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import settings
from ..db import read_sql
from ..features import price as price_feat
from .price_forecast import BACKTEST_MODEL, QUANTILES, SPIKE_EUR, WF_MODEL


def load_frozen(zone: str, model: str = WF_MODEL) -> pd.DataFrame:
    """Frozen forecasts as a wide frame: index target_ts, cols p10/p50/p90/spike_prob/neg_prob."""
    df = read_sql(
        "select target_ts, target, quantile, value from forecasts "
        f"where model='{model}' and zone='{zone}'"
    )
    if df.empty:
        return pd.DataFrame()
    df["target_ts"] = pd.to_datetime(df["target_ts"])
    q = df[df["target"] == "dam_price"].pivot_table(
        index="target_ts", columns="quantile", values="value")
    q.columns = [f"p{int(c*100)}" for c in q.columns]
    ev = df[df["target"].isin(["spike_prob", "neg_prob"])].pivot_table(
        index="target_ts", columns="target", values="value")
    return q.join(ev).sort_index()


def _pinball(y, pred, q):
    d = y - pred
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def scorecard(zone: str | None = None) -> dict:
    """Full verification scorecard for the frozen forecasts vs realized prices."""
    zone = zone or settings.default_zone
    fc = load_frozen(zone, WF_MODEL)
    if fc.empty:                                   # fall back to the single-split backtest freeze
        fc = load_frozen(zone, BACKTEST_MODEL)
    if fc.empty:
        return {"available": False}
    actual = price_feat.load_price_series(zone).rename("actual")

    df = fc.join(actual, how="inner").dropna(subset=["p50", "actual"])

    # HONESTY CUTOFF: only score days whose outcome is genuinely realized. A delivery day can
    # only be verified once it is in the past — you cannot grade a forecast against a price that
    # has not happened yet. (The synthetic bridge fabricates future prices too; this line stops
    # them leaking into the track record. In live operation the prices table simply has no
    # future rows, so this is a no-op safeguard.)
    realized_before = pd.Timestamp.now(tz=settings.timezone).normalize().tz_localize(None)
    df = df[df.index < realized_before]
    if df.empty:
        return {"available": False}

    err = df["actual"] - df["p50"]
    inside = (df["actual"] >= df["p10"]) & (df["actual"] <= df["p90"])

    # naive persistence baseline: same hour, previous day
    persist = actual.shift(24).reindex(df.index)
    pmask = persist.notna()
    persist_mae = float((actual.reindex(df.index)[pmask] - persist[pmask]).abs().mean())
    model_mae = float(err.abs().mean())
    skill = (1 - model_mae / persist_mae) * 100 if persist_mae else 0.0

    spike_actual = (df["actual"] > SPIKE_EUR).astype(int)
    brier = float(((df.get("spike_prob", 0.0) - spike_actual) ** 2).mean()) \
        if "spike_prob" in df else None

    # day-by-day, with 7-day rolling accuracy & coverage
    daily = pd.DataFrame({
        "mae": err.abs().groupby(df.index.normalize()).mean(),
        "cov": inside.groupby(df.index.normalize()).mean() * 100,
        "actual_avg": df["actual"].groupby(df.index.normalize()).mean(),
        "fcst_avg": df["p50"].groupby(df.index.normalize()).mean(),
    }).round(2)
    daily["mae7"] = daily["mae"].rolling(7, min_periods=1).mean().round(2)
    daily["cov7"] = daily["cov"].rolling(7, min_periods=1).mean().round(1)
    daily = daily.astype(object).where(pd.notna(daily), None)  # NaN -> null for JSON

    # spike-probability reliability (10 bins)
    reliability = []
    if "spike_prob" in df:
        bins = np.clip((df["spike_prob"] * 10).astype(int), 0, 9)
        for b, g in spike_actual.groupby(bins):
            reliability.append({"bin": int(b) * 10 + 5,
                                "predicted": round(float(df["spike_prob"][g.index].mean()) * 100, 1),
                                "observed": round(float(g.mean()) * 100, 1),
                                "n": int(g.size)})

    return {
        "available": True,
        "zone": zone,
        "window": [str(df.index.min()), str(df.index.max())],
        "n_hours": int(len(df)),
        "n_days": int(daily.shape[0]),
        "mae": round(model_mae, 2),
        "rmse": round(float(np.sqrt((err ** 2).mean())), 2),
        "bias": round(float(err.mean()), 2),
        "coverage_pct": round(float(inside.mean() * 100), 1),
        "pinball": {str(q): round(_pinball(df["actual"].to_numpy(),
                                           df[f"p{int(q*100)}"].to_numpy(), q), 2)
                    for q in QUANTILES},
        "persistence_mae": round(persist_mae, 2),
        "skill_pct": round(skill, 1),
        "spike_brier": round(brier, 4) if brier is not None else None,
        "daily": [{"date": str(d.date()), **row} for d, row in daily.iterrows()],
        "reliability": reliability,
    }

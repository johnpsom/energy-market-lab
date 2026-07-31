"""Morning brief + driver attribution + sensitivity — Layer 6.

For a chosen delivery day: forecast the 24 hourly prices (P10/P50/P90, spike & negative-price
probabilities), attribute the day's average to its top drivers via SHAP, run a wind-sensitivity
what-if, and render it all as a readable brief.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..features.build import build_matrix
from ..models import price_forecast as pf

# Human-readable driver names (feature -> label). Anything unmapped falls back to the raw name.
FRIENDLY = {
    "residual_load": "residual load (demand net of wind & solar)",
    "load": "system load",
    "temp_demand": "temperature",
    "cdd": "cooling demand",
    "hdd": "heating demand",
    "wind_power_proxy": "wind generation",
    "wind_speed_agg": "wind speed",
    "solar_power_proxy": "solar generation",
    "solar_rad_agg": "solar irradiance",
    "cloud_index": "cloud cover",
    "renewable_penetration": "renewable share of demand",
    "wind_share": "wind share",
    "solar_share": "solar share",
    "price_lag_24h": "yesterday's price at this hour",
    "price_lag_48h": "price two days ago",
    "price_lag_168h": "price last week",
    "price_roll_mean_24h": "recent 24h price level",
    "price_roll_mean_168h": "recent weekly price level",
    "price_roll_std_24h": "recent price volatility",
    "price_momentum_24h": "price momentum",
    "is_peak": "peak-hour timing",
    "hour": "time of day",
    "is_weekend": "weekend demand",
    "is_holiday": "holiday demand",
    "thermal_gen": "thermal generation",
}
WIND_FEATURES = ["wind_power_proxy", "wind_speed_agg", "wind_share",
                 "wind_ramp_1h", "wind_ramp_3h", "renewable_penetration"]


def _label(feat: str) -> str:
    return FRIENDLY.get(feat, feat.replace("_", " "))


def _shap_drivers(booster, X: pd.DataFrame, features: list[str], top: int = 5) -> list[dict]:
    """Mean SHAP contribution per feature over the day (EUR/MWh), ranked by magnitude."""
    contrib = booster.predict(X, pred_contrib=True)          # (n, n_features + 1)
    mean_contrib = contrib[:, :-1].mean(axis=0)              # drop base value
    order = np.argsort(np.abs(mean_contrib))[::-1][:top]
    return [{"feature": features[i], "label": _label(features[i]),
             "eur": round(float(mean_contrib[i]), 1),
             "direction": "raising" if mean_contrib[i] > 0 else "lowering"}
            for i in order]


def _wind_sensitivity(X: pd.DataFrame, models: dict, drop: float = 0.3) -> dict:
    """What-if: cut wind generation by `drop` and re-price. Reports the mean price impact."""
    base = pf.predict(X, models)["p50"].mean()
    Xw = X.copy()
    for f in WIND_FEATURES:
        if f in Xw.columns:
            if f in ("wind_power_proxy", "wind_speed_agg", "wind_share", "renewable_penetration"):
                Xw[f] = Xw[f] * (1 - drop)
    shocked = pf.predict(Xw, models)["p50"].mean()
    return {"drop_pct": int(drop * 100),
            "delta_eur": round(float(shocked - base), 1),
            "base_eur": round(float(base), 1),
            "shocked_eur": round(float(shocked), 1)}


def generate(date: str | None = None, zone: str | None = None) -> dict:
    """Build the brief for a delivery day (default: last full day in the warehouse)."""
    matrix = build_matrix(zone)
    models = pf.load_models()

    if date is None:
        # Default = TOMORROW (the day-ahead delivery day) if it has a complete feature row;
        # otherwise the most recent complete day. "Complete" = grid + price history present,
        # i.e. not the weather-only forecast tail.
        core = [c for c in ("residual_load", "price_lag_168h") if c in matrix.columns]
        complete = matrix.dropna(subset=core) if core else matrix
        full_days = complete.groupby(complete.index.normalize()).size()
        complete_days = full_days[full_days >= 24].index
        tomorrow = pd.Timestamp.now(tz=pf.settings.timezone).normalize().tz_localize(None) \
            + pd.Timedelta(days=1)
        day = tomorrow if tomorrow in complete_days else complete_days.max()
    else:
        day = pd.Timestamp(date).normalize()
    X = matrix[matrix.index.normalize() == day]
    if X.empty:
        raise SystemExit(f"No feature rows for {day.date()}.")

    preds = pf.predict(X, models)
    hours = X.index.hour
    peak_i = int(preds["p50"].to_numpy().argmax())
    # evening peak (18:00-22:00) — the operationally-quoted peak, distinct from the daily max
    ev_mask = (hours >= 18) & (hours <= 22)
    ev_hour = ev_eur = None
    if ev_mask.any():
        ev = preds["p50"].to_numpy()[ev_mask]
        ev_i = int(ev.argmax())
        ev_hour = int(X.index[ev_mask][ev_i].hour)
        ev_eur = round(float(ev[ev_i]), 1)
    drivers = _shap_drivers(models["quantiles"][0.5], X[models["features"]],
                            models["features"])
    sens = _wind_sensitivity(X[models["features"]], models)

    summary = {
        "date": str(day.date()),
        "zone": zone or pf.settings.default_zone,
        "avg_eur": round(float(preds["p50"].mean()), 1),
        "peak_hour": int(X.index[peak_i].hour),
        "peak_eur": round(float(preds["p50"].iloc[peak_i]), 1),
        "evening_peak_hour": ev_hour,
        "evening_peak_eur": ev_eur,
        "min_eur": round(float(preds["p50"].min()), 1),
        "band_lo_eur": round(float(preds["p10"].mean()), 1),
        "band_hi_eur": round(float(preds["p90"].mean()), 1),
        "spike_prob_pct": round(float(preds["spike_prob"].max() * 100), 0),
        "neg_prob_pct": round(float(preds["neg_prob"].max() * 100), 0),
        "drivers": drivers,
        "wind_sensitivity": sens,
        "hourly": preds.assign(hour=X.index.hour).reset_index(drop=True).to_dict("records"),
    }
    summary["text"] = render(summary)
    return summary


def outlook(n_days: int = 7, zone: str | None = None) -> dict:
    """Hourly forecast (P10/P50/P90 + spike/neg prob) for the next `n_days` complete days,
    starting tomorrow. Powers the multi-day chart on the dashboard."""
    matrix = build_matrix(zone)
    models = pf.load_models()
    core = [c for c in ("residual_load", "price_lag_168h") if c in matrix.columns]
    complete = matrix.dropna(subset=core) if core else matrix
    full_days = complete.groupby(complete.index.normalize()).size()
    complete_days = sorted(full_days[full_days >= 24].index)
    tomorrow = pd.Timestamp.now(tz=pf.settings.timezone).normalize().tz_localize(None) \
        + pd.Timedelta(days=1)
    days = [d for d in complete_days if d >= tomorrow][:n_days]
    if not days:                                   # fall back to the most recent days
        days = complete_days[-n_days:]

    X = complete[complete.index.normalize().isin(days)]
    preds = pf.predict(X[models["features"]], models)
    preds.insert(0, "ts", [t.isoformat() for t in X.index])
    daily = preds.groupby(X.index.normalize()).agg(
        avg=("p50", "mean"), peak=("p50", "max"), low=("p50", "min")).round(1)
    # Days beyond ENTSO-E's day-ahead renewable-forecast horizon are weather-derived (indicative).
    from ..db import read_sql
    h = read_sql("select max(ts) m from generation where source='entsoe_fc'")["m"].iloc[0]
    horizon = pd.to_datetime(h).date() if h else None
    return {
        "days": [str(d.date()) for d in days],
        "hourly": preds.to_dict("records"),
        "daily": [{"date": str(d.date()),
                   "indicative": bool(horizon and d.date() > horizon), **row}
                  for d, row in daily.iterrows()],
        "entsoe_horizon": str(horizon) if horizon else None,
    }


def render(s: dict) -> str:
    """Plain-text morning brief from the structured summary."""
    up = [d for d in s["drivers"] if d["direction"] == "raising"][:3]
    down = [d for d in s["drivers"] if d["direction"] == "lowering"][:3]
    lines = [
        f"MORNING BRIEF - {s['zone']} day-ahead - {s['date']}",
        "",
        f"Expected average: EUR {s['avg_eur']}/MWh "
        f"(80% band EUR {s['band_lo_eur']} - {s['band_hi_eur']}).",
        f"Daily peak: {s['peak_hour']:02d}:00 at ~EUR {s['peak_eur']}/MWh"
        + (f"; evening peak {s['evening_peak_hour']:02d}:00 ~EUR {s['evening_peak_eur']}/MWh"
           if s.get("evening_peak_hour") is not None else "")
        + f"; daily low ~EUR {s['min_eur']}/MWh.",
        f"Risk: spike>EUR150 probability {s['spike_prob_pct']:.0f}% | "
        f"negative-price probability {s['neg_prob_pct']:.0f}%.",
        "",
        "Why:",
    ]
    for d in up:
        lines.append(f"  + {d['label']} is raising prices (+EUR {abs(d['eur'])}/MWh).")
    for d in down:
        lines.append(f"  - {d['label']} is lowering prices (-EUR {abs(d['eur'])}/MWh).")
    ws = s["wind_sensitivity"]
    lines += [
        "",
        f"What if wind falls {ws['drop_pct']}%? Expected average moves "
        f"{'+' if ws['delta_eur'] >= 0 else ''}{ws['delta_eur']} EUR/MWh "
        f"(EUR {ws['base_eur']} -> {ws['shocked_eur']}).",
    ]
    return "\n".join(lines)

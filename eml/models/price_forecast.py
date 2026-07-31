"""DAM price forecaster — Layer 5 flagship.

LightGBM gradient-boosted trees:
  * three quantile models (P10 / P50 / P90) -> point forecast + prediction interval
  * two binary classifiers -> P(spike > SPIKE_EUR) and P(negative price)

Evaluated with a time-ordered holdout (train on the past, test on the future — never shuffle
a timeseries). Reports MAE/RMSE on the median, pinball loss per quantile, and P10–P90 coverage.
Final artifacts are refit on all data and saved for the dashboard / narrative layers.

LightGBM natively handles NaN and outputs per-feature SHAP contributions (pred_contrib=True),
so the "why" layer needs no extra dependency.
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import sqlalchemy as sa

from ..config import ROOT, settings
from ..db import get_engine, upsert
from ..features import price as price_feat
from ..features.build import build_matrix

BACKTEST_MODEL = "lgbm_cqr"   # name under which frozen out-of-sample forecasts are stored

ARTIFACTS = ROOT / "models" / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

TARGET = "dam_price"
QUANTILES = (0.1, 0.5, 0.9)
SPIKE_EUR = 150.0

_QPARAMS = dict(objective="quantile", n_estimators=500, learning_rate=0.05,
                num_leaves=63, min_child_samples=40, subsample=0.8,
                subsample_freq=1, colsample_bytree=0.8, verbose=-1)
_CPARAMS = dict(objective="binary", n_estimators=400, learning_rate=0.05,
                num_leaves=63, min_child_samples=40, subsample=0.8,
                subsample_freq=1, colsample_bytree=0.8, verbose=-1)


# --- data assembly -------------------------------------------------------------

def training_frame(zone: str | None = None) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Return (X, y, feature_names) aligned on ts, target rows non-null."""
    zone = zone or settings.default_zone
    matrix = build_matrix(zone)
    target = price_feat.load_price_series(zone).rename(TARGET)
    df = matrix.join(target, how="inner").dropna(subset=[TARGET])
    features = [c for c in matrix.columns if c != TARGET]
    return df[features], df[TARGET], features


# --- metrics -------------------------------------------------------------------

def _pinball(y: np.ndarray, pred: np.ndarray, q: float) -> float:
    d = y - pred
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def _metrics(y: np.ndarray, preds: dict[float, np.ndarray]) -> dict:
    med = preds[0.5]
    lo, hi = preds[0.1], preds[0.9]
    return {
        "n": int(len(y)),
        "mae": round(float(np.mean(np.abs(y - med))), 2),
        "rmse": round(float(np.sqrt(np.mean((y - med) ** 2))), 2),
        "pinball": {str(q): round(_pinball(y, preds[q], q), 3) for q in QUANTILES},
        "p10_p90_coverage_pct": round(float(np.mean((y >= lo) & (y <= hi)) * 100), 1),
    }


# --- train ---------------------------------------------------------------------

def _fit_quantiles(X, y) -> dict[float, lgb.LGBMRegressor]:
    models = {}
    for q in QUANTILES:
        m = lgb.LGBMRegressor(alpha=q, **_QPARAMS)
        m.fit(X, y)
        models[q] = m
    return models


def _cqr_offset(qmods: dict, Xcal: pd.DataFrame, ycal: pd.Series, target_cov=0.8) -> float:
    """Conformalized Quantile Regression: the interval widening that guarantees ~target
    coverage on exchangeable data. Conformity score = max(p_lo - y, y - p_hi)."""
    lo = qmods[QUANTILES[0]].predict(Xcal)
    hi = qmods[QUANTILES[-1]].predict(Xcal)
    scores = np.maximum(lo - ycal.to_numpy(), ycal.to_numpy() - hi)
    n = len(scores)
    # finite-sample conformal level
    level = min(1.0, np.ceil((n + 1) * target_cov) / n)
    return float(np.quantile(scores, level, method="higher"))


def _apply_cqr(preds: dict, offset: float) -> dict:
    out = dict(preds)
    out[QUANTILES[0]] = preds[QUANTILES[0]] - offset
    out[QUANTILES[-1]] = preds[QUANTILES[-1]] + offset
    return out


def _persist_forecasts(index, preds: dict, spike_prob, neg_prob, zone: str,
                       model: str, run_ts, replace: bool = True) -> int:
    """Freeze a batch of forecasts into the `forecasts` table for later verification.
    Stores quantiles as target='dam_price' and the event probabilities as their own targets.
    replace=True wipes any prior rows for this model (one clean backtest history); replace=False
    accumulates (the live day-by-day loop)."""
    rows = []
    for i, ts in enumerate(index):
        for q in QUANTILES:
            rows.append((ts, zone, "dam_price", q, model, run_ts, float(preds[q][i])))
        rows.append((ts, zone, "spike_prob", -1.0, model, run_ts, float(spike_prob[i])))
        rows.append((ts, zone, "neg_prob", -1.0, model, run_ts, float(neg_prob[i])))
    df = pd.DataFrame(rows, columns=["target_ts", "zone", "target", "quantile",
                                     "model", "run_ts", "value"])
    if replace:
        with get_engine().begin() as conn:
            conn.execute(sa.text("DELETE FROM forecasts WHERE model=:m"), {"m": model})
    return upsert("forecasts", df)


WF_MODEL = "lgbm_wf"            # frozen walk-forward forecasts (the honest track record)
DEPLOY_WINDOW_DAYS = 540       # trailing training window (drops stale-regime old data)
CALIB_DAYS = 45                # recent held-out slice for bias + CQR calibration


def _calibrate(qmods: dict, Xcal: pd.DataFrame, ycal, target_cov: float = 0.8) -> tuple[float, float]:
    """Learn an additive bias correction and the CQR interval offset from a held-out slice.
    Returns (bias, offset). bias > 0 means the model systematically over-forecasts."""
    cs = np.sort(np.vstack([qmods[q].predict(Xcal) for q in QUANTILES]).T, axis=1)
    yc = np.asarray(ycal)
    bias = float(np.mean(cs[:, 1] - yc))
    lo, hi = cs[:, 0] - bias, cs[:, 2] - bias
    scores = np.maximum(lo - yc, yc - hi)
    n = len(scores)
    offset = float(np.quantile(scores, min(1.0, np.ceil((n + 1) * target_cov) / n), method="higher"))
    return bias, offset


def train(zone: str | None = None, window_days: int = DEPLOY_WINDOW_DAYS,
          calib_days: int = CALIB_DAYS) -> dict:
    """Deploy the production model: fit quantile + event models on a TRAILING window (so the
    forecast tracks the current price regime, not the 2023 crisis era), and learn bias + CQR
    from the most recent held-out slice. Saves artifacts. Verification is `walk_forward`."""
    X, y, features = training_frame(zone)
    last = X.index[-1]
    cal_start = last - pd.Timedelta(days=calib_days)
    win_start = cal_start - pd.Timedelta(days=window_days)

    # calibration models: trained up to cal_start, calibrated on the recent held-out slice
    tr_m = (X.index >= win_start) & (X.index < cal_start)
    cal_m = X.index >= cal_start
    qcal = _fit_quantiles(X[tr_m], y[tr_m])
    bias, offset = _calibrate(qcal, X[cal_m], y[cal_m])

    # deployment models: trailing window including the most recent data
    dcut = last - pd.Timedelta(days=window_days)
    Xd, yd = X[X.index >= dcut], y[X.index >= dcut]
    final_q = _fit_quantiles(Xd, yd)
    final_spike = lgb.LGBMClassifier(**_CPARAMS).fit(Xd, (yd > SPIKE_EUR).astype(int))
    final_neg = lgb.LGBMClassifier(**_CPARAMS).fit(Xd, (yd < 0).astype(int))

    for q, m in final_q.items():
        m.booster_.save_model(str(ARTIFACTS / f"price_q{int(q*100)}.txt"))
    final_spike.booster_.save_model(str(ARTIFACTS / "price_spike.txt"))
    final_neg.booster_.save_model(str(ARTIFACTS / "price_neg.txt"))
    (ARTIFACTS / "features.json").write_text(json.dumps(features, indent=2))
    (ARTIFACTS / "calibration.json").write_text(json.dumps({"cqr_offset": offset, "bias": bias}))
    return {"deploy_rows": int(len(Xd)),
            "deploy_span": (str(Xd.index.min()), str(last)),
            "calib_span": (str(cal_start), str(last)),
            "bias": round(bias, 2), "cqr_offset": round(offset, 2)}


def walk_forward(zone: str | None = None, window_days: int = DEPLOY_WINDOW_DAYS,
                 step_days: int = 21, calib_days: int = CALIB_DAYS) -> dict:
    """Walk-forward verification: repeatedly retrain on a trailing window and forecast the next
    block, so every prediction comes from a model that only saw prior data — the honest,
    regime-tracking out-of-sample test. Freezes all blocks as WF_MODEL and writes metrics.json."""
    X, y, features = training_frame(zone)
    idx = X.index
    t = (idx[0] + pd.Timedelta(days=window_days + calib_days)).normalize()
    last = idx[-1]
    parts, P, SP, NG = [], {q: [] for q in QUANTILES}, [], []
    blocks = 0
    while t < last:
        block_end = t + pd.Timedelta(days=step_days)
        cal_start = t - pd.Timedelta(days=calib_days)
        win_start = cal_start - pd.Timedelta(days=window_days)
        tr_m = (idx >= win_start) & (idx < cal_start)
        cal_m = (idx >= cal_start) & (idx < t)
        te_m = (idx >= t) & (idx < block_end)
        if tr_m.sum() < 2000 or cal_m.sum() < 200 or te_m.sum() == 0:
            t = block_end
            continue
        qmods = _fit_quantiles(X[tr_m], y[tr_m])
        bias, off = _calibrate(qmods, X[cal_m], y[cal_m])
        ts = np.sort(np.vstack([qmods[q].predict(X[te_m]) for q in QUANTILES]).T, axis=1)
        P[0.1].append(ts[:, 0] - bias - off)
        P[0.5].append(ts[:, 1] - bias)
        P[0.9].append(ts[:, 2] - bias + off)
        ytr = y[tr_m]
        SP.append(lgb.LGBMClassifier(**_CPARAMS).fit(X[tr_m], (ytr > SPIKE_EUR).astype(int))
                  .predict_proba(X[te_m])[:, 1])
        NG.append(lgb.LGBMClassifier(**_CPARAMS).fit(X[tr_m], (ytr < 0).astype(int))
                  .predict_proba(X[te_m])[:, 1])
        parts.append(X[te_m].index)
        blocks += 1
        t = block_end

    if not parts:
        return {"blocks": 0}
    index = parts[0]
    for p in parts[1:]:
        index = index.append(p)
    preds = {q: np.concatenate(P[q]) for q in QUANTILES}
    spike = np.concatenate(SP)
    neg = np.concatenate(NG)
    _persist_forecasts(index, preds, spike, neg, zone or settings.default_zone,
                       WF_MODEL, pd.Timestamp.utcnow().tz_localize(None), replace=True)

    yv = y.reindex(index).to_numpy()
    m = _metrics(yv, preds)
    m.update(method="walk-forward", blocks=blocks, window_days=window_days, step_days=step_days,
             bias_eur=round(float(np.mean(yv - preds[0.5])), 2),
             test_span=(str(index.min()), str(index.max())))
    (ARTIFACTS / "metrics.json").write_text(json.dumps(m, indent=2))
    return m


# --- load & predict ------------------------------------------------------------

def load_models() -> dict:
    feats = json.loads((ARTIFACTS / "features.json").read_text())
    boosters = {q: lgb.Booster(model_file=str(ARTIFACTS / f"price_q{int(q*100)}.txt"))
                for q in QUANTILES}
    spike = lgb.Booster(model_file=str(ARTIFACTS / "price_spike.txt"))
    neg = lgb.Booster(model_file=str(ARTIFACTS / "price_neg.txt"))
    calib = json.loads((ARTIFACTS / "calibration.json").read_text())
    return {"features": feats, "quantiles": boosters, "spike": spike, "neg": neg,
            "cqr_offset": calib["cqr_offset"], "bias": calib.get("bias", 0.0)}


def predict(X: pd.DataFrame, models: dict | None = None) -> pd.DataFrame:
    """Return per-row p10/p50/p90/spike_prob/neg_prob for a feature frame.
    Intervals are CQR-widened for calibrated ~80% coverage."""
    models = models or load_models()
    X = X[models["features"]]
    q = models["quantiles"]
    stack = np.sort(np.vstack([q[qq].predict(X) for qq in QUANTILES]).T, axis=1)
    off = models.get("cqr_offset", 0.0)
    bias = models.get("bias", 0.0)
    out = pd.DataFrame(index=X.index)
    out["p10"] = stack[:, 0] - bias - off
    out["p50"] = stack[:, 1] - bias
    out["p90"] = stack[:, 2] - bias + off
    out["spike_prob"] = models["spike"].predict(X)
    out["neg_prob"] = models["neg"].predict(X)
    return out

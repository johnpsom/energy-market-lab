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

from ..config import ROOT, settings
from ..features import price as price_feat
from ..features.build import build_matrix

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


def train(zone: str | None = None, test_frac: float = 0.3, calib_frac: float = 0.15) -> dict:
    """Time-ordered train / calibrate / test backtest with CQR-calibrated intervals, then
    refit on all data and save artifacts (+ the calibration offset)."""
    X, y, features = training_frame(zone)
    n = len(X)
    s_cal = int(n * (1 - test_frac - calib_frac))
    s_test = int(n * (1 - test_frac))
    Xtr, ytr = X.iloc[:s_cal], y.iloc[:s_cal]
    Xcal, ycal = X.iloc[s_cal:s_test], y.iloc[s_cal:s_test]
    Xte, yte = X.iloc[s_test:], y.iloc[s_test:]

    # --- backtest: fit on train, calibrate on calib, evaluate on test ---
    qmods = _fit_quantiles(Xtr, ytr)
    offset = _cqr_offset(qmods, Xcal, ycal)
    raw = {q: qmods[q].predict(Xte) for q in QUANTILES}
    raw = {q: v for q, v in zip(QUANTILES, np.sort(np.vstack([raw[q] for q in QUANTILES]).T, axis=1).T)}
    uncal_cov = round(float(np.mean((yte.to_numpy() >= raw[0.1]) & (yte.to_numpy() <= raw[0.9])) * 100), 1)
    cal = _apply_cqr(raw, offset)
    metrics = _metrics(yte.to_numpy(), cal)
    metrics["p10_p90_coverage_uncalibrated_pct"] = uncal_cov
    metrics["cqr_offset_eur"] = round(offset, 2)

    spike_clf = lgb.LGBMClassifier(**_CPARAMS).fit(Xtr, (ytr > SPIKE_EUR).astype(int))
    neg_clf = lgb.LGBMClassifier(**_CPARAMS).fit(Xtr, (ytr < 0).astype(int))
    metrics["spike_base_rate_pct"] = round(float((yte > SPIKE_EUR).mean() * 100), 2)
    metrics["neg_base_rate_pct"] = round(float((yte < 0).mean() * 100), 2)
    metrics["train_span"] = (str(X.index.min()), str(Xtr.index.max()))
    metrics["test_span"] = (str(Xte.index.min()), str(X.index.max()))

    # --- refit on ALL data for deployment; keep the calibration offset ---
    final_q = _fit_quantiles(X, y)
    final_spike = lgb.LGBMClassifier(**_CPARAMS).fit(X, (y > SPIKE_EUR).astype(int))
    final_neg = lgb.LGBMClassifier(**_CPARAMS).fit(X, (y < 0).astype(int))

    for q, m in final_q.items():
        m.booster_.save_model(str(ARTIFACTS / f"price_q{int(q*100)}.txt"))
    final_spike.booster_.save_model(str(ARTIFACTS / "price_spike.txt"))
    final_neg.booster_.save_model(str(ARTIFACTS / "price_neg.txt"))
    (ARTIFACTS / "features.json").write_text(json.dumps(features, indent=2))
    (ARTIFACTS / "calibration.json").write_text(json.dumps({"cqr_offset": offset}))
    (ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


# --- load & predict ------------------------------------------------------------

def load_models() -> dict:
    feats = json.loads((ARTIFACTS / "features.json").read_text())
    boosters = {q: lgb.Booster(model_file=str(ARTIFACTS / f"price_q{int(q*100)}.txt"))
                for q in QUANTILES}
    spike = lgb.Booster(model_file=str(ARTIFACTS / "price_spike.txt"))
    neg = lgb.Booster(model_file=str(ARTIFACTS / "price_neg.txt"))
    offset = json.loads((ARTIFACTS / "calibration.json").read_text())["cqr_offset"]
    return {"features": feats, "quantiles": boosters, "spike": spike, "neg": neg,
            "cqr_offset": offset}


def predict(X: pd.DataFrame, models: dict | None = None) -> pd.DataFrame:
    """Return per-row p10/p50/p90/spike_prob/neg_prob for a feature frame.
    Intervals are CQR-widened for calibrated ~80% coverage."""
    models = models or load_models()
    X = X[models["features"]]
    q = models["quantiles"]
    stack = np.sort(np.vstack([q[qq].predict(X) for qq in QUANTILES]).T, axis=1)
    off = models.get("cqr_offset", 0.0)
    out = pd.DataFrame(index=X.index)
    out["p10"] = stack[:, 0] - off
    out["p50"] = stack[:, 1]
    out["p90"] = stack[:, 2] + off
    out["spike_prob"] = models["spike"].predict(X)
    out["neg_prob"] = models["neg"].predict(X)
    return out

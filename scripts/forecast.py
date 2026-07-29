"""Freeze a live forecast into the `forecasts` table (for prospective verification).

Usage:  python scripts/forecast.py [YYYY-MM-DD]   (default: tomorrow)
Uses the deployed models to forecast the given delivery day and stores P10/P50/P90 +
spike/negative-price probabilities stamped with the issue time (run_ts). When the realized
DAM prices later arrive, scripts/verify.py scores this frozen forecast against them.
"""
import sys

import pandas as pd

from eml.config import settings
from eml.features.build import build_matrix
from eml.models import price_forecast as pf

LIVE_MODEL = "lgbm_cqr_live"

if __name__ == "__main__":
    matrix = build_matrix()
    models = pf.load_models()
    if len(sys.argv) > 1:
        day = pd.Timestamp(sys.argv[1]).normalize()
    else:
        day = pd.Timestamp.now(tz=settings.timezone).normalize().tz_localize(None) \
            + pd.Timedelta(days=1)

    X = matrix[matrix.index.normalize() == day]
    if X.empty:
        raise SystemExit(f"No feature rows for {day.date()} — need weather + recent prices.")
    preds = pf.predict(X[models["features"]], models)
    run_ts = pd.Timestamp.utcnow().tz_localize(None)
    n = pf._persist_forecasts(
        X.index, {q: preds[f"p{int(q*100)}"].to_numpy() for q in pf.QUANTILES},
        preds["spike_prob"].to_numpy(), preds["neg_prob"].to_numpy(),
        settings.default_zone, LIVE_MODEL, run_ts, replace=False)
    print(f"froze {n} forecast rows for {day.date()} (issued {run_ts:%Y-%m-%d %H:%M} UTC, "
          f"model={LIVE_MODEL})")

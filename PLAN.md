# Energy Market Lab — Build Plan

> **Authoritative blueprint. Read first each session.**
> A quantitative energy-market intelligence platform. Every incoming datum ultimately
> feeds a forecast and a natural-language answer to **"Why?"** and **"What happens next?"**

## Positioning & scope (the *why*)

- **Primary purpose:** portfolio / proof-of-skill for an **Energy Analyst** role. Optimize for an
  impressive, *explainable*, real-data demo — forecasting + narrative quality first, breadth later.
- **Anchor market:** Greece (ADMIE/IPTO grid, HEnEx day-ahead). Bidding zone `GR`.
- **Data policy for MVP:** free feeds only — **ENTSO-E Transparency Platform** (free token) and
  **Open-Meteo** (no key). Paid feeds (Montel, Bloomberg, ECMWF operational) are *later*, behind
  the same collector interface.

## Core principle: spine-first, not layer-first

The vision is 6 layers, ~100 sources, ~40 modules. That is a destination. We build **one thin
vertical slice through all six layers** and prove it end-to-end before widening any layer:

```
ENTSO-E + Open-Meteo → warehouse → feature engine → DAM price model → "why" brief → dashboard
   (real free data)     (SQLite→PG)    (pandas)      (LightGBM+quantiles) (SHAP→text)  (FastAPI+Plotly)
```

The narrative layer ("why will tomorrow's price rise?") is the hardest to fake and the fastest to
demo — it is the differentiator, so it ships in the first slice, not last.

## Deliberate MVP shortcuts (each upgrades cleanly — do not treat as accidental)

| MVP choice            | Upgrade path                                  | Why defer                                            |
|-----------------------|-----------------------------------------------|-----------------------------------------------------|
| SQLite                | PostgreSQL + TimescaleDB (conn-string swap)   | Timescale/Celery/Redis before a working forecast = wasted month |
| Sync pull scripts     | Celery + Redis + scheduler                    | Cron/manual is enough until pipelines are proven    |
| Server-rendered Plotly| React + TypeScript + AG Grid + Plotly.js      | Charts matter only once numbers are worth charting  |
| Templated "why" brief | LLM-generated brief over the same features    | SHAP→template is deterministic & free; LLM is a drop-in |

Schema stays **Postgres-compatible** throughout so the DB swap is a config change.

## Milestones

- **M0 — Foundation** ✅: repo, config, warehouse schema, collectors (ENTSO-E, Open-Meteo, fuel).
- **M1 — Warehouse filled** ✅: real GR day-ahead price, load, generation-by-fuel (2023→now,
  15-min/hourly resampled), ERA5 weather archive + forecast, TTF gas / EUA carbon / Brent.
- **M2 — Feature engine** ✅: calendar, weather (power-curves, HDD/CDD, ramps), price (AR, vol,
  momentum, skew), grid (residual load, penetration, shares), **fuel/carbon (gas, EUA, gas SRMC)** —
  ~49 features. See `docs/FEATURES.md`.
- **M3 — DAM price forecast** ✅: LightGBM P10/P50/P90 + spike/negative classifiers, **CQR interval
  calibration**, **additive bias correction**, **trailing-window deployment**.
- **M4 — "Why" layer** ✅: SHAP driver attribution, wind-sensitivity what-if, templated morning
  brief, 6-day outlook (real + weather-derived-indicative).
- **M5 — Dashboard** ✅: Home page — brief, forecast band, drivers, risk, outlook, **live
  verification panel**.
- **M5.5 — Verification & ops** ✅: freeze→verify loop, **walk-forward retraining** (regime-tracking
  out-of-sample track record), daily/weekly Task Scheduler automation, pinned dashboard port.

### Current status (real data, live)

Fully live on real ENTSO-E/Open-Meteo/market data. Walk-forward verification (722 days):
MAE ≈ €20/MWh, P10–P90 coverage ≈ 78%, bias ≈ −€1.6, **skill vs persistence ≈ +18%**. Synthetic
bridge retired. Deployed model retrains daily on a trailing window; dashboard at
`http://127.0.0.1:8010/`.

### M6 — Active roadmap (in progress)

1. **Cross-border features:** neighbor day-ahead prices (Italy, Bulgaria) + net import position and
   interconnector flows (ENTSO-E). Greece is a net importer — the biggest remaining skill lever.
2. **RES-forecast residual load:** backfill ENTSO-E historical wind/solar *day-ahead forecast* and
   build residual load from it (removes the current actual-generation leakage; matches live serving).
3. **Normalized / Mondrian conformal intervals:** width proportional to predicted local uncertainty
   → tighter bands in calm hours, wider only where spikes are plausible.

### M7+ — Later

More bidding zones, risk module (VaR / Expected Shortfall / Monte Carlo), optimization
(battery/VPP/storage dispatch), LLM-written briefs, React + TypeScript frontend,
PostgreSQL/TimescaleDB + Celery/Redis scheduling.

## Layout

```
energy-market-lab/
  PLAN.md                 # this file — authoritative
  eml/
    config.py             # env-driven settings
    db.py                 # engine + warehouse schema + upsert
    collectors/           # Layer 1–2: acquisition (one module per source)
    features/             # Layer 4: feature engine
    models/               # Layer 5: forecasting services
    narrative/            # Layer 6 (AI): SHAP → "why" briefs
    api/                  # FastAPI app + dashboard
  scripts/                # init_db, pull_*, train, forecast (become Celery tasks later)
  docs/                   # FEATURES.md, data-source notes
```

## Working agreement

- Ask before any non-obvious decision.
- Every collector implements the same interface (`fetch() -> DataFrame`) so paid feeds slot in later.
- Every table is timestamped and Postgres-compatible.

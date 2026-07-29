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

- **M0 — Foundation** (in progress): repo, config, warehouse schema, ENTSO-E + Open-Meteo collectors.
  Open-Meteo runs today (no key). ENTSO-E runs once a free token is added.
- **M1 — Warehouse filled:** pull ≥2 years of GR day-ahead price, load, generation-by-fuel + weather.
- **M2 — Feature engine:** price (rolling stats, momentum, calendar), weather (HDD/CDD, wind ramp,
  solar efficiency), grid (residual load, renewable penetration, reserve margin), fuel (spark/dark
  spreads) — see `docs/FEATURES.md`.
- **M3 — DAM price forecast:** LightGBM quantile models → hourly point + P10/P50/P90, spike prob,
  negative-price prob. Walk-forward backtest with MAE/pinball loss.
- **M4 — "Why" layer:** SHAP attribution → top drivers → templated morning brief. "Why did price
  move?" / "How sensitive to lower wind?".
- **M5 — Dashboard:** one Home page — today's forecast, drivers, alerts — then widen to workspaces.
- **M6+ — Widen:** more zones, fuel/carbon feeds, load/wind/solar sub-forecasts, risk (VaR/ES/MC),
  optimization (battery/VPP), React frontend, Postgres/Timescale, Celery scheduling.

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

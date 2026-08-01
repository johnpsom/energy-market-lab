# Energy Market Lab

A quantitative **energy-market decision-support platform** for the Greek electricity day-ahead
market. It ingests real market, weather, and fuel data, engineers them into predictive features,
forecasts tomorrow's hourly power price *with calibrated uncertainty*, explains **why** in plain
language, and continuously **verifies its own track record** against realized prices. The goal is
not another data dashboard; it answers the two questions a trader or analyst actually asks:
**"What happens next?"** and **"Why?"** — and then proves how well it answered them.

Built **spine-first** (one thin vertical slice through all six layers), now **fully live on real
data** with walk-forward retraining. See [`PLAN.md`](PLAN.md) for the roadmap and
[`docs/FEATURES.md`](docs/FEATURES.md) for the feature catalog.

---

## Table of contents

1. [What it does today](#1-what-it-does-today)
2. [The problem](#2-the-problem)
3. [Energy-market primer](#3-energy-market-primer)
4. [Architecture — the six layers](#4-architecture--the-six-layers)
5. [What drives the price](#5-what-drives-the-price)
6. [The forecasting model](#6-the-forecasting-model)
7. [Verification — the honest track record](#7-verification--the-honest-track-record)
8. [The narrative / "why" layer](#8-the-narrative--why-layer)
9. [Data sources](#9-data-sources)
10. [Operations — daily automation](#10-operations--daily-automation)
11. [Technology stack & dependencies](#11-technology-stack--dependencies)
12. [Project layout](#12-project-layout)
13. [Quickstart](#13-quickstart)
14. [Roadmap](#14-roadmap)
15. [Glossary](#15-glossary)

---

## 1. What it does today

| Layer | Component | Status |
|---|---|---|
| 1–2 Acquisition | ENTSO-E (prices/load/generation + day-ahead load & RES forecasts), Open-Meteo (weather + ERA5 archive), yfinance (TTF gas, EUA carbon, Brent) | ✅ live, real data |
| 3 Warehouse | SQLite (Postgres/TimescaleDB-compatible schema), idempotent upsert | ✅ |
| 4 Features | 49 features: calendar, weather (power-curves, HDD/CDD), price (AR, volatility), grid (residual load), **fuel/carbon (gas, EUA, SRMC)** | ✅ |
| 5 Forecast | LightGBM P10/P50/P90, CQR-calibrated intervals, spike & negative-price classifiers, **bias correction**, **trailing-window deployment** | ✅ |
| 5 Verification | **Walk-forward retraining** — the honest, regime-tracking out-of-sample track record | ✅ |
| 6 Narrative + UI | SHAP drivers, wind-sensitivity what-if, morning brief, 6-day outlook, FastAPI + Plotly dashboard | ✅ |
| Ops | Daily data refresh + model retrain, weekly walk-forward (Task Scheduler) | ✅ |

**Live walk-forward verification (722 days, Greek DAM):** MAE ≈ €20/MWh · P10–P90 coverage ≈ 78% ·
bias ≈ −€1.6 · **skill vs naïve persistence ≈ +18%**.

> This is real, out-of-sample performance on the Greek market — not a demo on fabricated data.
> A synthetic-fundamentals bridge (now retired) was used only to build the pipeline before the
> ENTSO-E token arrived; all models now run on real ENTSO-E/Open-Meteo/market data.

---

## 2. The problem

Electricity can't be stored cheaply at scale, so supply and demand must match every instant — which
makes its wholesale price one of the most volatile of any commodity. It sits calm at €40/MWh one
hour, spikes past €900 the next when demand peaks and wind drops, or crashes **below zero** when
solar floods a low-demand afternoon (generators pay to keep producing rather than shut down).

The **forecast target** is the **Day-Ahead Market (DAM) price**: the price set once per day for each
of the next day's 24 hours. Outputs: median (P50), an 80% prediction interval (P10–P90), the
probability of a **spike** (>€150/MWh) and of a **negative price**, the top **drivers**, and a
**what-if** sensitivity. Anchor market: Greek bidding zone `GR` (ADMIE/IPTO grid, HEnEx exchange).

---

## 3. Energy-market primer

**Market timeframes** (each a separate market): **Day-Ahead (DAM)** — hourly auction clearing
~noon the day before delivery, *our target*; **Intraday (IDM)** — continuous adjustments after DAM;
**Balancing** — real-time, the TSO keeping the grid at 50 Hz.

**Gate closure** — the DAM bid deadline (~12:00 D-1). Everything a feature uses to forecast day D
must be knowable before then; using anything later is **data leakage** (§7).

**Merit order & the marginal plant** — to meet demand the grid dispatches generators cheapest-first:
near-zero-cost renewables and must-run hydro/nuclear, then gas and coal, finally expensive peakers.
The price for *everyone* is set by the **last (most expensive) plant needed**. So more wind/solar →
price falls (sometimes below zero); high demand + low renewables → an expensive plant sets the price
→ spike. This is why **residual load** (demand net of wind & solar) is the model's strongest driver.

**Bidding zone / TSO / exchange** — Greece is one price zone (`GR`), grid run by **ADMIE/IPTO**,
market by **HEnEx**; **ENTSO-E** publishes the pan-European data. Greece is a net **importer**, so
prices in **Italy** and **Bulgaria** and the interconnector flows matter (§5, roadmap item 1).

---

## 4. Architecture — the six layers

```
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ 1. Sources   │ → │ 2. Acquisition│ → │ 3. Warehouse │
│ market/wx/   │   │ collectors   │   │ timestamped  │
│ fuel/carbon  │   │ (API)        │   │ tables       │
└──────────────┘   └──────────────┘   └──────┬───────┘
                                              ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ 6. Dashboard │ ← │ 5. Forecast  │ ← │ 4. Features  │
│ + narrative  │   │ + verify     │   │ engine       │
└──────────────┘   └──────────────┘   └──────────────┘
```

| Layer | Package | Responsibility |
|---|---|---|
| 1–2 Acquisition | `eml/collectors/` | One module per source, all exposing `fetch()`. `entsoe_collector`, `weather_collector`, `fuel_collector`. |
| 3 Warehouse | `eml/db.py` | Timestamped tables (`prices`, `load`, `generation`, `weather`, `fuel`, `forecasts`). SQLite now; Postgres/Timescale-compatible schema. Idempotent upsert. |
| 4 Features | `eml/features/` | ~49 model-ready signals (§5). |
| 5 Forecast | `eml/models/price_forecast.py` | LightGBM quantiles + classifiers, bias/CQR calibration, trailing-window deploy. |
| 5 Verification | `eml/models/verify.py` | Walk-forward scorecard vs realized prices (§7). |
| 6 Narrative + UI | `eml/narrative/`, `eml/api/` | SHAP briefs, outlook; FastAPI + Plotly dashboard. |

---

## 5. What drives the price

Fundamental contributors, each turned into features (catalog: [`docs/FEATURES.md`](docs/FEATURES.md)):

| Contributor | Effect | Features |
|---|---|---|
| **Demand (load)** | ↑ demand → ↑ price (temperature, calendar) | load, HDD/CDD, calendar |
| **Wind / solar** | ↑ renewables → ↓ price (solar can crash midday negative) | power-curve proxies, shares |
| **Residual load** | `load − wind − solar`; the merit-order dial — top driver | `residual_load`, `renewable_penetration` |
| **Fuel (TTF gas)** | sets thermal running cost → the price floor | `gas_price`, `gas_srmc` |
| **Carbon (EUA)** | CO₂ cost on fossils; shifts coal↔gas margin | `carbon_price`, `gas_srmc` |
| **Cross-border (IT/BG)** | imports cap price; congestion isolates & spikes | *roadmap 1* |
| **Autoregression** | yesterday/last-week price persistence | price lags, rolling vol |

**Real-market finding:** in 2025–26 Greek prices often clear **below gas marginal cost** —
renewables, lignite, and imports set the margin, not gas. (Modeling price *relative* to gas SRMC
was tried and reverted for exactly this reason; gas enters as a plain feature instead.)

---

## 6. The forecasting model

**Algorithm — LightGBM gradient-boosted trees.** Captures non-linear interactions (high demand *and*
low wind *and* evening), handles missing values, trains in seconds, and emits per-feature SHAP
contributions for free.

**Probabilistic, not point.** Three **quantile** models give **P10 / P50 / P90** → a median plus an
80% prediction interval. Two **classifiers** give calibrated P(spike > €150) and P(negative price).

**Calibration (two stages, both from a held-out recent slice — no leakage):**
- **Bias correction** — an additive shift that removes systematic over/under-forecast. Critical:
  trees can't extrapolate price *levels* beyond their training range, so a falling-price regime
  otherwise produces a standing bias.
- **Conformalized Quantile Regression (CQR)** — widens the interval by exactly enough to hit its
  target coverage, with a finite-sample guarantee.

**Regime tracking.** The deployed model is refit on a **trailing window** (drops stale 2023
crisis-era data) and re-calibrated on the most recent slice, so live forecasts track the current
regime. The daily job retrains it (§10).

---

## 7. Verification — the honest track record

The differentiator: the platform **grades its own forecasts** against realized DAM prices.

**Walk-forward** (`walk_forward`) is the core method: repeatedly retrain on a trailing window and
forecast the next block, so *every* prediction comes from a model that saw only prior data — the
correct, regime-tracking out-of-sample test (never shuffle a time series). Each block is frozen to
the `forecasts` table and scored.

**Freeze → verify loop** for live operation: `scripts/forecast.py` freezes each day's forecast
before the outcome exists; when the realized DAM lands, `scripts/verify.py` joins and scores it.
An **honesty cutoff** ensures only genuinely-realized days (before today) enter the track record.

**Metrics:** MAE / RMSE / bias, P10–P90 **coverage** (target 80%), **pinball** loss per quantile,
**skill vs naïve persistence** ("tomorrow = same hour yesterday"), and spike-probability **Brier**
score + reliability. Current: **722 days, MAE ≈ €20, coverage ≈ 78%, +18% skill, bias ≈ −€1.6.**

---

## 8. The narrative / "why" layer

- **SHAP attribution** — decomposes each forecast into per-feature €/MWh contributions (LightGBM
  native), so the top drivers are exact, not guessed.
- **Sensitivity / what-if** — perturb an input (e.g. wind −30%) and re-price to quantify exposure.
- **Morning brief** — plain-language summary: expected average, evening peak (vs daily max), the
  band, spike/negative-price risk, ranked drivers, wind what-if. Deterministic today; an LLM can
  later rewrite the same structured facts.
- **6-day outlook** — day-ahead days use real ENTSO-E forecasts; beyond that horizon, renewables are
  derived from the Open-Meteo power-curves and load from a recent climatology, tagged *indicative*.

---

## 9. Data sources

| Source | Cost | Provides | Module |
|---|---|---|---|
| **ENTSO-E Transparency** | free (token) | DAM prices, load (actual + forecast), generation by fuel, day-ahead wind/solar forecast; *roadmap:* cross-border | `entsoe_collector` |
| **Open-Meteo** | free, no key | weather forecast + ERA5 archive (temp, wind@100m, irradiance, cloud) | `weather_collector` |
| **Yahoo Finance** | free | TTF gas (`TTF=F`), EUA carbon (`CO2.L`), Brent (`BZ=F`) daily settlements | `fuel_collector` |

**ENTSO-E token:** register at transparency.entsoe.eu, email `transparency@entsoe.eu` (subject
"Restful API access"), then Account Settings → generate token → put in `.env` as `ENTSOE_TOKEN`
(gitignored).

---

## 10. Operations — daily automation

- **`scripts/update_live.py`** — daily refresh: catch up realized actuals, pull forward day-ahead
  load & RES forecasts, refresh weather, extend the outlook horizon.
- **`scripts/daily_update.cmd`** — Task Scheduler job (`EnergyMarketLab-DailyUpdate`, 14:00 daily,
  after the DAM publishes): runs the refresh **and retrains the deployed model** on the latest window.
- **`scripts/weekly_verify.cmd`** — Task Scheduler job (`EnergyMarketLab-WeeklyVerify`, Sun 15:00):
  re-runs walk-forward verification so the track record stays current.
- **`scripts/serve.cmd`** — serves the dashboard on a fixed port (**http://127.0.0.1:8010/**).

---

## 11. Technology stack & dependencies

| Package | Role |
|---|---|
| pandas / numpy | timeseries + numeric core |
| SQLAlchemy | warehouse access (SQLite→Postgres = config swap) |
| requests | HTTP for Open-Meteo |
| entsoe-py | ENTSO-E Transparency client |
| yfinance | fuel/carbon settlements |
| holidays | Greek calendar incl. moveable Orthodox Easter |
| lightgbm | gradient boosting — quantiles, classifiers, native SHAP |
| scikit-learn | LightGBM estimator API + metrics |
| fastapi + uvicorn | dashboard server |
| jinja2 | HTML templating |
| plotly | interactive charts |
| pyarrow | fast Parquet I/O |

**Planned:** psycopg (Postgres), celery + redis (scheduling/queue), pyomo/pypsa (optimization),
react + typescript (production frontend).

---

## 12. Project layout

```
eml/
  config.py        env-driven settings (token, DB, zone, timezone)
  db.py            Layer 3  schema + idempotent upsert
  collectors/      Layer 1-2  entsoe_collector, weather_collector, fuel_collector
  features/        Layer 4  calendar, weather, price, grid, fuel, build (orchestrator)
  models/          Layer 5  price_forecast (quantiles/CQR/bias/walk-forward), verify
  narrative/       Layer 6  brief (SHAP, sensitivity, morning brief, outlook)
  api/             Layer 6  FastAPI + Plotly dashboard
  outlook_extend.py         weather-derived renewables + load climatology for the outlook horizon
scripts/           backfill_entsoe, pull_fuel, update_live, train, walk_forward, verify, forecast,
                   serve, daily_update.cmd, weekly_verify.cmd
docs/FEATURES.md   the feature catalog with reasoning + leakage tags
PLAN.md            authoritative build plan & roadmap
```

## 13. Quickstart

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt      # Windows; use .venv/bin on POSIX
cp .env.example .env                                # add ENTSOE_TOKEN
export PYTHONPATH=.

python scripts/backfill_entsoe.py 2023-01-01 2026-08-01   # real GR prices/load/generation
python scripts/pull_history.py    2023-01-01 2026-08-01   # ERA5 weather archive
python scripts/pull_fuel.py                               # TTF gas, EUA carbon, Brent
python scripts/update_live.py                             # forward forecasts + outlook extension
python scripts/train.py                                   # deploy model + walk-forward verification
python scripts/serve.cmd                                  # dashboard -> http://127.0.0.1:8010
```

## 14. Roadmap

Active next steps (see PLAN.md for the full list):
1. **Cross-border features** — neighbor day-ahead prices (Italy, Bulgaria) + net import position and
   interconnector flows from ENTSO-E. Greece is a net importer, so these are a large skill lever.
2. **Train residual load on RES *forecasts*, not actuals** — backfill ENTSO-E's historical
   wind/solar day-ahead forecast and build residual load from it, removing the current mild leakage
   and matching live conditions.
3. **Normalized / Mondrian conformal intervals** — scale interval width by a predicted local
   uncertainty, so the band is tight in calm hours and only fans out where a spike is plausible
   (sharper 80%/90% bands at the same coverage).

Then: risk module (VaR / Expected Shortfall / Monte Carlo), optimization (battery/VPP dispatch),
LLM-written briefs, React frontend, PostgreSQL/TimescaleDB + Celery.

## 15. Glossary

- **ADMIE / IPTO** — Greek transmission system operator. **HEnEx** — Hellenic Energy Exchange.
- **DAM / IDM** — Day-Ahead / Intraday Market. **Gate closure** — DAM bid deadline (leakage boundary).
- **Merit order / marginal plant** — cheapest-first dispatch; the last plant needed sets the price.
- **Residual load** — demand − wind − solar; the key price driver.
- **TTF** — benchmark European gas hub. **EUA** — EU carbon allowance. **SRMC** — short-run marginal cost.
- **Quantile regression / P10–P90** — predicts percentiles; the 80% prediction interval.
- **CQR** — Conformalized Quantile Regression (calibrates interval coverage).
- **Walk-forward** — retrain-then-test rolling forward; the honest time-series validation.
- **SHAP** — per-feature contribution to a prediction. **Pinball loss** — quantile scoring rule.
- **Skill vs persistence** — % improvement over "tomorrow = same hour yesterday".
- **MAE / RMSE / Brier** — accuracy metrics (point / squared / probability).

---

*Educational / portfolio project. Not trading or investment advice.*

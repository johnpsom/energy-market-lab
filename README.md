# Energy Market Lab

A quantitative **energy-market decision-support platform** for the Greek electricity day-ahead
market. It ingests weather and market data, engineers them into predictive features, forecasts
tomorrow's hourly power price *with calibrated uncertainty*, and then explains **why** — in plain
language. The goal is not another data dashboard; it is a system that answers the two questions a
trader or analyst actually asks: **"What happens next?"** and **"Why?"**

It is built **spine-first**: one thin vertical slice through all six architectural layers, running
end-to-end on **free data only**, before any layer is widened. See [`PLAN.md`](PLAN.md) for the
roadmap and [`docs/FEATURES.md`](docs/FEATURES.md) for the full feature catalog.

---

## Table of contents

1. [The problem being solved](#1-the-problem-being-solved)
2. [Energy-market primer (read this first)](#2-energy-market-primer)
3. [Architecture — the six layers](#3-architecture--the-six-layers)
4. [What drives an electricity price (the "contributors")](#4-what-drives-an-electricity-price)
5. [The feature engine](#5-the-feature-engine)
6. [The forecasting model](#6-the-forecasting-model)
7. [How the model is evaluated](#7-how-the-model-is-evaluated)
8. [The narrative / "why" layer](#8-the-narrative--why-layer)
9. [The synthetic-data bridge](#9-the-synthetic-data-bridge)
10. [Data sources & how to get them](#10-data-sources--how-to-get-them)
11. [Technology stack & every dependency](#11-technology-stack--every-dependency)
12. [Project layout](#12-project-layout)
13. [Quickstart](#13-quickstart)
14. [Roadmap](#14-roadmap)
15. [Glossary](#15-glossary)
16. [Disclaimer](#16-disclaimer)

---

## 1. The problem being solved

Electricity is an unusual commodity: it **cannot be stored cheaply at scale**, so supply and
demand must match *every instant*. That makes its wholesale price one of the most volatile of any
traded commodity — calm at €40/MWh one hour, spiking to €200+/MWh the next when demand peaks and
the wind drops, or crashing **below zero** when solar floods a low-demand afternoon (negative
prices mean generators pay to keep producing rather than shut down).

The **forecasting target** here is the **Day-Ahead Market (DAM) price**: the price, set once per
day, for each of the next day's 24 hours. Market participants submit bids/offers, an auction
clears, and the resulting hourly prices are what generators are paid and suppliers pay. Forecasting
those 24 numbers — with honest uncertainty and an explanation — is the core deliverable.

- **Target:** hourly DAM price (€/MWh), Greek bidding zone `GR`.
- **Horizon:** tomorrow (24 h), extended to a 6–7 day outlook.
- **Outputs:** median price (P50), an 80% prediction interval (P10–P90), the probability of a
  **spike** (>€150/MWh), the probability of a **negative price**, the top **drivers**, and a
  **what-if** sensitivity (e.g. "if wind falls 30%…").

---

## 2. Energy-market primer

If you are new to power markets, these are the concepts the whole system is built on.

**Wholesale electricity markets** are split into sequential "timeframes", each a separate market:

| Market | When it clears | What it is |
|---|---|---|
| **Day-Ahead Market (DAM)** | ~noon the day before delivery | Hourly auction for the next day. **Our forecast target.** |
| **Intraday Market (IDM)** | Continuously, up to near real-time | Adjustments after DAM as forecasts update. |
| **Balancing / real-time** | Minute-to-minute | The TSO buys/sells to keep the grid at exactly 50 Hz. |

**Gate closure** — the deadline to submit DAM bids (~12:00 the day before). Everything the model
uses to forecast a delivery day must be knowable *before* gate closure; using anything later is
**data leakage** (see §7).

**Bidding zone** — a geographic area with a single price. Greece is one zone (`GR`), operated by
**ADMIE/IPTO** (the transmission system operator) with the market run by **HEnEx** (the Hellenic
Energy Exchange). **ENTSO-E** is the pan-European body whose Transparency Platform publishes the
data for every zone.

**Merit order & the marginal plant** — the single most important idea for price. To meet demand,
the grid dispatches generators from cheapest to most expensive: first near-zero-cost **renewables**
(wind, solar) and must-run **nuclear/hydro**, then **gas** and **coal**, finally expensive peaking
plant. The price for *everyone* is set by the **last (most expensive) plant needed** — the
**marginal plant**. So:

- More wind/solar → the expensive plants aren't needed → **price falls** (sometimes below zero).
- High demand + low renewables → an expensive gas/oil peaker sets the price → **price spikes**.

This is why the model's strongest single feature is **residual load** (§4) — it is essentially a
proxy for *how far up the merit order the market has to climb*.

---

## 3. Architecture — the six layers

```
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ 1. Sources   │ → │ 2. Acquisition│ → │ 3. Warehouse │
│ weather,     │   │ collectors   │   │ timestamped  │
│ market, fuel │   │ (API/scrape) │   │ tables       │
└──────────────┘   └──────────────┘   └──────┬───────┘
                                              ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ 6. Dashboard │ ← │ 5. Forecast  │ ← │ 4. Features  │
│ + narrative  │   │ engine (ML)  │   │ engine       │
└──────────────┘   └──────────────┘   └──────────────┘
```

| Layer | Package | Responsibility |
|---|---|---|
| **1–2 Acquisition** | `eml/collectors/` | One module per source, all exposing `fetch()`. Free feeds today (Open-Meteo, ENTSO-E); paid feeds slot in behind the same interface. |
| **3 Warehouse** | `eml/db.py` | Timestamped tables (`prices`, `load`, `generation`, `weather`, `forecasts`). SQLite now; schema is **PostgreSQL/TimescaleDB-compatible** so going to a production time-series DB is a connection-string change. Idempotent upsert (re-pulling a window updates, never duplicates). |
| **4 Features** | `eml/features/` | Turns raw rows into ~44 model-ready signals (§5). |
| **5 Forecast** | `eml/models/` | LightGBM quantile models + calibrated intervals + spike/negative classifiers (§6). |
| **6 Narrative + UI** | `eml/narrative/`, `eml/api/` | SHAP attribution → morning brief; FastAPI + Plotly dashboard. |

---

## 4. What drives an electricity price

These are the **fundamental contributors** — the real-world variables that move price. Each is
turned into one or more features (the mapping to feature names is in [`docs/FEATURES.md`](docs/FEATURES.md)).

| Contributor | Why it moves price | Direction |
|---|---|---|
| **Demand (load)** | More consumption → dearer plant needed → higher price. Driven by temperature (heating/cooling), time of day, weekday/holiday, economic activity. | ↑ demand → ↑ price |
| **Wind generation** | Near-zero marginal cost; displaces expensive plant. | ↑ wind → ↓ price |
| **Solar generation** | Same, but only daytime; can crash midday prices, even negative. | ↑ solar → ↓ midday price |
| **Residual load** | `demand − wind − solar` = what thermal plant must cover. **The best single price predictor.** | ↑ residual → ↑ price |
| **Fuel prices (gas/coal)** | Set the *running cost* of the marginal thermal plant → the price floor in tight hours. Gas ≈ **TTF**; benchmark oil ≈ **Brent/WTI**. | ↑ fuel → ↑ price |
| **Carbon (EU ETS / EUA)** | CO₂ permits add cost to fossil generation; shifts which fuel is marginal (coal vs gas). | ↑ carbon → ↑ price |
| **Cross-border flows** | Imports from cheaper neighbours cap price; congestion isolates a zone and can spike it. | context |
| **Outages** | A large plant or interconnector offline removes cheap supply. | ↑ price |
| **Weather forecasts** | The upstream driver of *both* demand (temperature) and renewable supply (wind speed, irradiance, cloud). | mixed |

The **spark spread** and **dark spread** are derived contributors traders watch:
- **Spark spread** = power price − (gas price ÷ plant efficiency). The gross margin of a **gas** plant.
- **Dark spread** = power price − (coal price ÷ efficiency). The gross margin of a **coal** plant.
- **"Clean" spreads** subtract the carbon cost — they tell you *which fuel is marginal today*.

---

## 5. The feature engine

Raw data is not fed to the model directly; it is transformed into features that expose the
structure above. Four families (full catalog: [`docs/FEATURES.md`](docs/FEATURES.md)):

- **Calendar** — hour, day-of-week, month encoded as **sine/cosine pairs** (so 23:00 sits next to
  00:00), peak-hour flag, **Greek public holidays** including the *moveable* Orthodox Easter,
  and "bridge day" flags. Demand is strongly cyclical, so these are high-value and leak-free.
- **Weather** — **HDD/CDD** (Heating/Cooling Degree-hours: how far temperature sits below 18 °C or
  above 22 °C — a near-linear proxy for heating/cooling demand); **turbine and solar-panel power
  curves** (raw wind speed and irradiance pushed through the *physical* response curve, so the
  model isn't forced to relearn non-linear physics from scarce data); **ramp** features (hour-over-
  hour change — price spikes live in the *changes*, not the levels).
- **Price (autoregressive)** — lagged prices (24 h / 48 h / 168 h ago), rolling mean/volatility,
  momentum, and **skew** (flags spike-prone regimes). All lagged ≥24 h to respect gate closure.
- **Grid** — **residual load**, **renewable penetration**, wind/solar share, load ramps.

> **Leakage discipline:** a feature for a delivery hour may only use information available *before
> gate closure*. Realized prices are lagged ≥24 h; weather uses *forecasts*, not outturn. This is
> what makes the backtest honest — see §7.

---

## 6. The forecasting model

**Algorithm — gradient-boosted decision trees (LightGBM).** Gradient boosting builds an ensemble
of small decision trees, each correcting the errors of the last. It is the workhorse of tabular
forecasting because it captures non-linear interactions (e.g. "high demand *and* low wind *and*
evening"), handles missing values natively, trains in seconds, and — crucially here — can output
**per-feature attributions** (SHAP) for free.

**Probabilistic, not point.** A single number ("€82") is useless for risk. The model instead
predicts three **quantiles**:

- **Quantile regression** trains a model to predict, say, the 10th percentile of the price
  distribution — the level the price should exceed 90% of the time. We train **P10, P50, P90**.
- **P50** is the median (the point forecast); **P10–P90** is an **80% prediction interval** — the
  band tomorrow's price should land in 80% of the time.

**Calibration with Conformalized Quantile Regression (CQR).** Raw quantile models are often
*overconfident* — here the raw 80% interval only covered ~54% of outcomes. CQR fixes this with a
finite-sample guarantee: on a held-out calibration slice it measures how far reality falls outside
the predicted band, then widens the band by exactly that amount. Result: **coverage moved from
53.7% → 80.2%** (target 80%). This matters — an uncalibrated interval *lies about risk*.

**Event probabilities.** Two extra LightGBM **classifiers** output calibrated probabilities for the
events a trader cares about: **P(spike > €150/MWh)** and **P(negative price)**.

---

## 7. How the model is evaluated

- **Time-ordered split (never shuffle a time series).** Train on the past, calibrate on a middle
  slice, test on the most recent, unseen period. Shuffling would let the model "see the future".
- **Walk-forward** is the natural extension (repeatedly retrain-then-test rolling forward); the
  current backtest is a single train/calibrate/test holdout, upgradeable to full walk-forward.
- **Metrics:**
  - **MAE** (Mean Absolute Error, €/MWh) — average miss of the median.
  - **RMSE** (Root Mean Squared Error) — penalizes big misses more; sensitive to spikes.
  - **Pinball (quantile) loss** — the correct scoring rule for a quantile forecast; low pinball
    means the P10/P50/P90 are individually well-placed.
  - **P10–P90 coverage** — the % of outcomes that actually land in the interval; should ≈ 80%.
- **Current backtest (out-of-sample):** MAE **€7.38/MWh**, RMSE €10.1, coverage **81.9%**.
  *(On synthetic prices whose structure is learnable — see §9; real-data numbers will differ.)*

---

## 8. The narrative / "why" layer

A forecast nobody understands won't be trusted. This layer turns numbers into explanation:

- **SHAP attribution** — SHAP (SHapley Additive exPlanations) decomposes each forecast into a sum
  of per-feature contributions (in €/MWh), grounded in cooperative game theory. LightGBM computes
  these natively (`pred_contrib=True`), so the top drivers ("residual load is adding €5/MWh",
  "solar is subtracting €7/MWh") are exact, not guessed.
- **Sensitivity / what-if** — perturb an input (e.g. cut wind 30%) and re-price to quantify the
  impact. Answers "how exposed is tomorrow to a wind miss?".
- **Morning brief** — a templated, plain-language summary: expected average, evening peak, the
  band, spike/negative-price risk, the ranked drivers, and the wind what-if. It is deterministic
  and free today; an **LLM** can later rewrite the same structured facts into richer prose — a
  drop-in, because the inputs don't change.

---

## 9. The synthetic-data bridge

The real Greek price feed (ENTSO-E) requires a free token that takes a day or two to be granted
(§10). So the model does not have to wait, prices are currently produced by a **transparent
synthetic-fundamentals model** (`eml/synthetic/`) that is driven by **real** ERA5/Open-Meteo
weather: it derives load from temperature and calendar, wind/solar from the physical power curves,
**residual load** from their difference, and price from a **merit-order curve** plus a fuel level,
with realistic **spikes** and **negative-price** regimes.

This is a **scaffold, not a forecast of record.** Its purpose is to exercise the entire pipeline
end-to-end and prove the relationships are recoverable (a well-trained model + SHAP *do* recover
residual load, wind, and solar as the top drivers — exactly what was built in). Every synthetic row
is tagged `source='synthetic'`; connecting ENTSO-E simply replaces them with `source='entsoe'`,
**with zero changes to the feature engine, model, or dashboard.**

---

## 10. Data sources & how to get them

| Source | Cost | Provides | Access |
|---|---|---|---|
| **Open-Meteo** | Free, **no key** | Weather forecast + **ERA5 archive** (years of history): temperature, wind@100 m, irradiance, cloud, precipitation | Just call it. Used by `weather_collector`. |
| **ENTSO-E Transparency** | **Free** (token) | DAM prices, load (actual+forecast), generation by fuel, cross-border flows — all EU zones incl. Greece | See below. Used by `entsoe_collector`. |
| Montel / EEX / Refinitiv | Paid | Curated fuel/carbon/news feeds | Future; same collector interface. |
| ECMWF operational | Paid | Higher-res weather ensembles | Future. |

**Getting the free ENTSO-E token:**
1. Register at **https://transparency.entsoe.eu/** and confirm your email.
2. Email **`transparency@entsoe.eu`**, subject **"Restful API access"**, with your account email in
   the body. They enable API access (usually 1–2 days — the only manual step).
3. **Account Settings → "Web Api Security Token" → Generate.** Paste into `.env` as `ENTSOE_TOKEN`.

No credit card, no paid tier.

---

## 11. Technology stack & every dependency

**Runtime dependencies** (`requirements.txt`) and *why each is here*:

| Package | Role |
|---|---|
| **pandas / numpy** | Time-series wrangling and the numeric core of every layer. |
| **SQLAlchemy** | Warehouse access; the dialect-agnostic layer that lets SQLite→Postgres be a config swap. |
| **requests** | HTTP calls to Open-Meteo / ENTSO-E. |
| **python-dotenv** | Loads secrets (`ENTSOE_TOKEN`) and config from `.env`. |
| **entsoe-py** | Typed client for the ENTSO-E Transparency API (prices/load/generation). |
| **holidays** | Greek public-holiday calendar, incl. *moveable* Orthodox Easter (a real demand driver). |
| **lightgbm** | The gradient-boosting engine — quantile models, classifiers, and native SHAP. |
| **scikit-learn** | Required by LightGBM's estimator API; utility metrics/splitting. |
| **fastapi + uvicorn** | The dashboard web server (async, typed, auto-docs at `/docs`). |
| **jinja2** | HTML templating for the dashboard page. |
| **plotly** | Interactive forecast charts (band + median, multi-day outlook). |
| **pyarrow** | Fast Parquet I/O for cached feature matrices. |

**Planned:** `psycopg` (Postgres driver), `celery` + `redis` (scheduling/queue), `pyomo`/`pypsa`
(optimization & power-system modeling), `react`+`typescript` (production frontend).

**Conceptual prerequisites** to build a model like this: a grasp of **power-market mechanics**
(§2), **time-series ML** (leakage, walk-forward validation, stationarity), **probabilistic
forecasting** (quantiles, calibration, proper scoring rules), and **basic meteorology→energy**
translation (degree-days, power curves).

---

## 12. Project layout

```
eml/
  config.py       env-driven settings (token, DB URL, zone, timezone)
  db.py           Layer 3  warehouse schema + idempotent upsert
  collectors/     Layer 1-2  weather_collector (Open-Meteo), entsoe_collector
  features/       Layer 4  calendar, weather, price, grid, build (orchestrator)
  synthetic/      bridge   real weather → synthetic load/generation/price
  models/         Layer 5  price_forecast (quantiles + CQR + classifiers)
  narrative/      Layer 6  brief (SHAP drivers, sensitivity, morning brief, outlook)
  api/            Layer 6  FastAPI + Plotly dashboard
scripts/          thin CLI entry points (become Celery tasks later)
docs/FEATURES.md  the full feature catalog with reasoning + leakage tags
PLAN.md           authoritative build plan & roadmap
```

---

## 13. Quickstart

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt      # Windows; use .venv/bin on POSIX
cp .env.example .env                                # add ENTSOE_TOKEN later for live data
export PYTHONPATH=.                                 # so `eml` is importable

# --- synthetic-bridge demo (no API key needed) ---
python scripts/pull_history.py  2023-01-01 2025-01-01   # real ERA5 weather → warehouse
python scripts/gen_synthetic.py 2023-01-01 2025-01-01   # synthetic load/gen/price
python scripts/build_features.py                        # 44-feature matrix + coverage report
python scripts/train.py                                 # train + backtest (prints scorecard)
python scripts/brief.py                                 # print tomorrow's morning brief
python -m uvicorn eml.api.main:app --port 8000          # dashboard → http://127.0.0.1:8000
```

**Going live:** get the ENTSO-E token (§10), then
`python scripts/pull_entsoe.py 2022-01-01 2024-01-01` and retrain. Nothing downstream changes.

---

## 14. Roadmap

`PLAN.md` is authoritative. In brief: real ENTSO-E feed → more bidding zones → fuel/carbon feeds
(spark/dark spreads) → dedicated load/wind/solar sub-forecasts → risk module (VaR, Expected
Shortfall, Monte Carlo) → optimization (battery/VPP dispatch) → LLM-written briefs → React
frontend → PostgreSQL/TimescaleDB + Celery scheduling.

---

## 15. Glossary

- **ADMIE / IPTO** — Greek electricity transmission system operator (runs the grid).
- **Balancing market** — real-time market the TSO uses to keep supply=demand.
- **Bidding zone** — area with a single wholesale price (Greece = `GR`).
- **CQR** — Conformalized Quantile Regression; calibrates prediction intervals to true coverage.
- **DAM** — Day-Ahead Market; hourly auction for next-day power. **The forecast target.**
- **Dark spread** — coal-plant gross margin: power − coal/efficiency.
- **Degree-day (HDD/CDD)** — heating/cooling demand proxy from temperature vs a base.
- **ENTSO-E** — European TSO association; runs the free Transparency data platform.
- **ETS / EUA** — EU Emissions Trading System / its carbon allowance (CO₂ price).
- **Gate closure** — DAM bid deadline; the leakage boundary for features.
- **HEnEx** — Hellenic Energy Exchange (runs the Greek power market).
- **IDM** — Intraday Market; continuous trading after the DAM.
- **LightGBM** — gradient-boosted-tree library; the model engine.
- **MAE / RMSE** — mean absolute / root-mean-square error (accuracy metrics).
- **Marginal plant** — the most expensive generator needed; it sets the price for all.
- **Merit order** — generators dispatched cheapest-first; the pricing mechanism.
- **MWh** — megawatt-hour, the energy unit prices are quoted in (€/MWh).
- **Pinball loss** — proper scoring rule for quantile forecasts.
- **Prediction interval (P10–P90)** — range the outcome should fall in 80% of the time.
- **Quantile regression** — predicts a percentile (e.g. P90) rather than the mean.
- **Residual load** — demand minus wind and solar; the key price driver.
- **SHAP** — per-feature contribution to a prediction; powers the "why".
- **Spark spread** — gas-plant gross margin: power − gas/efficiency.
- **TSO** — Transmission System Operator (ADMIE in Greece).
- **TTF** — the benchmark European natural-gas price hub.

---

## 16. Disclaimer

This is an educational / portfolio project. Prices are currently generated by a synthetic bridge
(§9), **not** a market forecast of record, and nothing here is trading or investment advice.

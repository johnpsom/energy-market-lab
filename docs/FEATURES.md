# Feature Catalog — Layer 4

The feature engine turns every raw datum into a model-ready signal. All features are hourly,
indexed on local **Europe/Athens** time (the market's clock; DST-aware once tz data flows).

**Leakage rule (critical for a credible backtest):** a feature for target hour *h* may only use
information knowable *before the day-ahead auction closes* (gate closure ~12:00 D-1 for delivery
day D). So realized-price features use lags ≥ 24 h; weather uses *forecasts* (which is what we
store), not realized weather. Every feature below is tagged **[safe]** or **[leak-risk]**.

Legend: ✅ live on real data · 🔜 roadmap. (All calendar/weather/price/grid/fuel features are live;
cross-border is the active next addition.)

---

## Calendar ✅  (`features/calendar.py`)
Demand and price have strong deterministic cycles; models need them explicitly.

| Feature | Notes |
|---|---|
| `hour`, `dayofweek`, `month`, `day_of_year` | raw ordinals [safe] |
| `hour_sin/cos`, `dow_sin/cos`, `doy_sin/cos` | cyclical encodings so 23:00≈00:00 [safe] |
| `is_weekend` | Sat/Sun demand regime [safe] |
| `is_peak` | 08:00–20:00 local (HEnEx peak block) [safe] |
| `is_holiday` | Greek public holidays incl. **moveable Orthodox Easter** [safe] |
| `is_bridge_day` | workday wedged between holiday & weekend (low load) [safe] |
| `season` | 0–3; couples with HDD/CDD [safe] |

## Weather ✅  (`features/weather.py`)
Aggregated across representative Greek sites (demand centres vs wind/solar regions).

| Feature | Reasoning |
|---|---|
| `temp_demand` | population-weighted temp (Athens+Thessaloniki) → drives load [safe, forecast] |
| `hdd`, `cdd` | heating/cooling degree-hours vs 18 °C / 22 °C bases → non-linear load [safe] |
| `wind_speed_agg` | mean 100 m wind over wind sites [safe, forecast] |
| `wind_power_proxy` | wind speed pushed through a turbine power curve (cut-in 3, rated 12, cut-out 25 m/s) → non-linear supply [safe] |
| `wind_ramp_1h`, `wind_ramp_3h` | Δ wind proxy → balancing/price-spike driver [safe] |
| `wind_persistence_24h` | rolling std of wind → predictability of wind supply [safe] |
| `solar_rad_agg` | mean shortwave radiation over solar sites [safe, forecast] |
| `solar_power_proxy` | radiation × temperature-derating (panels lose ~0.4 %/°C above 25) [safe] |
| `cloud_index` | mean cloud cover → solar suppression [safe] |
| `precip_agg` | precipitation → hydro inflow & demand context [safe] |

## Price ✅  (`features/price.py`)
Autoregressive structure dominates short-term price.

| Feature | Notes |
|---|---|
| `price_lag_24h/48h/168h` | same-hour prices 1d/2d/1w ago [safe] |
| `price_roll_mean_24h/168h` | level [safe if shifted ≥24h] |
| `price_roll_std_24h` | realized **volatility** [safe] |
| `price_momentum_24h` | price − price_24h [safe] |
| `price_accel` | Δmomentum → turning points [safe] |
| `price_roll_skew_168h` | asymmetry → spike-prone regimes [safe] |
| `peak_offpeak_spread_d1` | prior-day peak minus off-peak level [safe] |

## Grid ✅  (`features/grid.py`) — ENTSO-E load + generation
The single most important price driver in a renewables system.

| Feature | Reasoning |
|---|---|
| `residual_load` | **load − wind − solar** → what thermal must cover; the merit-order dial. Built from the day-ahead **load forecast**; wind/solar from actuals today, moving to the **RES day-ahead forecast** (roadmap 2) to be fully leak-clean |
| `renewable_penetration` | (wind+solar)/load → negative-price & low-price signal [safe] |
| `wind_share`, `solar_share` | composition of supply [safe] |
| `load_ramp_1h/3h` | demand ramps → reserve stress [safe] |
| `thermal_gen` | fossil dispatch level [leak-risk realized — NaN in the forecast horizon, handled] |

## Fuel & Carbon ✅  (`features/fuel.py`) — TTF gas, EUA carbon, Brent (yfinance)
Set the marginal cost of thermal plant → the price floor in tight hours. Daily settlements lagged
one trading day (prior close known before gate closure) and forward-filled to hourly.

| Feature | Meaning [all safe] |
|---|---|
| `gas_price` | TTF front-month (EUR/MWh_th) — the dominant fuel-cost driver |
| `carbon_price` | EUA (EUR/t) — CO₂ cost on fossil generation |
| `gas_srmc` | gas-plant short-run marginal cost `(gas + EUA·EF)/η` — the theoretical price floor |
| `gas_mom_7d` | ~1-week change in gas → momentum of the cost base |
| `brent_price` | Brent crude (context / oil-indexed contracts) |

*Note:* spark/dark/clean spreads embed the power price (the target) so they are **outputs**, not
model inputs — they belong in the analytics/dashboard, not the feature matrix.

## Cross-border ✅  (`features/crossborder.py`) — ENTSO-E neighbor prices
Greece is a net importer; neighbor price levels and the GR−neighbor spread strongly shape GR price.

| Feature | Reasoning |
|---|---|
| `it_price_lag24/168`, `bg_price_lag24/168` | lagged Italy-South / Bulgaria day-ahead prices (coupled markets clear together, so use lags) [safe] |
| `gr_it_spread_lag24`, `gr_bg_spread_lag24` | prior-day GR−neighbor gap → congestion & flow-direction signal [safe] |
| `neighbor_min_lag24` | cheapest import source yesterday → the effective price cap [safe] |

*Net position / interconnector flows omitted for now — ENTSO-E returns zeros for GR day-ahead net
position; the neighbor-price spread already captures the import-pull dynamic.*

---

### Beyond the original list (proposed additions, rationale)
- **`residual_load`** — the best single predictor of price in a high-RES grid; more informative than raw load or raw wind alone.
- **Turbine/panel power-curve proxies** instead of raw wind speed / radiation — the physics is non-linear; giving the model the curve beats making it relearn it from little data.
- **Ramp features** (`wind_ramp`, `load_ramp`) — price spikes and balancing activations live in the *changes*, not the levels.
- **`clean` spreads & `switch_price`** — carbon-adjusted; capture the coal↔gas marginal-fuel flip that moves European prices.
- **`price_roll_skew`** — flags regimes where spikes cluster, feeding the spike-probability output.
- **Neighbor-zone spread** (later, multi-zone) — GR vs IT/BG price gap predicts congestion & flow direction.

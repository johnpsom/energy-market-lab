"""Live update: catch up realized data + pull forward forecasts so the dashboard sees today
and tomorrow, not just the last fully-realized day.

- recent ACTUALS: prices, load, generation (last ~12 days) — catches publication lag
- forward FORECASTS: load forecast + wind/solar forecast (through tomorrow) — the inputs a
  day-ahead price forecast needs before the day happens
- weather: refresh Open-Meteo (recent past + 7-day forecast)

Renewable forecasts are stored as source='entsoe_fc' for FUTURE timestamps only (beyond the
last realized generation), so they never double-count or overwrite actuals. Re-run any time.
"""
import pandas as pd
import sqlalchemy as sa

from eml.collectors import entsoe_collector as ec
from eml.collectors import weather_collector as wc
from eml.db import get_engine, init_schema, read_sql, upsert
from eml.config import settings


def _last_actual_gen_ts() -> pd.Timestamp:
    df = read_sql("select max(ts) m from generation where source='entsoe'")
    return pd.to_datetime(df["m"].iloc[0])


if __name__ == "__main__":
    init_schema()
    now = pd.Timestamp.now(tz=settings.timezone).normalize().tz_localize(None)
    recent = (now - pd.Timedelta(days=12)).strftime("%Y-%m-%d")
    fwd_end = (now + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    today = now.strftime("%Y-%m-%d")

    # --- realized actuals (catch up lag) ---
    np_ = upsert("prices", ec.fetch_day_ahead_prices(recent, fwd_end))      # DAM incl. tomorrow
    nla = upsert("load", ec.fetch_load(recent, today))
    nlf = upsert("load", ec.fetch_load_forecast(recent, fwd_end))            # forecast incl. fwd
    ng = upsert("generation", ec.fetch_generation(recent, today))
    print(f"actuals: prices={np_} load_actual={nla} load_fcst={nlf} generation={ng}", flush=True)

    # --- weather refresh ---
    nw = upsert("weather", wc.fetch(days=7, past_days=16))
    print(f"weather: {nw} rows", flush=True)

    # --- forward renewable forecast (future timestamps only) ---
    last_gen = _last_actual_gen_ts()
    res = ec.fetch_wind_solar_forecast(today, fwd_end)
    res["ts"] = pd.to_datetime(res["ts"])
    if getattr(res["ts"].dt, "tz", None) is not None:
        res["ts"] = res["ts"].dt.tz_localize(None)   # match stored naive-local timestamps
    res = res[res["ts"] > last_gen]
    with get_engine().begin() as c:
        c.execute(sa.text("DELETE FROM generation WHERE source='entsoe_fc'"))
    nres = upsert("generation", res)
    print(f"renewable forecast (entsoe_fc): {nres} rows, {res['ts'].min()} -> {res['ts'].max()}"
          if nres else "renewable forecast: none returned", flush=True)
    print(f"last realized generation: {last_gen}")

    # --- extend outlook beyond ENTSO-E horizon (weather-derived renewables + load climatology) ---
    from eml.outlook_extend import extend
    ext = extend(days=7)
    print(f"outlook extension: gen={ext['gen_rows']} load={ext['load_rows']} "
          f"(wind_cap~{ext.get('wind_cap')}MW solar_cap~{ext.get('solar_cap')}MW) -> {ext.get('horizon')}")

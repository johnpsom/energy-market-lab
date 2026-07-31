"""Extend the forecast horizon beyond ENTSO-E's day-ahead publication.

ENTSO-E only publishes load & renewable forecasts ~a day out, so a real multi-day outlook needs
estimated inputs for days 2-7:
  * renewables — from the Open-Meteo weather forecast via the turbine/panel power curves,
    scaled by recent installed-capacity (source='weather_fc')
  * load — a recent (day-of-week, hour) climatology of actual load (kind='forecast',
    source='clim_fc')

Both are written ONLY for future timestamps beyond the existing ENTSO-E coverage, so they never
double-count or overwrite the real day-ahead forecasts. Clearly a lower-fidelity, weather-driven
extension — flagged as such on the dashboard. Re-run wipes and rebuilds the extension.
"""
from __future__ import annotations

import pandas as pd
import sqlalchemy as sa

from .config import settings
from .db import get_engine, read_sql, upsert
from .features import weather as wx


def _res_scale(fuel: str, proxy: pd.Series) -> float:
    """Scale that maps the weather power-curve proxy (0-1) to real MW output, calibrated so
    recent proxy energy matches recent actual generation energy (scale = Σactual / Σproxy)."""
    df = read_sql("select ts, value from generation "
                  f"where source='entsoe' and fuel='{fuel}' and ts > date('now','-120 day')")
    if df.empty:
        return 0.0
    g = pd.Series(df["value"].values, index=pd.to_datetime(df["ts"])).resample("h").mean()
    p = proxy.reindex(g.index)
    m = g.notna() & p.notna() & (p > 0.02)
    if m.sum() < 50 or p[m].sum() == 0:
        return float(g.max() / max(proxy.max(), 1e-6))
    return float(g[m].sum() / p[m].sum())


def _load_climatology() -> pd.Series:
    """Median actual load by (day-of-week, hour) over the last ~8 weeks."""
    df = read_sql("select ts, value from load where source='entsoe' and kind='actual' "
                  "and ts > date('now','-56 day')")
    if df.empty:
        return pd.Series(dtype=float)
    s = pd.Series(df["value"].values, index=pd.to_datetime(df["ts"])).resample("h").mean().dropna()
    return s.groupby([s.index.dayofweek, s.index.hour]).median()


def extend(days: int = 7) -> dict:
    now = pd.Timestamp.now(tz=settings.timezone).normalize().tz_localize(None)
    horizon = now + pd.Timedelta(days=days)
    zone = settings.default_zone

    wfeat = wx.build()                       # weather features incl. wind/solar power proxies
    fut = wfeat[(wfeat.index >= now) & (wfeat.index < horizon)]
    if fut.empty:
        return {"gen_rows": 0, "load_rows": 0}

    wind_cap = _res_scale("Wind Onshore", wfeat["wind_power_proxy"])
    solar_cap = _res_scale("Solar", wfeat["solar_power_proxy"])

    # renewables beyond existing real coverage (actual + ENTSO-E day-ahead forecast)
    last_res = pd.to_datetime(read_sql(
        "select max(ts) m from generation where source in ('entsoe','entsoe_fc') "
        f"and fuel in ('Wind Onshore','Solar')")["m"].iloc[0])
    fr = fut[fut.index > last_res] if pd.notna(last_res) else fut
    gen_rows = []
    for _, ts in enumerate(fr.index):
        gen_rows.append((ts, zone, "Wind Onshore", "weather_fc",
                         float(fr.loc[ts, "wind_power_proxy"]) * wind_cap))
        gen_rows.append((ts, zone, "Solar", "weather_fc",
                         float(fr.loc[ts, "solar_power_proxy"]) * solar_cap))
    gdf = pd.DataFrame(gen_rows, columns=["ts", "zone", "fuel", "source", "value"])

    # load climatology beyond existing forecast coverage
    clim = _load_climatology()
    last_load_fc = pd.to_datetime(read_sql(
        "select max(ts) m from load where source='entsoe' and kind='forecast'")["m"].iloc[0])
    fl = fut[fut.index > last_load_fc] if pd.notna(last_load_fc) else fut
    load_rows = []
    if not clim.empty:
        for ts in fl.index:
            key = (ts.dayofweek, ts.hour)
            if key in clim.index:
                load_rows.append((ts, zone, "forecast", "clim_fc", float(clim.loc[key])))
    ldf = pd.DataFrame(load_rows, columns=["ts", "zone", "kind", "source", "value"])

    with get_engine().begin() as c:
        c.execute(sa.text("DELETE FROM generation WHERE source='weather_fc'"))
        c.execute(sa.text("DELETE FROM load WHERE source='clim_fc'"))
    ng = upsert("generation", gdf)
    nl = upsert("load", ldf)
    return {"gen_rows": ng, "load_rows": nl,
            "wind_cap": round(wind_cap), "solar_cap": round(solar_cap),
            "horizon": str(fr.index.max()) if not fr.empty else None}


if __name__ == "__main__":
    print(extend())

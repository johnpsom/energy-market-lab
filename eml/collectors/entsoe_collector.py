"""ENTSO-E Transparency Platform collector — Layer 1-2.

Free token required (see .env.example). Covers the Greek bidding zone (GR): day-ahead
prices, system load (actual + forecast), and generation by fuel/technology. Returns
long-format DataFrames ready for `db.upsert(...)`.

Wraps `entsoe-py` (EntsoePandasClient). Times are handled tz-aware in Europe/Athens.
"""
from __future__ import annotations

import pandas as pd

from ..config import settings

SOURCE = "entsoe"


def _client():
    if not settings.has_entsoe:
        raise RuntimeError(
            "ENTSOE_TOKEN is not set. Get a free token (see .env.example), put it in .env, "
            "and re-run. Until then, ENTSO-E data cannot be pulled."
        )
    from entsoe import EntsoePandasClient
    return EntsoePandasClient(api_key=settings.entsoe_token)


def _window(start: str, end: str, tz: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    return pd.Timestamp(start, tz=tz), pd.Timestamp(end, tz=tz)


def fetch_day_ahead_prices(start: str, end: str, zone: str | None = None) -> pd.DataFrame:
    """Day-ahead (DAM) prices. Columns [ts, zone, product, source, value] (EUR/MWh)."""
    zone = zone or settings.default_zone
    s, e = _window(start, end, settings.timezone)
    series = _client().query_day_ahead_prices(zone, start=s, end=e)
    df = series.rename("value").rename_axis("ts").reset_index()
    df["zone"] = zone
    df["product"] = "day_ahead"
    df["source"] = SOURCE
    return df[["ts", "zone", "product", "source", "value"]]


def fetch_load(start: str, end: str, zone: str | None = None) -> pd.DataFrame:
    """Actual system load. Columns [ts, zone, kind, source, value] (MW)."""
    zone = zone or settings.default_zone
    s, e = _window(start, end, settings.timezone)
    series = _client().query_load(zone, start=s, end=e)
    # entsoe-py returns a 1-col DataFrame ('Actual Load'); normalise to a Series.
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    df = series.rename("value").rename_axis("ts").reset_index()
    df["zone"] = zone
    df["kind"] = "actual"
    df["source"] = SOURCE
    return df[["ts", "zone", "kind", "source", "value"]]


def fetch_load_forecast(start: str, end: str, zone: str | None = None) -> pd.DataFrame:
    """Day-ahead load forecast. Columns [ts, zone, kind, source, value] (MW)."""
    zone = zone or settings.default_zone
    s, e = _window(start, end, settings.timezone)
    series = _client().query_load_forecast(zone, start=s, end=e)
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    df = series.rename("value").rename_axis("ts").reset_index()
    df["zone"] = zone
    df["kind"] = "forecast"
    df["source"] = SOURCE
    return df[["ts", "zone", "kind", "source", "value"]]


def fetch_generation(start: str, end: str, zone: str | None = None) -> pd.DataFrame:
    """Actual generation per fuel/technology. Columns [ts, zone, fuel, source, value] (MW)."""
    zone = zone or settings.default_zone
    s, e = _window(start, end, settings.timezone)
    wide = _client().query_generation(zone, start=s, end=e)
    # Columns may be a MultiIndex (fuel, 'Actual Aggregated'/'Actual Consumption').
    if isinstance(wide.columns, pd.MultiIndex):
        wide = wide.xs("Actual Aggregated", axis=1, level=-1, drop_level=True)
    long = wide.rename_axis("ts").reset_index().melt(
        id_vars="ts", var_name="fuel", value_name="value"
    )
    long["zone"] = zone
    long["source"] = SOURCE
    long = long.dropna(subset=["value"])
    return long[["ts", "zone", "fuel", "source", "value"]]


if __name__ == "__main__":  # smoke test (needs a token)
    prices = fetch_day_ahead_prices("2024-01-01", "2024-01-08")
    print(prices.head())
    print(f"{len(prices)} DAM price rows for {settings.default_zone}")

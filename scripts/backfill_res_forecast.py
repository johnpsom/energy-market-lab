"""Backfill ENTSO-E historical wind/solar DAY-AHEAD forecasts (source='entsoe_fc').

Usage:  python scripts/backfill_res_forecast.py 2023-01-01 2026-08-02
Lets residual load be built from the FORECAST renewables that were known before each auction,
instead of realized generation — removing leakage and matching live serving conditions.
"""
import sys

import pandas as pd

from eml.collectors import entsoe_collector as ec
from eml.db import init_schema, upsert


def _chunks(start, end):
    edges = [pd.Timestamp(start)] + \
        [e for e in pd.date_range(start, end, freq="YS") if pd.Timestamp(start) < e < pd.Timestamp(end)] + \
        [pd.Timestamp(end)]
    edges = sorted(set(edges))
    return [(edges[i].strftime("%Y-%m-%d"), edges[i + 1].strftime("%Y-%m-%d")) for i in range(len(edges) - 1)]


if __name__ == "__main__":
    start = sys.argv[1] if len(sys.argv) > 1 else "2023-01-01"
    end = sys.argv[2] if len(sys.argv) > 2 else "2026-08-02"
    init_schema()
    n = 0
    for s, e in _chunks(start, end):
        try:
            n += upsert("generation", ec.fetch_wind_solar_forecast(s, e))
            print(f"  {s} -> {e}: ok", flush=True)
        except Exception as ex:
            print(f"  {s} -> {e}: WARN {type(ex).__name__}", flush=True)
    print(f"RES day-ahead forecast: {n} rows (source=entsoe_fc)")

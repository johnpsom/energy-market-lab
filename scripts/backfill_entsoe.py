"""Backfill real ENTSO-E history year-by-year (respects the API's per-request range limit).

Usage:  python scripts/backfill_entsoe.py 2023-01-01 2026-07-31
Pulls day-ahead prices, load (actual + forecast), and generation-by-fuel for the Greek zone
and upserts them (source='entsoe'). Idempotent — safe to re-run.
"""
import sys

import pandas as pd

from eml.collectors import entsoe_collector as ec
from eml.db import init_schema, upsert


def _chunks(start: str, end: str):
    edges = pd.date_range(start, end, freq="YS").tolist()
    edges = [pd.Timestamp(start)] + [e for e in edges if pd.Timestamp(start) < e < pd.Timestamp(end)] + [pd.Timestamp(end)]
    edges = sorted(set(edges))
    return [(edges[i].strftime("%Y-%m-%d"), edges[i + 1].strftime("%Y-%m-%d"))
            for i in range(len(edges) - 1)]


if __name__ == "__main__":
    start = sys.argv[1] if len(sys.argv) > 1 else "2023-01-01"
    end = sys.argv[2] if len(sys.argv) > 2 else "2026-07-31"
    init_schema()
    tot = {"prices": 0, "load": 0, "generation": 0}
    for s, e in _chunks(start, end):
        try:
            tot["prices"] += upsert("prices", ec.fetch_day_ahead_prices(s, e))
            tot["load"] += upsert("load", ec.fetch_load(s, e))
            tot["load"] += upsert("load", ec.fetch_load_forecast(s, e))
            tot["generation"] += upsert("generation", ec.fetch_generation(s, e))
            print(f"  {s} -> {e}: ok", flush=True)
        except Exception as ex:  # keep going; some series have gaps in some years
            print(f"  {s} -> {e}: WARN {type(ex).__name__}: {ex}", flush=True)
    print(f"DONE  prices={tot['prices']}  load={tot['load']}  generation={tot['generation']}")

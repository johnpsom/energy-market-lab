"""Backfill neighbor day-ahead prices (Italy-South, Bulgaria) for cross-border features.

Usage:  python scripts/pull_crossborder.py 2023-01-01 2026-08-01
Greece is a net importer; the price LEVEL in coupled neighbors (and the GR-neighbor spread) is a
strong predictor. Stored in the `prices` table under zone='IT_SUD' / 'BG'. Idempotent.
"""
import sys

import pandas as pd

from eml.collectors import entsoe_collector as ec
from eml.db import init_schema, upsert

NEIGHBORS = ["IT_SUD", "BG"]


def _chunks(start, end):
    edges = [pd.Timestamp(start)] + \
        [e for e in pd.date_range(start, end, freq="YS") if pd.Timestamp(start) < e < pd.Timestamp(end)] + \
        [pd.Timestamp(end)]
    edges = sorted(set(edges))
    return [(edges[i].strftime("%Y-%m-%d"), edges[i + 1].strftime("%Y-%m-%d")) for i in range(len(edges) - 1)]


if __name__ == "__main__":
    start = sys.argv[1] if len(sys.argv) > 1 else "2023-01-01"
    end = sys.argv[2] if len(sys.argv) > 2 else "2026-08-01"
    init_schema()
    for zone in NEIGHBORS:
        n = 0
        for s, e in _chunks(start, end):
            try:
                n += upsert("prices", ec.fetch_day_ahead_prices(s, e, zone=zone))
            except Exception as ex:
                print(f"  {zone} {s}->{e}: WARN {type(ex).__name__}", flush=True)
        print(f"{zone}: {n} day-ahead price rows", flush=True)

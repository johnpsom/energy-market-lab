"""Backfill historical weather (Open-Meteo ERA5 archive) into the warehouse.

Usage:  python scripts/pull_history.py 2023-01-01 2025-01-01
Free, no API key. Provides the training span for the synthetic bridge and, later, real models.
"""
import sys

from eml.collectors import weather_collector as wc
from eml.db import init_schema, upsert

if __name__ == "__main__":
    start = sys.argv[1] if len(sys.argv) > 1 else "2023-01-01"
    end = sys.argv[2] if len(sys.argv) > 2 else "2025-01-01"
    init_schema()
    df = wc.fetch_history(start, end)
    n = upsert("weather", df)
    print(f"history: upserted {n} weather rows "
          f"({df['location'].nunique()} sites x {df['variable'].nunique()} vars, "
          f"{df['ts'].min()} -> {df['ts'].max()})")

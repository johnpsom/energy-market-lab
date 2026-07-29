"""Pull ENTSO-E day-ahead prices, load, and generation into the warehouse.

Usage:  python scripts/pull_entsoe.py 2024-01-01 2024-02-01
Needs ENTSOE_TOKEN in .env (see .env.example).
"""
import sys

from eml.collectors import entsoe_collector as ec
from eml.db import init_schema, upsert

if __name__ == "__main__":
    start = sys.argv[1] if len(sys.argv) > 1 else "2024-01-01"
    end = sys.argv[2] if len(sys.argv) > 2 else "2024-02-01"
    init_schema()

    n_price = upsert("prices", ec.fetch_day_ahead_prices(start, end))
    n_load = upsert("load", ec.fetch_load(start, end))
    n_fc = upsert("load", ec.fetch_load_forecast(start, end))
    n_gen = upsert("generation", ec.fetch_generation(start, end))
    print(f"prices={n_price}  load(actual)={n_load}  load(forecast)={n_fc}  generation={n_gen}"
          f"  [{start} -> {end}]")

"""Generate synthetic fundamentals (load/generation/price) from stored weather.

Usage:  python scripts/gen_synthetic.py 2023-01-01 2025-01-01
Requires weather history in the warehouse (run scripts/pull_history.py first).
"""
import sys

from eml.db import init_schema
from eml.synthetic.generator import generate

if __name__ == "__main__":
    start = sys.argv[1] if len(sys.argv) > 1 else "2023-01-01"
    end = sys.argv[2] if len(sys.argv) > 2 else "2025-01-01"
    init_schema()
    summary = generate(start, end)
    print(f"synthetic: {summary['rows']} hourly rows  {summary['span'][0]} -> {summary['span'][1]}")
    print(f"  DAM price EUR/MWh: {summary['price_stats']}")

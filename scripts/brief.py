"""Print the natural-language morning brief for a delivery day.

Usage:  python scripts/brief.py [YYYY-MM-DD]   (default: last full day in the warehouse)
"""
import sys

from eml.narrative.brief import generate

if __name__ == "__main__":
    date = sys.argv[1] if len(sys.argv) > 1 else None
    print(generate(date)["text"])

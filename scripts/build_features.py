"""Build the feature matrix from the warehouse and print a coverage report.

Runs today on weather+calendar; auto-widens to price+grid once ENTSO-E data is loaded.
"""
import pandas as pd

from eml.features.build import build_matrix, feature_report

pd.set_option("display.max_rows", 200)
pd.set_option("display.width", 120)

if __name__ == "__main__":
    m = build_matrix()
    if m.empty:
        raise SystemExit("No data in warehouse yet — run scripts/pull_weather.py first.")
    print(f"Feature matrix: {m.shape[0]} hourly rows x {m.shape[1]} features "
          f"[{m.index.min()} -> {m.index.max()}]\n")
    print(feature_report(m).to_string())
    out = "data/features_preview.parquet"
    m.to_parquet(out)
    print(f"\nSaved -> {out}")

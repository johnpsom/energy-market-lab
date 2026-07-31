"""Pull TTF gas, EUA carbon, and Brent daily settlements into the warehouse (free, yfinance)."""
from eml.collectors import fuel_collector as fc
from eml.db import init_schema, upsert

if __name__ == "__main__":
    init_schema()
    df = fc.fetch("2022-06-01")
    n = upsert("fuel", df)
    print(f"fuel: upserted {n} rows")
    print(df.groupby("commodity")["value"].agg(["count", "min", "max", "last"]).round(1).to_string())

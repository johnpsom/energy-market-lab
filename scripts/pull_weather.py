"""Pull Open-Meteo weather into the warehouse. Runs today with no API key."""
from eml.collectors import weather_collector
from eml.db import init_schema, upsert

if __name__ == "__main__":
    init_schema()
    df = weather_collector.fetch(days=7)
    n = upsert("weather", df)
    print(f"weather: upserted {n} rows "
          f"({df['location'].nunique()} sites x {df['variable'].nunique()} vars, "
          f"{df['ts'].min()} -> {df['ts'].max()})")

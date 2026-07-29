"""Open-Meteo weather collector — Layer 1-2.

Free, no API key. Pulls hourly forecast variables that drive power demand and renewable
generation, for a set of named Greek sites (a demand centre + representative wind/solar
regions). Returns long-format rows ready for `db.upsert('weather', df)`.

Docs: https://open-meteo.com/en/docs
"""
from __future__ import annotations

import pandas as pd
import requests

API = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_API = "https://archive-api.open-meteo.com/v1/archive"  # ERA5 reanalysis, free, no key
SOURCE = "open-meteo"

# Representative Greek sites. Demand is temperature-driven (Athens/Thessaloniki load
# centres); renewables are sited where the resource is (Cyclades wind, Peloponnese/Crete solar).
SITES = {
    "athens":       (37.98, 23.73),   # main demand centre
    "thessaloniki": (40.64, 22.94),   # northern demand centre
    "cyclades":     (37.10, 25.15),   # strong wind corridor
    "peloponnese":  (37.30, 22.10),   # solar / mixed
    "crete":        (35.24, 24.81),   # solar + wind, weakly interconnected
}

# Variables: demand (temperature, humidity), wind gen (wind speed/direction @100m hub
# height), solar gen (shortwave radiation, cloud cover), plus pressure/precip context.
HOURLY = [
    "temperature_2m",
    "relative_humidity_2m",
    "cloud_cover",
    "wind_speed_100m",
    "wind_direction_100m",
    "shortwave_radiation",
    "surface_pressure",
    "precipitation",
]


def fetch(days: int = 7, timezone: str = "Europe/Athens", past_days: int = 0) -> pd.DataFrame:
    """Fetch `days` of hourly forecast (+ optional `past_days` of recent history to bridge the
    ERA5 archive's ~5-day lag) for all SITES. Returns long-format DataFrame:
    columns [ts, location, variable, source, value]."""
    frames: list[pd.DataFrame] = []
    for name, (lat, lon) in SITES.items():
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": ",".join(HOURLY),
            "forecast_days": days,
            "past_days": past_days,
            "timezone": timezone,
            "wind_speed_unit": "ms",   # default is km/h; our turbine curve expects m/s
        }
        resp = requests.get(API, params=params, timeout=30)
        resp.raise_for_status()
        hourly = resp.json()["hourly"]
        ts = pd.to_datetime(hourly["time"])
        wide = pd.DataFrame({"ts": ts})
        for var in HOURLY:
            wide[var] = hourly.get(var)
        long = wide.melt(id_vars="ts", var_name="variable", value_name="value")
        long["location"] = name
        frames.append(long)

    df = pd.concat(frames, ignore_index=True)
    df["source"] = SOURCE
    df = df.dropna(subset=["value"])
    return df[["ts", "location", "variable", "source", "value"]]


def fetch_history(start_date: str, end_date: str, timezone: str = "Europe/Athens") -> pd.DataFrame:
    """Fetch historical hourly weather (ERA5 archive) for all SITES over [start_date, end_date].
    Same long-format schema as `fetch`. Used to build a training span for the models."""
    frames: list[pd.DataFrame] = []
    for name, (lat, lon) in SITES.items():
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(HOURLY),
            "timezone": timezone,
            "wind_speed_unit": "ms",
        }
        resp = requests.get(ARCHIVE_API, params=params, timeout=90)
        resp.raise_for_status()
        hourly = resp.json()["hourly"]
        wide = pd.DataFrame({"ts": pd.to_datetime(hourly["time"])})
        for var in HOURLY:
            wide[var] = hourly.get(var)
        long = wide.melt(id_vars="ts", var_name="variable", value_name="value")
        long["location"] = name
        frames.append(long)
    df = pd.concat(frames, ignore_index=True)
    df["source"] = SOURCE
    df = df.dropna(subset=["value"])
    return df[["ts", "location", "variable", "source", "value"]]


if __name__ == "__main__":  # quick smoke test
    out = fetch(days=2)
    print(out.head())
    print(f"\n{len(out)} rows | {out['location'].nunique()} sites | "
          f"{out['variable'].nunique()} variables | "
          f"{out['ts'].min()} -> {out['ts'].max()}")

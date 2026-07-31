"""Fuel & carbon collector — Layer 1-2.

Daily settlement prices for the commodities that set the marginal cost of thermal generation,
i.e. the LEVEL of power price: TTF gas, EUA carbon, and Brent (context). Free via Yahoo Finance
(yfinance). Returns long-format rows ready for db.upsert('fuel', df).

These are the drivers the weather/calendar/grid features can't supply — without them the price
model has a systematic level bias (it can't know fuel got cheaper or carbon rose).
"""
from __future__ import annotations

import warnings

import pandas as pd

warnings.filterwarnings("ignore")

SOURCE = "yfinance"
# Yahoo ticker -> (commodity name, unit)
TICKERS = {
    "TTF=F": ("gas_ttf", "EUR/MWh"),    # Dutch TTF natural gas front-month
    "CO2.L": ("carbon_eua", "EUR/t"),   # ICE EUA carbon front-December
    "BZ=F": ("brent", "USD/bbl"),       # Brent crude (context)
}


def fetch(start: str = "2022-06-01") -> pd.DataFrame:
    """Fetch daily close for each commodity since `start`. Long format:
    columns [ts, commodity, source, unit, value]. ts is naive daily (settlement date)."""
    import yfinance as yf

    frames: list[pd.DataFrame] = []
    for ticker, (name, unit) in TICKERS.items():
        h = yf.Ticker(ticker).history(start=start, auto_adjust=False)
        if h.empty:
            continue
        s = h["Close"].dropna()
        idx = s.index
        idx = idx.tz_localize(None) if idx.tz is not None else idx
        d = pd.DataFrame({"ts": idx.normalize(), "value": s.values})
        d["commodity"], d["source"], d["unit"] = name, SOURCE, unit
        frames.append(d)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)[["ts", "commodity", "source", "unit", "value"]]


if __name__ == "__main__":
    df = fetch()
    print(df.groupby("commodity")["ts"].agg(["min", "max", "count"]))
    print(df.groupby("commodity")["value"].last())

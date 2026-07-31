"""Warehouse: SQLAlchemy engine + schema + upsert.

Schema is deliberately Postgres/TimescaleDB-compatible so migrating off SQLite is a
DATABASE_URL change, not a rewrite. On Postgres the timeseries tables become hypertables
(see docs/ for the `create_hypertable` calls) — on SQLite they are plain tables.

All timeseries tables share the shape (ts, zone, ...dimensions..., value, source) and a
composite primary key so re-pulling the same window upserts instead of duplicating.
"""
from __future__ import annotations

import pandas as pd
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    MetaData,
    String,
    Table,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.engine import Engine

from .config import settings

metadata = MetaData()

# --- Layer 3: warehouse tables -------------------------------------------------

# Day-ahead / intraday / balancing prices.
prices = Table(
    "prices", metadata,
    Column("ts", DateTime(timezone=True), primary_key=True),
    Column("zone", String, primary_key=True),
    Column("product", String, primary_key=True),   # 'day_ahead' | 'intraday' | 'balancing'
    Column("source", String, primary_key=True),     # 'entsoe' | ...
    Column("value", Float),                          # EUR/MWh
)

# System load (demand). kind = 'actual' | 'forecast'.
load = Table(
    "load", metadata,
    Column("ts", DateTime(timezone=True), primary_key=True),
    Column("zone", String, primary_key=True),
    Column("kind", String, primary_key=True),
    Column("source", String, primary_key=True),
    Column("value", Float),                          # MW
)

# Generation by fuel/technology (long format — one row per fuel per timestamp).
generation = Table(
    "generation", metadata,
    Column("ts", DateTime(timezone=True), primary_key=True),
    Column("zone", String, primary_key=True),
    Column("fuel", String, primary_key=True),        # 'Wind Onshore' | 'Solar' | 'Fossil Gas' | ...
    Column("source", String, primary_key=True),
    Column("value", Float),                          # MW
)

# Weather (forecast). Long format: one row per (location, variable) per timestamp.
weather = Table(
    "weather", metadata,
    Column("ts", DateTime(timezone=True), primary_key=True),
    Column("location", String, primary_key=True),    # named site, e.g. 'athens'
    Column("variable", String, primary_key=True),     # 'temperature_2m' | 'wind_speed_100m' | ...
    Column("source", String, primary_key=True),       # 'open-meteo'
    Column("value", Float),
)

# Fuel & carbon daily settlements (TTF gas, EUA carbon, Brent). Daily granularity.
fuel = Table(
    "fuel", metadata,
    Column("ts", DateTime(timezone=True), primary_key=True),
    Column("commodity", String, primary_key=True),     # 'gas_ttf' | 'carbon_eua' | 'brent'
    Column("source", String, primary_key=True),          # 'yfinance'
    Column("unit", String),
    Column("value", Float),
)

# Model outputs (Layer 5). One row per target timestamp per quantile per model run.
forecasts = Table(
    "forecasts", metadata,
    Column("target_ts", DateTime(timezone=True), primary_key=True),
    Column("zone", String, primary_key=True),
    Column("target", String, primary_key=True),      # 'dam_price'
    Column("quantile", Float, primary_key=True),      # 0.1 / 0.5 / 0.9 ; -1 for point
    Column("model", String, primary_key=True),
    Column("run_ts", DateTime(timezone=True), primary_key=True),
    Column("value", Float),
    UniqueConstraint("target_ts", "zone", "target", "quantile", "model", "run_ts"),
)

_TABLES = {t.name: t for t in (prices, load, generation, weather, fuel, forecasts)}


# --- engine + schema -----------------------------------------------------------

_engine: Engine | None = None


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = create_engine(settings.database_url, future=True)
    return _engine


def init_schema() -> None:
    metadata.create_all(get_engine())


# --- upsert --------------------------------------------------------------------

def upsert(table_name: str, df: pd.DataFrame) -> int:
    """Insert rows, updating `value` on primary-key conflict. Dialect-aware.

    df columns must match the table's columns. Returns number of rows written.
    """
    if df is None or df.empty:
        return 0
    table = _TABLES[table_name]
    cols = [c.name for c in table.columns]
    records = df[cols].to_dict("records")
    pk = [c.name for c in table.primary_key.columns]
    engine = get_engine()
    dialect = engine.dialect.name

    if dialect == "sqlite":
        from sqlalchemy.dialects.sqlite import insert as _insert
    elif dialect == "postgresql":
        from sqlalchemy.dialects.postgresql import insert as _insert
    else:  # pragma: no cover - fallback for other backends
        with engine.begin() as conn:
            conn.execute(table.insert(), records)
        return len(records)

    # SQLite caps bound variables per statement (999 on older builds). Chunk so
    # rows-per-statement * columns stays well under the limit.
    max_vars = 900
    chunk = max(1, max_vars // max(1, len(cols)))
    with engine.begin() as conn:
        for i in range(0, len(records), chunk):
            batch = records[i:i + chunk]
            stmt = _insert(table).values(batch)
            update_cols = {c: stmt.excluded[c] for c in cols if c not in pk}
            stmt = stmt.on_conflict_do_update(index_elements=pk, set_=update_cols)
            conn.execute(stmt)
    return len(records)


def read_sql(query: str, **kwargs) -> pd.DataFrame:
    """Convenience read into a DataFrame with tz-aware ts parsing left to the caller."""
    return pd.read_sql(query, get_engine(), **kwargs)

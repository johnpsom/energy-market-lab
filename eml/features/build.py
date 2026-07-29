"""Feature-engine orchestrator — assembles every block into one hourly matrix.

Blocks that have no data yet (price/grid before ENTSO-E lands) contribute nothing and are
skipped cleanly, so the same call works today (weather+calendar) and after the token lands
(full matrix). Returns a DataFrame indexed by hourly ts (naive Europe/Athens local time).
"""
from __future__ import annotations

import pandas as pd

from ..config import settings
from . import calendar, grid, price, weather


def build_matrix(zone: str | None = None) -> pd.DataFrame:
    """Assemble the model-ready feature matrix from whatever the warehouse currently holds."""
    zone = zone or settings.default_zone

    weather_f = weather.build()
    price_f = price.build(zone=zone)
    grid_f = grid.build(zone=zone)

    # Master hourly index = union of every block that has data.
    indexes = [b.index for b in (weather_f, price_f, grid_f) if not b.empty]
    if not indexes:
        return pd.DataFrame()
    idx = indexes[0]
    for other in indexes[1:]:
        idx = idx.union(other)
    idx = pd.DatetimeIndex(idx).sort_values()

    blocks = [calendar.build(idx)]                       # calendar always available
    for b in (weather_f, price_f, grid_f):
        if not b.empty:
            blocks.append(b.reindex(idx))

    matrix = pd.concat(blocks, axis=1)
    matrix.index.name = "ts"
    return matrix


def feature_report(matrix: pd.DataFrame) -> pd.DataFrame:
    """Per-feature coverage summary (non-null %, min/mean/max) — quick sanity/QA view."""
    n = len(matrix)
    rep = pd.DataFrame({
        "non_null_pct": (matrix.notna().sum() / n * 100).round(1),
        "min": matrix.min(numeric_only=True).round(2),
        "mean": matrix.mean(numeric_only=True).round(2),
        "max": matrix.max(numeric_only=True).round(2),
    })
    return rep

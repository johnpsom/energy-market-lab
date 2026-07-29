"""Environment-driven settings. Loaded once, imported everywhere."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Project root = parent of the `eml` package.
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    entsoe_token: str = os.getenv("ENTSOE_TOKEN", "").strip()
    database_url: str = os.getenv("DATABASE_URL", f"sqlite:///{DATA_DIR / 'eml.db'}")
    default_zone: str = os.getenv("DEFAULT_ZONE", "GR").strip()
    timezone: str = "Europe/Athens"  # HEnEx / ADMIE local time

    @property
    def has_entsoe(self) -> bool:
        return bool(self.entsoe_token)


settings = Settings()

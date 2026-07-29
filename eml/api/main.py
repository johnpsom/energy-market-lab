"""FastAPI dashboard — the decision-support Home workspace.

Serves the day-ahead price forecast (P10/P50/P90 band), the SHAP driver attribution, spike /
negative-price risk, the natural-language brief, and the model's backtest scorecard. This is
the first of the planned workspaces (Home); Market / Forecasts / Risk follow later.
"""
from __future__ import annotations

import datetime as _dt
import json
from functools import lru_cache

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates

from ..config import ROOT
from ..models import verify as verify_mod
from ..models.price_forecast import ARTIFACTS
from ..narrative import brief as brief_mod

app = FastAPI(title="Energy Market Lab")
templates = Jinja2Templates(directory=str(ROOT / "eml" / "api" / "templates"))


@lru_cache(maxsize=16)
def _brief_cached(date: str | None, day_key: str):
    return brief_mod.generate(date)


def _brief(date: str | None = None):
    # day_key busts the cache at date rollover so a None (=tomorrow) brief stays current.
    return _brief_cached(date, _dt.date.today().isoformat())


@lru_cache(maxsize=4)
def _outlook_cached(n_days: int, day_key: str):
    return brief_mod.outlook(n_days)


def _outlook(n_days: int = 7):
    return _outlook_cached(n_days, _dt.date.today().isoformat())


@lru_cache(maxsize=2)
def _verification_cached(day_key: str):
    return verify_mod.scorecard()


def _verification():
    return _verification_cached(_dt.date.today().isoformat())


def _metrics() -> dict:
    p = ARTIFACTS / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else {}


@app.get("/api/brief")
def api_brief(date: str | None = None):
    return JSONResponse(_brief(date))


@app.get("/api/metrics")
def api_metrics():
    return JSONResponse(_metrics())


@app.get("/api/outlook")
def api_outlook(days: int = 7):
    return JSONResponse(_outlook(days))


@app.get("/api/verification")
def api_verification():
    return JSONResponse(_verification())


@app.get("/")
def home(request: Request, date: str | None = None):
    s = _brief(date)
    ol = _outlook(7)
    ver = _verification()
    return templates.TemplateResponse(request, "home.html", {
        "s": s,
        "metrics": _metrics(),
        "outlook": ol,
        "verification": ver,
        "hourly_json": json.dumps(s["hourly"]),
        "drivers_json": json.dumps(s["drivers"]),
        "outlook_json": json.dumps(ol),
        "verification_json": json.dumps(ver),
    })

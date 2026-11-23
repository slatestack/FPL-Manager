"""Live xP run: load data, predict the next ``horizon`` gameweeks, persist with full audit info."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import asyncpg
import numpy as np
import pandas as pd

from app.ml import MODEL_VERSION
from app.ml import data as data_mod
from app.ml import engine
from app.ml.availability import availability
from app.ml.minutes import anchor_features
from app.ml.params import ModelParams

log = logging.getLogger(__name__)


@dataclass
class PredictionSummary:
    run_timestamp: datetime
    base_gameweek: int
    gameweeks: list[int]
    players: int
    seconds: float
    top: list[dict[str, Any]]


async def _upcoming_gameweeks(conn: asyncpg.Connection, now: datetime, horizon: int) -> dict[int, datetime]:
    rows = await conn.fetch(
        "SELECT gameweek_id, deadline_time FROM gameweeks WHERE deadline_time > $1 ORDER BY deadline_time LIMIT $2", now, horizon
    )
    return {r["gameweek_id"]: r["deadline_time"] for r in rows}


async def _inputs_fingerprint(conn: asyncpg.Connection) -> dict[str, Any]:
    snap = await conn.fetch(
        """SELECT DISTINCT ON (source) source, payload_hash, last_seen_at FROM raw_snapshots
           ORDER BY source, last_seen_at DESC"""
    )
    odds = await conn.fetchval("SELECT MAX(captured_at) FROM fixture_odds")
    hist = await conn.fetchval("SELECT MAX(updated_at) FROM player_gw_history")
    archive = await conn.fetch("SELECT season, COUNT(*) AS n FROM archive_player_matches GROUP BY season ORDER BY season")
    return {
        "snapshots": {r["source"]: r["payload_hash"] for r in snap},
        "odds_captured_at": odds.isoformat() if odds else None,
        "history_updated_at": hist.isoformat() if hist else None,
        "archive_rows": {r["season"]: r["n"] for r in archive},
    }


def compute(
    data: data_mod.ModelData, players: pd.DataFrame, targets: pd.DataFrame, deadlines: dict[int, datetime],
    odds: pd.DataFrame, now: datetime, params: ModelParams, scorer_odds: pd.DataFrame | None = None,
) -> engine.EngineResult:
    """Pure computation of one live run (no I/O)."""
    as_of = pd.Timestamp(now)
    data = data.before(as_of)
    avail = availability(players, deadlines, now).rename(columns={"player_id": "key"})
    anchors = anchor_features(data.matches)
    models = engine.train_models(data, anchors, as_of, params)
    return engine.predict(data, players, targets, as_of, params, models, anchors, avail=avail, odds=odds,
                          gameweeks=sorted(deadlines), scorer_odds=scorer_odds)


async def run_prediction(pool: asyncpg.Pool, params: ModelParams | None = None, now: datetime | None = None) -> PredictionSummary:
    params = params or ModelParams()
    now = now or datetime.now(timezone.utc)
    t0 = time.monotonic()
    async with pool.acquire() as conn:
        data = await data_mod.load_model_data(conn)
        players = await data_mod.load_current_players(conn)
        deadlines = await _upcoming_gameweeks(conn, now, params.horizon)
        if not deadlines:
            raise RuntimeError("no upcoming gameweeks to predict")
        gws = sorted(deadlines)
        targets = await data_mod.load_target_fixtures(conn, gws, data.current_season)
        odds = await data_mod.load_latest_odds(conn, now)
        scorer_odds = await data_mod.load_latest_scorer_odds(conn, now, data.current_season)
        fingerprint = await _inputs_fingerprint(conn)

    players = players.assign(key=players["player_id"])
    # CPU-bound (model training + simulation): keep the event loop free.
    res = await asyncio.to_thread(compute, data, players, targets, deadlines, odds, now, params, scorer_odds)

    run_ts = now.replace(microsecond=(now.microsecond // 1000) * 1000)
    gw = res.by_gameweek
    seconds = time.monotonic() - t0
    diagnostics = {**res.diagnostics, "seconds": round(seconds, 2), "fixtures_with_odds": int((res.fixtures["odds_w"] > 0).sum())}
    async with pool.acquire() as conn, conn.transaction():
        await conn.execute(
            """INSERT INTO prediction_runs (run_timestamp, model_version, base_gameweek, horizon, params, inputs, diagnostics)
               VALUES ($1, $2, $3, $4, $5, $6, $7)""",
            run_ts, MODEL_VERSION, gws[0], len(gws), params.as_dict(), fingerprint, _jsonable(diagnostics),
        )
        await conn.copy_records_to_table(
            "player_predictions",
            records=[
                (
                    int(r.key), int(r.gameweek), run_ts, _r(r.prob_play, 3), _r(r.xp, 2), _r(r.xp_attacking, 2),
                    _r(r.xp_defensive, 2), _r(r.xp_bonus, 2), int(r.n_fixtures), _r(r.p_start, 3),
                    _r(r.expected_minutes, 1), _r(r.expected_goals, 3), _r(r.expected_assists, 3),
                    _r(r.p_clean_sheet, 3), _r(r.xp_appearance, 2), _r(r.xp_deductions, 2),
                )
                for r in gw.itertuples()
            ],
            columns=["player_id", "target_gw", "run_timestamp", "prob_play", "xp_total", "xp_attacking", "xp_defensive",
                     "xp_bonus", "n_fixtures", "p_start", "expected_minutes", "expected_goals", "expected_assists",
                     "p_clean_sheet", "xp_appearance", "xp_deductions"],
        )
        fx = res.fixtures
        await conn.copy_records_to_table(
            "fixture_predictions",
            records=[
                (run_ts, int(r.fixture_id), _r(r.lh, 3), _r(r.la, 3), _r(r.lh_model, 3), _r(r.la_model, 3), _r(r.odds_w, 3))
                for r in fx.itertuples()
            ],
            columns=["run_timestamp", "fixture_id", "lambda_home", "lambda_away", "lambda_home_model", "lambda_away_model", "odds_weight"],
        )

    first = gw[gw["gameweek"] == gws[0]].merge(players[["key", "web_name", "position"]], on="key")
    top = first.nlargest(10, "xp")[["web_name", "position", "xp", "prob_play"]].round(2).to_dict("records")
    log.info("xP run %s: %d players x %d GWs in %.1fs", run_ts, players.shape[0], len(gws), seconds)
    return PredictionSummary(run_ts, gws[0], gws, len(players), round(seconds, 1), top)


def _r(v: Any, nd: int) -> Any:
    from decimal import Decimal

    v = float(v)
    if np.isnan(v):
        return None
    return Decimal(str(round(v, nd)))


def _jsonable(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o

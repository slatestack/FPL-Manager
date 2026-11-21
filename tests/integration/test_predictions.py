from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest
import respx

from app.ingestion import pipeline
from app.ingestion.adapters.archive import FPLArchiveAdapter
from app.ingestion.http import ResilientClient
from app.ml.service import run_prediction

FIX = Path(__file__).parents[1] / "fixtures"
NOW = datetime(2026, 11, 1, tzinfo=timezone.utc)


async def _ingest_archive(pool):
    with respx.mock(base_url="https://archive.test/data/2025-26") as router:
        for name in ("teams", "players_raw", "fixtures"):
            router.get(f"/{name}.csv").mock(return_value=httpx.Response(200, text=(FIX / f"archive_2025-26_{name}.csv").read_text()))
        router.get("/gws/merged_gw.csv").mock(
            return_value=httpx.Response(200, text=(FIX / "archive_2025-26_merged_gw.csv").read_text())
        )
        async with ResilientClient(max_retries=0) as http:
            return await pipeline.ingest_archive(pool, FPLArchiveAdapter(http, base_url="https://archive.test/data"), ["2025-26"])


def _add_double_and_blank(state):
    """GW4: Arsenal play twice, Wolves blank. GW5: nobody plays."""
    state["bootstrap"]["events"] += [
        {"id": 4, "name": "Gameweek 4", "deadline_time": "2026-11-28T11:00:00Z", "finished": False, "data_checked": False},
        {"id": 5, "name": "Gameweek 5", "deadline_time": "2026-12-05T11:00:00Z", "finished": False, "data_checked": False},
    ]
    template = dict(state["fixtures"][4])
    state["fixtures"] += [
        {**template, "id": 8, "code": 2500008, "event": 4, "team_h": 1, "team_a": 2, "kickoff_time": "2026-11-28T15:00:00Z"},
        {**template, "id": 9, "code": 2500009, "event": 4, "team_h": 3, "team_a": 1, "kickoff_time": "2026-12-01T19:45:00Z"},
    ]


async def test_archive_ingestion(pool):
    result = await _ingest_archive(pool)
    assert result.details["2025-26"]["matches"] == 529
    assert await pool.fetchval("SELECT COUNT(*) FROM archive_player_matches") == 529
    assert await pool.fetchval("SELECT COUNT(DISTINCT position) FROM archive_players") == 4


async def test_prediction_run_end_to_end(pool, fpl, fpl_api):
    _add_double_and_blank(fpl_api)
    await pipeline.ingest_core(pool, fpl)
    await pipeline.ingest_player_histories(pool, fpl)
    await _ingest_archive(pool)

    summary = await run_prediction(pool, now=NOW)
    assert summary.gameweeks == [3, 4, 5]

    rows = {(r["player_id"], r["target_gw"]): r for r in await pool.fetch("SELECT * FROM latest_player_predictions")}
    assert len(rows) == 10 * 3, "every player gets a row for every gameweek, blanks included"

    # Blank gameweeks are exactly zero.
    assert all(rows[(pid, 5)]["xp_total"] == 0 and rows[(pid, 5)]["n_fixtures"] == 0 for pid in range(1, 11))
    assert rows[(9, 4)]["xp_total"] == 0, "Wolves blank in GW4"

    # Double gameweek: two fixtures, roughly double the single-fixture projection.
    gab3, gab4 = rows[(2, 3)], rows[(2, 4)]
    assert gab4["n_fixtures"] == 2 and gab3["n_fixtures"] == 1
    assert float(gab4["xp_total"]) > 1.5 * float(gab3["xp_total"]) > 0

    # Injured with a return date after the horizon -> zero; doubtful -> discounted, not zero.
    assert all(rows[(10, gw)]["xp_total"] == 0 for gw in (3, 4, 5))
    saka = rows[(3, 3)]
    assert 0.5 < float(saka["prob_play"]) < 0.8

    # Component columns add up to the total.
    for r in rows.values():
        parts = sum(float(r[c]) for c in ("xp_appearance", "xp_attacking", "xp_defensive", "xp_bonus", "xp_deductions"))
        assert parts == pytest.approx(float(r["xp_total"]), abs=0.03)
        assert 0 <= float(r["prob_play"]) <= 1

    run = await pool.fetchrow("SELECT * FROM prediction_runs")
    assert run["base_gameweek"] == 3 and run["params"]["horizon"] == 5 and "snapshots" in run["inputs"]
    assert await pool.fetchval("SELECT COUNT(*) FROM fixture_predictions") == 4


async def test_predictions_are_deterministic(pool, fpl):
    await pipeline.ingest_core(pool, fpl)
    await pipeline.ingest_player_histories(pool, fpl)
    await _ingest_archive(pool)
    a = await run_prediction(pool, now=NOW)
    b = await run_prediction(pool, now=NOW.replace(minute=1))
    q = "SELECT player_id, target_gw, xp_total, xp_bonus FROM player_predictions WHERE run_timestamp=$1 ORDER BY 1, 2"
    first = [tuple(r) for r in await pool.fetch(q, a.run_timestamp)]
    second = [tuple(r) for r in await pool.fetch(q, b.run_timestamp)]
    assert first == second and len(first) > 0

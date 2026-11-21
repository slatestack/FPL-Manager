"""FastAPI control plane. Phase 1 exposes health and ingestion status only;
the Section 9 endpoints arrive in Phase 6."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.ingestion.fpl_client import FPLClient
from redis.asyncio import Redis

from app.config import get_settings
from app.db.pool import create_pool


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    app.state.pool = await create_pool(settings.database_url)
    app.state.redis = Redis.from_url(settings.redis_url)
    try:
        yield
    finally:
        await app.state.redis.aclose()
        await app.state.pool.close()


app = FastAPI(title="FPL Manager Control Plane", version="0.1.0", lifespan=lifespan)

STATIC = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/", include_in_schema=False)
async def dashboard_page() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/healthz")
async def healthz(response: Response) -> dict[str, str]:
    checks: dict[str, str] = {}
    try:
        await app.state.pool.fetchval("SELECT 1")
        checks["postgres"] = "ok"
    except Exception as exc:
        checks["postgres"] = f"error: {type(exc).__name__}"
    try:
        await app.state.redis.ping()
        checks["redis"] = "ok"
    except Exception as exc:
        checks["redis"] = f"error: {type(exc).__name__}"
    if any(v != "ok" for v in checks.values()):
        response.status_code = 503
    return checks


@app.get("/api/v1/ingestion/status")
async def ingestion_status() -> dict[str, object]:
    rows = await app.state.pool.fetch(
        """SELECT DISTINCT ON (job) job, status, started_at, finished_at, rows_written, error_message
           FROM ingestion_runs ORDER BY job, started_at DESC"""
    )
    next_gw = await app.state.pool.fetchrow(
        "SELECT gameweek_id, deadline_time FROM gameweeks WHERE deadline_time > NOW() ORDER BY deadline_time LIMIT 1"
    )
    return {
        "next_gameweek": dict(next_gw) if next_gw else None,
        "jobs": {r["job"]: {k: r[k] for k in r.keys() if k != "job"} for r in rows},
    }


@app.get("/api/v1/predictions/latest")
async def latest_predictions(
    gameweek: int | None = None,
    position: str | None = Query(default=None, pattern="^(GKP|DEF|MID|FWD)$"),
    limit: int = Query(default=50, ge=1, le=1000),
) -> dict[str, object]:
    run = await app.state.pool.fetchrow(
        "SELECT run_timestamp, model_version, base_gameweek, horizon FROM prediction_runs ORDER BY run_timestamp DESC LIMIT 1"
    )
    if run is None:
        raise HTTPException(404, "no prediction runs yet")
    gw = gameweek or run["base_gameweek"]
    rows = await app.state.pool.fetch(
        """SELECT pp.player_id, p.web_name, t.short_name AS team, p.position, p.now_cost, pp.target_gw,
                  pp.xp_total, pp.prob_play, pp.p_start, pp.expected_minutes, pp.expected_goals, pp.expected_assists,
                  pp.p_clean_sheet, pp.xp_appearance, pp.xp_attacking, pp.xp_defensive, pp.xp_bonus, pp.xp_deductions,
                  pp.n_fixtures
           FROM player_predictions pp JOIN players p USING (player_id) JOIN teams t ON t.team_id = p.team_id
           WHERE pp.run_timestamp = $1 AND pp.target_gw = $2 AND ($3::text IS NULL OR p.position = $3)
           ORDER BY pp.xp_total DESC LIMIT $4""",
        run["run_timestamp"], gw, position, limit,
    )
    return {"run": dict(run), "gameweek": gw, "players": [dict(r) for r in rows]}


class DryRunRequest(BaseModel):
    team_id: int | None = None
    free_transfers: int | None = None


@app.post("/api/v1/optimize/dry-run")
async def optimize_dry_run(req: DryRunRequest) -> dict[str, object]:
    """Section 9: run the MILP now against the latest xP; nothing is submitted to FPL."""
    from app.optimizer.service import run_optimization

    settings = get_settings()
    team_id = req.team_id or settings.team_id
    if not team_id:
        raise HTTPException(400, "team_id required (or set FPL_TEAM_ID)")
    async with FPLClient(settings) as fpl:
        r = await run_optimization(app.state.pool, fpl, team_id, free_transfers=req.free_transfers)
    return await _optimization_view(r.run_id)


@app.get("/api/v1/optimize/latest")
async def optimize_latest(team_id: int | None = None) -> dict[str, object]:
    team_id = team_id or get_settings().team_id
    run_id = await app.state.pool.fetchval(
        "SELECT id FROM optimization_runs WHERE team_id = $1 ORDER BY created_at DESC LIMIT 1", team_id
    )
    if run_id is None:
        raise HTTPException(404, "no optimisation runs yet")
    return await _optimization_view(str(run_id))


@app.get("/api/v1/dashboard")
async def dashboard(team_id: int | None = None) -> dict[str, object]:
    """Everything the web dashboard renders, in one call."""
    pool = app.state.pool
    team_id = team_id or get_settings().team_id
    team = await pool.fetchrow("SELECT * FROM manager_entries WHERE entry_id = $1", team_id) if team_id else None
    next_gw = await pool.fetchrow(
        "SELECT gameweek_id, name, deadline_time FROM gameweeks WHERE deadline_time > NOW() ORDER BY deadline_time LIMIT 1"
    )
    jobs = await pool.fetch(
        """SELECT DISTINCT ON (job) job, status, started_at, finished_at, error_message
           FROM ingestion_runs ORDER BY job, started_at DESC"""
    )
    run = await pool.fetchrow("SELECT run_timestamp, base_gameweek, horizon, model_version FROM prediction_runs ORDER BY run_timestamp DESC LIMIT 1")
    players: list[dict[str, object]] = []
    gameweeks: list[int] = []
    if run:
        preds = await pool.fetch(
            """SELECT pp.player_id, pp.target_gw, pp.xp_total::float AS xp, pp.prob_play::float AS p_play
               FROM player_predictions pp WHERE pp.run_timestamp = $1""",
            run["run_timestamp"],
        )
        gameweeks = sorted({r["target_gw"] for r in preds})
        opp = await pool.fetch(
            """SELECT f.gameweek_id AS gw, t.team_id, o.short_name AS opp, (f.team_h = t.team_id) AS home, f.kickoff_time
               FROM fixtures f JOIN teams t ON t.team_id IN (f.team_h, f.team_a)
               JOIN teams o ON o.team_id = CASE WHEN f.team_h = t.team_id THEN f.team_a ELSE f.team_h END
               WHERE f.gameweek_id = ANY($1::int[]) ORDER BY f.kickoff_time""",
            gameweeks,
        )
        opponents: dict[tuple[int, int], list[str]] = {}
        for r in opp:
            opponents.setdefault((r["team_id"], r["gw"]), []).append(f"{r['opp']} ({'H' if r['home'] else 'A'})")
        info = {
            r["player_id"]: dict(r)
            for r in await pool.fetch(
                """SELECT p.player_id, p.web_name AS name, p.team_id, t.short_name AS team, p.position, p.now_cost AS price,
                          p.status, p.news, p.selected_by_percent::float AS owned
                   FROM players p JOIN teams t ON t.team_id = p.team_id"""
            )
        }
        by_player: dict[int, dict[str, object]] = {}
        for r in preds:
            pl = by_player.setdefault(r["player_id"], {**info[r["player_id"]], "xp": {}, "p_play": {}, "opp": {}})
            pl["xp"][r["target_gw"]] = round(r["xp"], 2)
            pl["p_play"][r["target_gw"]] = round(r["p_play"], 3)
            pl["opp"][r["target_gw"]] = opponents.get((pl["team_id"], r["target_gw"]), [])
        players = list(by_player.values())
    opt_id = await pool.fetchval(
        "SELECT id FROM optimization_runs WHERE team_id = $1 ORDER BY created_at DESC LIMIT 1", team_id
    ) if team_id else None
    return {
        "team": dict(team) if team else None,
        "next_gameweek": dict(next_gw) if next_gw else None,
        "jobs": [dict(r) for r in jobs],
        "prediction_run": dict(run) if run else None,
        "gameweeks": gameweeks,
        "players": players,
        "optimization": await _optimization_view(str(opt_id)) if opt_id else None,
    }


async def _optimization_view(run_id: str) -> dict[str, object]:
    row = await app.state.pool.fetchrow("SELECT * FROM optimization_runs WHERE id = $1::uuid", run_id)
    names = {r["player_id"]: r["web_name"] for r in await app.state.pool.fetch("SELECT player_id, web_name FROM players")}
    plan = row["plan"]
    for g in plan:
        for k in ("transfers_in", "transfers_out", "starting_xi", "bench", "squad"):
            g[f"{k}_names"] = [names.get(p) for p in g[k]]
        g["captain_name"], g["vice_captain_name"] = names.get(g["captain"]), names.get(g["vice_captain"])
    return {
        "run_id": str(row["id"]), "created_at": row["created_at"], "team_id": row["team_id"],
        "gameweek": row["base_gameweek"], "objective": float(row["objective"]),
        "baseline_objective": float(row["baseline_objective"]), "chip_recommended": row["chip_recommended"],
        "chip_evaluation": row["chip_evaluation"], "plan": plan, "solver": row["solver"],
        "team_state": {k: row["inputs"]["team_state"][k] for k in ("squad", "bank", "free_transfers", "chips_available", "warnings")},
    }

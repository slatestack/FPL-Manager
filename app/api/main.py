"""FastAPI control plane. Phase 1 exposes health and ingestion status only;
the Section 9 endpoints arrive in Phase 6."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Response
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
        "team_state": {k: row["inputs"]["team_state"][k] for k in ("bank", "free_transfers", "chips_available", "warnings")},
    }

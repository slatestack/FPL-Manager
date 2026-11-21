"""FastAPI control plane. Phase 1 exposes health and ingestion status only;
the Section 9 endpoints arrive in Phase 6."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from pathlib import Path

import hmac
from datetime import datetime, timezone
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Response
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
        "control": await status(team_id) if team_id else None,
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


# ------------------------------------------------------------------ Section 9 control plane


def require_token(authorization: str | None = Header(default=None)) -> None:
    """Mutating endpoints need ``Authorization: Bearer $FPL_API_TOKEN``; they are disabled if no token is configured."""
    expected = get_settings().api_token
    if not expected:
        raise HTTPException(403, "control endpoints are disabled: set FPL_API_TOKEN")
    supplied = (authorization or "").removeprefix("Bearer ").strip()
    if not hmac.compare_digest(supplied.encode(), expected.encode()):
        raise HTTPException(401, "invalid API token")


@app.get("/api/v1/status/{team_id}")
async def status(team_id: int) -> dict[str, object]:
    pool = app.state.pool
    now = datetime.now(timezone.utc)
    gw = await pool.fetchrow(
        "SELECT gameweek_id, deadline_time FROM gameweeks WHERE deadline_time > $1 ORDER BY deadline_time LIMIT 1", now)
    cfg = await pool.fetchrow(
        """SELECT auto_execute_enabled, max_hits_per_gw, chip_aggression_profile, encrypted_auth_cookie IS NOT NULL AS has_tokens,
                  auth_expires_at, auth_last_check_at, auth_last_check_ok, auth_last_error
           FROM system_config WHERE team_id = $1""", team_id)
    opt = await pool.fetchrow(
        "SELECT inputs, plan FROM optimization_runs WHERE team_id = $1 ORDER BY created_at DESC LIMIT 1", team_id)
    squad: list[dict[str, object]] = []
    if opt:
        ids = opt["inputs"]["team_state"]["squad"]
        rows = await pool.fetch(
            """SELECT p.player_id, p.web_name, p.position, t.short_name AS team, pp.xp_total::float AS xp_next
               FROM players p JOIN teams t ON t.team_id = p.team_id
               LEFT JOIN latest_player_predictions pp ON pp.player_id = p.player_id AND pp.target_gw = $2
               WHERE p.player_id = ANY($1::int[])""", ids, gw["gameweek_id"] if gw else None)
        squad = [dict(r) for r in rows]
    last = await pool.fetchrow(
        """SELECT id, gameweek, execution_type, status, error_message, created_at FROM execution_logs
           WHERE team_id = $1 ORDER BY created_at DESC LIMIT 1""", team_id)
    actions = await pool.fetch(
        "SELECT action, status, finished_at FROM scheduled_actions WHERE team_id=$1 AND gameweek=$2",
        team_id, gw["gameweek_id"] if gw else None)
    return {
        "next_gameweek": gw["gameweek_id"] if gw else None,
        "deadline": gw["deadline_time"] if gw else None,
        "seconds_to_deadline": int((gw["deadline_time"] - now).total_seconds()) if gw else None,
        "config": ({k: cfg[k] for k in ("auto_execute_enabled", "max_hits_per_gw", "chip_aggression_profile")} if cfg
                   else {"auto_execute_enabled": False, "max_hits_per_gw": 1, "chip_aggression_profile": "BALANCED"}),
        "auth": ({k: cfg[k] for k in ("has_tokens", "auth_expires_at", "auth_last_check_at", "auth_last_check_ok", "auth_last_error")}
                 if cfg else {"has_tokens": False}),
        "squad": squad,
        "last_execution": dict(last) if last else None,
        "deadline_actions": {r["action"]: {"status": r["status"], "finished_at": r["finished_at"]} for r in actions},
    }


class ConfigPatch(BaseModel):
    auto_execute_enabled: bool | None = None
    max_hits_per_gw: int | None = None
    chip_aggression_profile: Literal["CONSERVATIVE", "BALANCED", "AGGRESSIVE"] | None = None
    auth_tokens: str | None = None   # the browser's oidc.user JSON (stored encrypted)


@app.patch("/api/v1/config/{team_id}", dependencies=[Depends(require_token)])
async def patch_config(team_id: int, body: ConfigPatch) -> dict[str, object]:
    from app.executor.vault import AuthError, TokenVault, parse_token_input

    pool = app.state.pool
    await pool.execute(
        "INSERT INTO system_config (team_id, auto_execute_enabled) VALUES ($1, FALSE) ON CONFLICT (team_id) DO NOTHING", team_id)
    if body.auth_tokens is not None:
        try:
            await TokenVault(get_settings(), pool).store(team_id, parse_token_input(body.auth_tokens))
        except (AuthError, ValueError) as exc:
            raise HTTPException(400, f"tokens not accepted: {exc}") from exc
    if body.auto_execute_enabled:
        if body.auth_tokens is not None:  # newly imported tokens: verify them against FPL before arming
            from app.executor.service import Executor
            from app.notify import LogNotifier

            await Executor(get_settings(), pool, app.state.redis, LogNotifier()).keepalive(team_id)
        verified = await pool.fetchval(
            "SELECT encrypted_auth_cookie IS NOT NULL AND auth_last_check_ok IS TRUE FROM system_config WHERE team_id=$1", team_id)
        if not verified:
            raise HTTPException(409, "FPL login must pass a live check before auto-execution can be enabled")
    if body.max_hits_per_gw is not None and not 0 <= body.max_hits_per_gw <= 5:
        raise HTTPException(422, "max_hits_per_gw must be between 0 and 5")
    for col in ("auto_execute_enabled", "max_hits_per_gw", "chip_aggression_profile"):
        value = getattr(body, col)
        if value is not None:
            await pool.execute(f"UPDATE system_config SET {col} = $2, updated_at = NOW() WHERE team_id = $1", team_id, value)
    return (await status(team_id))["config"]


@app.post("/api/v1/execute/force", dependencies=[Depends(require_token)])
async def execute_force(team_id: int | None = None) -> dict[str, object]:
    """Section 9: run the commit pipeline now. Bypasses the timer, not the guardrails (kill switch included)."""
    from app.executor.service import Executor
    from app.notify import build_notifier

    settings = get_settings()
    team_id = team_id or settings.team_id
    if not team_id:
        raise HTTPException(400, "team_id required")
    if not settings.secret_key:
        raise HTTPException(409, "FPL_SECRET_KEY is not configured")
    result = await Executor(settings, app.state.pool, app.state.redis, build_notifier(settings)).commit(team_id, trigger="api_force")
    return {"status": result.status, "reason": result.reason, "execution_log_id": result.execution_log_id,
            "gameweek": result.gameweek, "guardrails": result.report}


@app.get("/api/v1/audit/{team_id}")
async def audit(team_id: int, limit: int = Query(default=50, ge=1, le=500)) -> dict[str, object]:
    """Execution history with projected (plan) vs actual gameweek points."""
    rows = await app.state.pool.fetch(
        """SELECT e.id, e.created_at, e.gameweek, e.execution_type, e.status, e.trigger, e.chip_activated,
                  e.transfers_in, e.transfers_out, e.captain_id, e.projected_xp_gain::float AS projected_xp_gain,
                  e.error_message, (o.plan -> 0 ->> 'xp')::float AS projected_points, h.points AS actual_points,
                  h.event_transfers_cost AS hit_points
           FROM execution_logs e
           LEFT JOIN optimization_runs o ON o.execution_log_id = e.id
           LEFT JOIN manager_history h ON h.entry_id = e.team_id AND h.gameweek = e.gameweek
           WHERE e.team_id = $1 ORDER BY e.created_at DESC LIMIT $2""", team_id, limit)
    out = []
    for r in rows:
        d = dict(r)
        if d["projected_points"] is not None and d["actual_points"] is not None:
            d["variance"] = round(d["actual_points"] - d["projected_points"], 2)
        out.append(d)
    return {"team_id": team_id, "executions": out}

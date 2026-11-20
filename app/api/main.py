"""FastAPI control plane. Phase 1 exposes health and ingestion status only;
the Section 9 endpoints arrive in Phase 6."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response
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

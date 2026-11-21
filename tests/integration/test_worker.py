import os
from datetime import datetime, timedelta, timezone

import pytest
from redis.asyncio import Redis

from app.ingestion.worker import IngestionWorker

TEST_REDIS = os.environ.get("TEST_REDIS_URL")
pytestmark = pytest.mark.skipif(not TEST_REDIS, reason="TEST_REDIS_URL not set")


@pytest.fixture
async def redis():
    r = Redis.from_url(TEST_REDIS)
    await r.flushdb()
    yield r
    await r.flushdb()
    await r.aclose()


async def test_tick_runs_due_jobs_and_schedules_by_deadline(pool, redis, settings, fpl, fpl_api):
    # Put the next deadline 20h out -> pre-deadline window -> 30 minute cadence.
    # (Job due-ness compares against DB NOW(), so the worker uses the real clock here.)
    deadline = datetime.now(timezone.utc).replace(microsecond=0) + timedelta(hours=20)
    gw3 = next(e for e in fpl_api["bootstrap"]["events"] if e["id"] == 3)
    gw3["deadline_time"] = deadline.isoformat().replace("+00:00", "Z")
    worker = IngestionWorker(settings, pool, redis, fpl, predict_enabled=False)
    delay = await worker.tick()
    assert timedelta(minutes=29) < delay <= timedelta(minutes=30)

    jobs = {r["job"]: r["status"] for r in await pool.fetch("SELECT job, status FROM ingestion_runs")}
    assert jobs == {"core": "SUCCESS", "history": "SUCCESS", "entry": "SUCCESS", "prune": "SUCCESS"}

    # Second tick: history/entry are not due again yet.
    await worker.tick()
    counts = {r["job"]: r["n"] for r in await pool.fetch("SELECT job, COUNT(*) n FROM ingestion_runs GROUP BY job")}
    assert counts == {"core": 2, "history": 1, "entry": 1, "prune": 1}


async def test_lock_prevents_overlapping_runs(pool, redis, settings, fpl):
    worker = IngestionWorker(settings, pool, redis, fpl, predict_enabled=False)
    held = redis.lock("fpl_ingest_lock:core", timeout=30)
    assert await held.acquire(blocking=False)
    try:
        await worker.tick()
    finally:
        await held.release()
    assert await pool.fetchval("SELECT status FROM ingestion_runs WHERE job='core'") == "SKIPPED"

"""Long-running ingestion worker driven by the deadline-aware schedule.

Every tick runs core ingestion (bootstrap-static + fixtures). Slower-moving jobs run when
due, judged from the last successful run recorded in ``ingestion_runs`` so restarts
don't cause stampedes. Each job holds a Redis lock so concurrent workers can't overlap.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta, timezone

import asyncpg
from redis.asyncio import Redis

from app.config import Settings
from app.ingestion import pipeline
from app.ingestion.adapters.archive import FPLArchiveAdapter
from app.ingestion.adapters.base import OddsAdapter, XGAdapter
from app.ingestion.fpl_client import FPLClient
from app.ingestion.http import UpstreamError
from app.ingestion.schedule import Phase, next_poll_delay, polling_phase

log = logging.getLogger(__name__)

HISTORY_INTERVAL = timedelta(hours=12)
XG_INTERVAL = timedelta(hours=12)
ENTRY_INTERVAL = timedelta(hours=6)
PRUNE_INTERVAL = timedelta(hours=24)
ARCHIVE_INTERVAL = timedelta(days=7)
ODDS_INTERVAL = {Phase.STANDARD: timedelta(hours=24), Phase.PRE_DEADLINE: timedelta(hours=6)}
PREDICT_INTERVAL = {  # xP refresh cadence by deadline window
    Phase.STANDARD: timedelta(hours=6),
    Phase.PRE_DEADLINE: timedelta(minutes=30),
    Phase.CRITICAL: timedelta(minutes=10),
}


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class IngestionWorker:
    def __init__(
        self,
        settings: Settings,
        pool: asyncpg.Pool,
        redis: Redis,
        fpl: FPLClient,
        *,
        xg_adapter: XGAdapter | None = None,
        odds_adapter: OddsAdapter | None = None,
        archive_adapter: FPLArchiveAdapter | None = None,
        predict_enabled: bool = True,
        scheduler=None,
        executor=None,
        clock: Callable[[], datetime] = utcnow,
    ):
        self.settings = settings
        self.pool = pool
        self.redis = redis
        self.fpl = fpl
        self.xg_adapter = xg_adapter
        self.odds_adapter = odds_adapter
        self.archive_adapter = archive_adapter
        self.predict_enabled = predict_enabled
        self.scheduler = scheduler
        self.executor = executor
        self.clock = clock
        self._stop = asyncio.Event()

    # ------------------------------------------------------------------ helpers
    async def next_deadline(self) -> datetime | None:
        return await self.pool.fetchval(
            "SELECT MIN(deadline_time) FROM gameweeks WHERE deadline_time > $1", self.clock()
        )

    async def last_success(self, job: str) -> datetime | None:
        return await self.pool.fetchval(
            "SELECT MAX(finished_at) FROM ingestion_runs WHERE job = $1 AND status = 'SUCCESS'", job
        )

    async def _due(self, job: str, interval: timedelta) -> bool:
        last = await self.last_success(job)
        return last is None or self.clock() - last >= interval

    async def _history_due(self) -> bool:
        last = await self.last_success("history")
        if last is None or self.clock() - last >= HISTORY_INTERVAL:
            return True
        # Also refresh as soon as FPL marks a gameweek's data as checked (final bonus etc.).
        changed = await self.pool.fetchval(
            "SELECT EXISTS (SELECT 1 FROM gameweeks WHERE data_checked AND updated_at > $1)", last
        )
        return bool(changed)

    async def run_locked(self, job: str, fn: Callable[[], Awaitable[pipeline.JobResult]]) -> pipeline.JobResult | None:
        lock = self.redis.lock(f"fpl_ingest_lock:{job}", timeout=self.settings.ingest_lock_timeout_seconds)
        if not await lock.acquire(blocking=False):
            log.info("job %s already running elsewhere; skipping", job)
            await pipeline.record_skipped(self.pool, job, "lock held")
            return None
        try:
            return await pipeline.run_job(self.pool, job, fn)
        finally:
            try:
                await lock.release()
            except Exception:  # lock expired - nothing to release
                log.warning("lock for %s expired before release", job)

    async def _predict(self) -> pipeline.JobResult:
        from app.ml.service import run_prediction

        s = await run_prediction(self.pool, now=self.clock())
        return pipeline.JobResult(
            rows_written=s.players * len(s.gameweeks),
            details={"run_timestamp": s.run_timestamp.isoformat(), "gameweeks": s.gameweeks, "seconds": s.seconds},
        )

    async def _optimize(self) -> pipeline.JobResult:
        from app.optimizer.service import run_optimization

        r = await run_optimization(self.pool, self.fpl, self.settings.team_id)
        return pipeline.JobResult(rows_written=1, details={"run_id": r.run_id, "gameweek": r.plan.first.gameweek,
                                                           "gain": round(r.plan.objective - r.baseline.objective, 2)})

    # ------------------------------------------------------------------ one tick
    async def tick(self) -> timedelta:
        """Run every due job once; return how long to sleep before the next tick."""
        try:
            await self.run_locked("core", lambda: pipeline.ingest_core(self.pool, self.fpl))
        except Exception:
            log.exception("core ingestion failed")

        deadline = await self.next_deadline()
        phase = polling_phase(self.clock(), deadline)
        log.info("phase=%s next_deadline=%s", phase.value, deadline)

        jobs: list[tuple[str, Callable[[], Awaitable[pipeline.JobResult]]]] = []
        if await self._history_due():
            jobs.append(("history", lambda: pipeline.ingest_player_histories(
                self.pool, self.fpl,
                concurrency=self.settings.history_concurrency,
                only_with_minutes=self.settings.history_only_with_minutes,
            )))
        if self.settings.team_id and await self._due("entry", ENTRY_INTERVAL):
            team_id = self.settings.team_id
            jobs.append(("entry", lambda: pipeline.ingest_entry(self.pool, self.fpl, team_id)))
        if self.xg_adapter is not None and await self._due("xg", XG_INTERVAL):
            adapter = self.xg_adapter
            jobs.append(("xg", lambda: pipeline.ingest_xg(self.pool, adapter)))
        if self.odds_adapter is not None and phase in (Phase.STANDARD, Phase.PRE_DEADLINE):
            # Team odds only (2 credits). Goalscorer odds are fetched by the T-24h and T-30m re-plans.
            odds_interval = ODDS_INTERVAL[phase]
            if await self._due("odds", odds_interval):
                adapter_o = self.odds_adapter
                jobs.append(("odds", lambda: pipeline.ingest_odds(self.pool, adapter_o)))

        if self.archive_adapter is not None and await self._due("archive", ARCHIVE_INTERVAL):
            archive, seasons = self.archive_adapter, self.settings.archive_season_list
            jobs.append(("archive", lambda: pipeline.ingest_archive(self.pool, archive, seasons)))

        for job, fn in jobs:
            try:
                await self.run_locked(job, fn)
            except UpstreamError as exc:  # a source being down is expected now and then: one line, no traceback
                log.warning("%s skipped: %s", job, exc)
            except Exception:
                log.exception("%s ingestion failed", job)

        # Expected points: after ingestion so each run sees the freshest data.
        predict_every = PREDICT_INTERVAL.get(phase)
        if self.predict_enabled and predict_every is not None and await self._due("predict", predict_every):
            try:
                predicted = await self.run_locked("predict", self._predict)
            except Exception:
                predicted = None
                log.exception("xP prediction failed")
            # Keep the dashboard's plan current: re-plan on every new set of projections.
            if predicted is not None and self.settings.team_id:
                try:
                    await self.run_locked("optimize", self._optimize)
                except Exception:
                    log.exception("re-planning failed")

        # Section 7.1 keep-alive: detect expired FPL credentials long before a deadline.
        if self.executor is not None and self.settings.team_id and await self._due(
            "auth_keepalive", timedelta(hours=self.settings.keepalive_hours)
        ):
            has_tokens = await self.pool.fetchval(
                "SELECT encrypted_auth_cookie IS NOT NULL FROM system_config WHERE team_id=$1", self.settings.team_id)
            if has_tokens:
                async def keepalive() -> pipeline.JobResult:
                    ok = await self.executor.keepalive(self.settings.team_id)
                    if not ok:
                        raise RuntimeError("FPL credentials rejected or expired")
                    return pipeline.JobResult(rows_written=0)
                try:
                    await self.run_locked("auth_keepalive", keepalive)
                except Exception:
                    log.warning("auth keep-alive failed")

        # Section 7.2 deadline actions (T-24h dry run, T-30m final lock, T-15m commit).
        if self.scheduler is not None:
            try:
                await self.scheduler.run(self.clock())
            except Exception:
                log.exception("deadline actions failed")

        if await self._due("prune", PRUNE_INTERVAL):
            async def prune() -> pipeline.JobResult:
                n = await pipeline.prune_raw_snapshots(self.pool, self.settings.raw_snapshot_retention_days)
                return pipeline.JobResult(rows_written=n)
            try:
                await self.run_locked("prune", prune)
            except Exception:
                log.exception("raw snapshot pruning failed")

        delay = next_poll_delay(self.clock(), deadline)
        if self.scheduler is not None:
            wake = self.scheduler.next_wake(self.clock(), deadline)
            if wake is not None:
                delay = max(timedelta(seconds=30), min(delay, wake - self.clock() + timedelta(seconds=5)))
        return delay

    # ------------------------------------------------------------------ loop
    def stop(self) -> None:
        self._stop.set()

    async def run_forever(self) -> None:
        """Tick, sleep, repeat. A stop request interrupts a data-refresh tick at once, but never a
        commit to FPL in progress: that one is allowed to finish so the team is never half-updated."""
        while not self._stop.is_set():
            tick = asyncio.create_task(self.tick())
            stopper = asyncio.create_task(self._stop.wait())
            await asyncio.wait({tick, stopper}, return_when=asyncio.FIRST_COMPLETED)
            stopper.cancel()
            if not tick.done():
                if self.scheduler is not None and getattr(self.scheduler, "committing", False):
                    log.info("stop requested during a commit; letting it finish")
                    await tick
                else:
                    log.info("stop requested; cancelling the current cycle")
                    tick.cancel()
                    try:
                        await tick
                    except asyncio.CancelledError:
                        pass
                break
            try:
                delay = tick.result()
            except Exception:
                log.exception("worker cycle failed")
                delay = timedelta(minutes=5)
            log.info("sleeping %s", delay)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=delay.total_seconds())
            except TimeoutError:
                pass
        log.info("worker stopped")

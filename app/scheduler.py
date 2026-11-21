"""Deadline actions (Section 7.2 timeline), each run exactly once per gameweek:

* ``dry_run``    at T-24h: refresh data, xP, optimise; send the planned moves (with kill-switch button).
* ``final_lock`` at T-30m: final team news / odds, xP and optimise again; say if the plan changed.
* ``commit``     at T-15m: autonomous commit if enabled (guardrails apply), else a reminder to act manually.

The ``scheduled_actions`` table guarantees once-only execution across restarts and workers.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta
from typing import Any

import asyncpg

from app import ops
from app.config import Settings
from app.notify import Notifier

log = logging.getLogger(__name__)


class DeadlineScheduler:
    def __init__(
        self, settings: Settings, pool: asyncpg.Pool, notifier: Notifier, *,
        executor=None,
        refresh: Callable[[], Awaitable[None]] | None = None,
        predict: Callable[[], Awaitable[Any]] | None = None,
        optimize: Callable[[], Awaitable[Any]] | None = None,
    ):
        self.settings, self.pool, self.notifier, self.executor = settings, pool, notifier, executor
        self.team_id = settings.team_id
        self.committing = False  # read by the worker: never interrupt a commit on shutdown
        self._refresh = refresh or (lambda: ops.refresh_data(pool, settings))
        self._predict = predict or (lambda: ops.predict(pool))
        self._optimize = optimize or (lambda: ops.optimize(pool, settings, self.team_id))

    @property
    def windows(self) -> dict[str, timedelta]:
        s = self.settings
        return {
            "dry_run": timedelta(hours=s.dry_run_hours_before),
            "final_lock": timedelta(minutes=s.final_lock_minutes_before),
            "commit": timedelta(minutes=s.commit_minutes_before),
        }

    def next_wake(self, now: datetime, deadline: datetime | None) -> datetime | None:
        if deadline is None or not self.team_id:
            return None
        points = [deadline - w for w in self.windows.values()]
        future = [p for p in points if p > now]
        return min(future) if future else None

    async def _claim(self, gw: int, action: str) -> bool:
        row = await self.pool.fetchval(
            """INSERT INTO scheduled_actions (team_id, gameweek, action, status) VALUES ($1, $2, $3, 'RUNNING')
               ON CONFLICT (team_id, gameweek, action) DO NOTHING RETURNING action""",
            self.team_id, gw, action,
        )
        return row is not None

    async def _done(self, gw: int, action: str, status: str, detail: dict[str, Any]) -> None:
        await self.pool.execute(
            """UPDATE scheduled_actions SET status=$4, finished_at=NOW(), detail=$5
               WHERE team_id=$1 AND gameweek=$2 AND action=$3""",
            self.team_id, gw, action, status, detail,
        )

    async def _status(self, gw: int, action: str) -> str | None:
        return await self.pool.fetchval(
            "SELECT status FROM scheduled_actions WHERE team_id=$1 AND gameweek=$2 AND action=$3", self.team_id, gw, action)

    async def run(self, now: datetime) -> list[str]:
        """Run whatever deadline action is due now; returns the actions executed."""
        if not self.team_id:
            return []
        row = await self.pool.fetchrow(
            "SELECT gameweek_id, deadline_time FROM gameweeks WHERE deadline_time > $1 ORDER BY deadline_time LIMIT 1", now)
        if row is None:
            return []
        gw, deadline = row["gameweek_id"], row["deadline_time"]
        remaining = deadline - now
        w = self.windows
        cutoff = timedelta(minutes=self.settings.commit_cutoff_minutes)
        ran: list[str] = []

        if w["final_lock"] < remaining <= w["dry_run"] and await self._claim(gw, "dry_run"):
            ran.append("dry_run")
            await self._plan_action(gw, "dry_run", deadline)
        if cutoff < remaining <= w["final_lock"] and await self._claim(gw, "final_lock"):
            ran.append("final_lock")
            await self._plan_action(gw, "final_lock", deadline)
        if cutoff < remaining <= w["commit"] and await self._claim(gw, "commit"):
            ran.append("commit")
            await self._commit_action(gw)
        if remaining <= cutoff and await self._status(gw, "commit") is None and await self._claim(gw, "commit"):
            await self._done(gw, "commit", "MISSED", {"reason": "worker was not running inside the commit window"})
            if await self._auto_enabled():
                await self.notifier.send(f"GW{gw}: the T-15m commit window was missed (worker offline?). Check your team now.", urgent=True)
        return ran

    async def _auto_enabled(self) -> bool:
        return bool(await self.pool.fetchval("SELECT auto_execute_enabled FROM system_config WHERE team_id=$1", self.team_id))

    async def _plan_action(self, gw: int, action: str, deadline: datetime) -> None:
        try:
            await self._refresh()
            await self._predict()
            prev_hash = await self.pool.fetchval(
                """SELECT e.payload_hash FROM optimization_runs o JOIN execution_logs e ON e.id = o.execution_log_id
                   WHERE o.team_id=$1 AND o.base_gameweek=$2 ORDER BY o.created_at DESC LIMIT 1""", self.team_id, gw)
            await self._optimize()
            new_hash = await self.pool.fetchval(
                """SELECT e.payload_hash FROM optimization_runs o JOIN execution_logs e ON e.id = o.execution_log_id
                   WHERE o.team_id=$1 AND o.base_gameweek=$2 ORDER BY o.created_at DESC LIMIT 1""", self.team_id, gw)
        except Exception as exc:
            log.exception("%s failed", action)
            await self._done(gw, action, "FAILED", {"error": str(exc)[:1000]})
            await self.notifier.send(f"GW{gw} {action.replace('_', ' ')} failed: {exc}", urgent=True)
            return
        summary = await ops.plan_summary(self.pool, self.team_id, horizon=action == "dry_run")
        auto = await self._auto_enabled()
        mode = ("Auto-execute is ON: this will be submitted at T-15m unless you press the kill switch."
                if auto else "Auto-execute is OFF: make these changes yourself on the FPL site.")
        if action == "dry_run":
            await self.notifier.send(f"GW{gw} plan (deadline {deadline:%a %d %b %H:%M} UTC)\n{summary}\n{mode}", kill_button=auto)
        elif prev_hash != new_hash:
            await self.notifier.send(f"GW{gw} final plan CHANGED after late news:\n{summary}\n{mode}", kill_button=auto)
        await self._done(gw, action, "SUCCESS", {"changed": prev_hash != new_hash})

    async def _commit_action(self, gw: int) -> None:
        if await self._status(gw, "final_lock") is None and await self._claim(gw, "final_lock"):
            # Worker started inside the window: take the final data lock first.
            row = await self.pool.fetchrow("SELECT deadline_time FROM gameweeks WHERE gameweek_id=$1", gw)
            await self._plan_action(gw, "final_lock", row["deadline_time"])
        if not await self._auto_enabled():
            await self._done(gw, "commit", "SKIPPED", {"reason": "auto_execute_enabled is false"})
            summary = await ops.plan_summary(self.pool, self.team_id)
            await self.notifier.send(f"GW{gw}: auto-execute is off, so nothing was submitted. Recommended:\n{summary}")
            return
        if self.executor is None:
            await self._done(gw, "commit", "FAILED", {"reason": "executor not configured (FPL_SECRET_KEY missing?)"})
            await self.notifier.send(f"GW{gw}: auto-execute is on but the executor is not configured.", urgent=True)
            return
        self.committing = True
        try:
            result = await self.executor.commit(self.team_id, trigger="scheduled")
        finally:
            self.committing = False
        await self._done(gw, "commit", "SUCCESS" if result.status == "SUCCESS" else "FAILED",
                         {"status": result.status, "reason": result.reason, "execution_log_id": result.execution_log_id})

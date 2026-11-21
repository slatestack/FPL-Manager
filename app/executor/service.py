"""Autonomous commit (Section 7.2, T-15m) and authentication keep-alive (Section 7.1).

    lock -> load latest plan -> PENDING log -> guardrails on live data -> POST transfers
    -> POST my-team -> GET my-team parity check -> SUCCESS / FAILED / ABORTED + notification

Idempotent by state: if the live squad already equals the plan's target squad (an earlier
attempt applied the transfers), the transfer step is skipped instead of repeated.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import asyncpg
import httpx
from redis.asyncio import Redis

from app.config import Settings
from app.executor import guardrails
from app.executor.client import AuthenticatedFPLClient, MyTeam
from app.executor.payloads import TRANSFER_CHIPS, picks_payload, transfers_payload
from app.executor.vault import AuthError, TokenVault
from app.notify import Notifier

log = logging.getLogger(__name__)
CHIP_LABEL = {"3xc": "Triple Captain", "bboost": "Bench Boost", "freehit": "Free Hit", "wildcard": "Wildcard"}


@dataclass
class CommitResult:
    status: str                       # SUCCESS | FAILED | ABORTED
    reason: str
    execution_log_id: str | None = None
    gameweek: int | None = None
    report: list[dict[str, Any]] = field(default_factory=list)


def decision_hash(plan_gw: dict[str, Any], chip: str | None) -> str:
    payload = {
        "gameweek": plan_gw["gameweek"], "transfers_in": sorted(plan_gw["transfers_in"]),
        "transfers_out": sorted(plan_gw["transfers_out"]), "starting_xi": sorted(plan_gw["starting_xi"]),
        "bench_order": plan_gw["bench"], "captain": plan_gw["captain"], "vice_captain": plan_gw["vice_captain"], "chip": chip,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def parity_problems(live: MyTeam, target_squad: set[int], plan_gw: dict[str, Any], chip: str | None) -> list[str]:
    problems = []
    if live.squad != target_squad:
        problems.append(f"squad differs: +{sorted(live.squad - target_squad)} -{sorted(target_squad - live.squad)}")
    if live.starting_xi != set(plan_gw["starting_xi"]):
        problems.append("starting XI differs")
    if live.bench_order != list(plan_gw["bench"]):
        problems.append(f"bench order {live.bench_order} != {plan_gw['bench']}")
    if live.captain != plan_gw["captain"] or live.vice_captain != plan_gw["vice_captain"]:
        problems.append(f"captaincy {live.captain}/{live.vice_captain} != {plan_gw['captain']}/{plan_gw['vice_captain']}")
    if chip and live.chip_status(chip) != "active" and live.active_chip != chip:
        problems.append(f"{chip} not active (status {live.chip_status(chip)})")
    return problems


class Executor:
    def __init__(
        self, settings: Settings, pool: asyncpg.Pool, redis: Redis, notifier: Notifier, *,
        transport: httpx.AsyncBaseTransport | None = None, clock=lambda: datetime.now(timezone.utc),
    ):
        self.settings, self.pool, self.redis, self.notifier = settings, pool, redis, notifier
        self.transport = transport
        self.clock = clock
        self.vault = TokenVault(settings, pool, transport=transport)

    def _client(self, tokens) -> AuthenticatedFPLClient:
        async def nosleep(_: float) -> None:
            return None

        return AuthenticatedFPLClient(self.settings, tokens, transport=self.transport,
                                      **({"sleep": nosleep} if self.transport is not None else {}))

    async def _finish(self, log_id: str, status: str, reason: str, details: dict[str, Any]) -> None:
        await self.pool.execute(
            "UPDATE execution_logs SET status=$2, error_message=$3, details=$4, finished_at=NOW() WHERE id=$1::uuid",
            log_id, status, None if status == "SUCCESS" else reason[:4000], details,
        )

    async def commit(self, team_id: int, *, trigger: str = "scheduled", respect_kill_switch: bool = True) -> CommitResult:
        now = self.clock()
        gw_row = await self.pool.fetchrow(
            "SELECT gameweek_id, deadline_time FROM gameweeks WHERE deadline_time > $1 ORDER BY deadline_time LIMIT 1", now
        )
        if gw_row is None:
            return CommitResult("ABORTED", "no upcoming deadline")
        gw, deadline = gw_row["gameweek_id"], gw_row["deadline_time"]
        lock = self.redis.lock(f"fpl_commit_lock:gw_{gw}", timeout=600)
        if not await lock.acquire(blocking=False):
            return CommitResult("ABORTED", "another commit for this gameweek is in progress", gameweek=gw)
        try:
            return await self._commit_locked(team_id, gw, deadline, now, trigger, respect_kill_switch)
        finally:
            try:
                await lock.release()
            except Exception:
                log.warning("commit lock expired before release")

    async def _commit_locked(self, team_id: int, gw: int, deadline: datetime, now: datetime, trigger: str, respect_kill_switch: bool) -> CommitResult:
        opt = await self.pool.fetchrow(
            """SELECT id, created_at, plan, chip_recommended, inputs FROM optimization_runs
               WHERE team_id = $1 AND base_gameweek = $2 ORDER BY created_at DESC LIMIT 1""",
            team_id, gw,
        )
        if opt is None:
            msg = f"GW{gw}: no optimiser plan to execute"
            await self.notifier.send(msg, urgent=True)
            return CommitResult("ABORTED", msg, gameweek=gw)
        plan_gw, chip = opt["plan"][0], opt["chip_recommended"]
        outs, ins = list(plan_gw["transfers_out"]), list(plan_gw["transfers_in"])
        start_squad = set(opt["inputs"]["team_state"]["squad"])
        target_squad = set(plan_gw["squad"])
        digest = decision_hash(plan_gw, chip)
        duplicate = await self.pool.fetchval(
            """SELECT EXISTS (SELECT 1 FROM execution_logs WHERE team_id=$1 AND gameweek=$2 AND execution_type='LIVE_COMMIT'
               AND status='SUCCESS' AND payload_hash=$3)""",
            team_id, gw, digest,
        )
        log_id = str(await self.pool.fetchval(
            """INSERT INTO execution_logs (team_id, gameweek, execution_type, payload_hash, transfers_in, transfers_out,
                   starting_xi, bench_order, captain_id, vice_captain_id, chip_activated, status, trigger)
               VALUES ($1, $2, 'LIVE_COMMIT', $3, $4, $5, $6, $7, $8, $9, $10, 'PENDING', $11) RETURNING id""",
            team_id, gw, digest, ins, outs, plan_gw["starting_xi"], plan_gw["bench"], plan_gw["captain"],
            plan_gw["vice_captain"], chip, trigger,
        ))
        details: dict[str, Any] = {"optimization_run": str(opt["id"]), "trigger": trigger}

        try:
            tokens = await self.vault.valid_tokens(team_id)
        except AuthError as exc:
            msg = f"GW{gw} commit aborted: {exc}"
            await self._finish(log_id, "ABORTED", msg, details)
            await self.notifier.send(msg + ". Re-import tokens with `python -m app auth set`.", urgent=True)
            return CommitResult("ABORTED", msg, log_id, gw)

        cfg = await self.pool.fetchrow("SELECT auto_execute_enabled, max_hits_per_gw FROM system_config WHERE team_id=$1", team_id)
        prices = {r["player_id"]: (r["now_cost"], r["position"], r["web_name"]) for r in await self.pool.fetch(
            "SELECT player_id, now_cost, position, web_name FROM players")}
        now_cost = {k: v[0] for k, v in prices.items()}
        position = {k: v[1] for k, v in prices.items()}
        name = lambda p: prices.get(p, (0, "", str(p)))[2]  # noqa: E731

        try:
            async with self._client(tokens) as client:
                live = await client.my_team(team_id)
                already_applied = bool(outs) and live.squad == target_squad
                report = guardrails.evaluate(
                    auto_execute_enabled=bool(cfg and cfg["auto_execute_enabled"]),
                    respect_kill_switch=respect_kill_switch, now=now, deadline=deadline,
                    cutoff=timedelta(minutes=self.settings.commit_cutoff_minutes), plan_created_at=opt["created_at"],
                    plan_max_age=timedelta(minutes=self.settings.plan_max_age_minutes), plan_gameweek=plan_gw["gameweek"],
                    next_gameweek=gw, expected_squad=target_squad if already_applied else start_squad, live_squad=live.squad,
                    outs=[] if already_applied else outs, ins=[] if already_applied else ins, chip=chip,
                    live_free_transfers=live.free_transfers, max_hits=int(cfg["max_hits_per_gw"]) if cfg else 1,
                    live_bank=live.bank, live_selling={p.element: p.selling_price for p in live.picks}, now_cost=now_cost,
                    duplicate_success=bool(duplicate), chip_status=live.chip_status(chip) if chip else None,
                )
                details["guardrails"] = report.as_list()
                if not report.ok:
                    msg = f"GW{gw} commit aborted: {report.summary()}"
                    await self._finish(log_id, "ABORTED", msg, details)
                    only_dup = [c.name for c in report.failures] == ["duplicate"]
                    only_kill = [c.name for c in report.failures] == ["kill_switch"]
                    await self.notifier.send(msg, urgent=not (only_dup or only_kill))
                    return CommitResult("ABORTED", msg, log_id, gw, report.as_list())

                if (outs and not already_applied) or chip in TRANSFER_CHIPS:
                    tp = transfers_payload(
                        entry=team_id, event=gw, outs=outs, ins=ins, chip=chip, position=position,
                        selling_price={p.element: p.selling_price for p in live.picks}, now_cost=now_cost,
                    )
                    details["transfers_request"] = tp
                    resp = await client.post_transfers(tp)
                    details["transfers_response"] = {"status": resp.status_code, "body": resp.text[:1000]}
                    if resp.status_code >= 300:
                        check = await client.my_team(team_id)
                        if check.squad != target_squad:
                            msg = f"GW{gw} transfers rejected (HTTP {resp.status_code}): {resp.text[:300]}"
                            await self._finish(log_id, "FAILED", msg, details)
                            await self.notifier.send(msg, urgent=True)
                            return CommitResult("FAILED", msg, log_id, gw, report.as_list())

                pp = picks_payload(starting_xi=plan_gw["starting_xi"], bench=plan_gw["bench"], captain=plan_gw["captain"],
                                   vice_captain=plan_gw["vice_captain"], chip=chip, position=position)
                details["picks_request"] = pp
                resp = await client.post_my_team(team_id, pp)
                details["picks_response"] = {"status": resp.status_code, "body": resp.text[:1000]}

                verify = await client.my_team(team_id)
                problems = parity_problems(verify, target_squad, plan_gw, chip)
                details["verify"] = {"problems": problems, "squad": sorted(verify.squad), "captain": verify.captain}
                if resp.status_code >= 300 or problems:
                    msg = f"GW{gw} commit not confirmed (lineup HTTP {resp.status_code}): {'; '.join(problems) or resp.text[:300]}"
                    await self._finish(log_id, "FAILED", msg, details)
                    await self.notifier.send(msg + " Check your team on the FPL site.", urgent=True)
                    return CommitResult("FAILED", msg, log_id, gw, report.as_list())
        except AuthError as exc:
            msg = f"GW{gw} commit aborted: {exc}"
            await self.vault.record_check(team_id, False, str(exc))
            await self._finish(log_id, "ABORTED", msg, details)
            await self.notifier.send(msg + ". Re-import tokens with `python -m app auth set`.", urgent=True)
            return CommitResult("ABORTED", msg, log_id, gw)
        except Exception as exc:
            msg = f"GW{gw} commit failed: {type(exc).__name__}: {exc}"
            await self._finish(log_id, "FAILED", msg, details)
            await self.notifier.send(msg + " Check your team on the FPL site.", urgent=True)
            return CommitResult("FAILED", msg, log_id, gw)

        await self._finish(log_id, "SUCCESS", "", details)
        from app.executor.payloads import pair_transfers

        moves = ", ".join(f"{name(o)} → {name(i)}" for o, i in pair_transfers(outs, ins, position)) if outs else "no transfers"
        chip_txt = f" · {CHIP_LABEL.get(chip, chip)} played" if chip else ""
        await self.notifier.send(
            f"✅ GW{gw} submitted and verified: {moves}; captain {name(plan_gw['captain'])}, vice {name(plan_gw['vice_captain'])}{chip_txt}."
        )
        return CommitResult("SUCCESS", "submitted and verified", log_id, gw, report.as_list())

    async def keepalive(self, team_id: int) -> bool:
        """Section 7.1: ping my-team (refreshing the token if due); alert once per 6h while broken."""
        try:
            tokens = await self.vault.valid_tokens(team_id)
            async with self._client(tokens) as client:
                await client.my_team(team_id)
        except Exception as exc:
            await self.vault.record_check(team_id, False, str(exc))
            row = await self.pool.fetchrow("SELECT last_auth_alert_at FROM system_config WHERE team_id=$1", team_id)
            last = row["last_auth_alert_at"] if row else None
            if last is None or self.clock() - last > timedelta(hours=6):
                await self.notifier.send(
                    f"FPL login needs attention: {exc}. Automatic execution cannot run until tokens are re-imported "
                    "(`python -m app auth set` or PATCH /api/v1/config).", urgent=True)
                await self.pool.execute("UPDATE system_config SET last_auth_alert_at = NOW() WHERE team_id=$1", team_id)
            return False
        await self.vault.record_check(team_id, True)
        return True

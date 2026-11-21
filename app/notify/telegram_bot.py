"""Interactive Telegram control (Section 2 box 6): long-polls ``getUpdates`` and only obeys the
configured chat. Commands: /status /plan /killswitch /enable /dryrun /help, plus the inline
"Kill switch" button attached to plan messages.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

import asyncpg
from redis.asyncio import Redis

from app import ops
from app.config import Settings
from app.notify import TelegramNotifier

log = logging.getLogger(__name__)
OFFSET_KEY = "telegram:update_offset"

HELP = (
    "/status – deadline, auto-execute, login health, last execution\n"
    "/plan – planned moves for the next 5 gameweeks\n"
    "/killswitch – disable autonomous execution immediately\n"
    "/enable – re-enable autonomous execution\n"
    "/dryrun – refresh projections and re-plan now"
)


class TelegramBot:
    def __init__(self, settings: Settings, pool: asyncpg.Pool, redis: Redis, telegram: TelegramNotifier, *, dry_run=None):
        self.settings, self.pool, self.redis, self.tg = settings, pool, redis, telegram
        self.team_id = settings.team_id
        self._dry_run = dry_run or self._default_dry_run
        self._tasks: set[asyncio.Task] = set()

    async def _default_dry_run(self) -> None:
        await ops.refresh_data(self.pool, self.settings)
        await ops.predict(self.pool)
        await ops.optimize(self.pool, self.settings, self.team_id)

    async def reply(self, text: str, *, kill_button: bool = False) -> None:
        await self.tg.send(text, kill_button=kill_button)

    async def set_auto_execute(self, enabled: bool) -> None:
        await self.pool.execute(
            """INSERT INTO system_config (team_id, auto_execute_enabled) VALUES ($1, $2)
               ON CONFLICT (team_id) DO UPDATE SET auto_execute_enabled = $2, updated_at = NOW()""",
            self.team_id, enabled,
        )

    async def status_text(self) -> str:
        now = datetime.now(timezone.utc)
        gw = await self.pool.fetchrow(
            "SELECT gameweek_id, deadline_time FROM gameweeks WHERE deadline_time > $1 ORDER BY deadline_time LIMIT 1", now)
        cfg = await self.pool.fetchrow(
            """SELECT auto_execute_enabled, encrypted_auth_cookie IS NOT NULL AS has_tokens, auth_last_check_ok,
                      auth_last_check_at FROM system_config WHERE team_id=$1""", self.team_id)
        last = await self.pool.fetchrow(
            """SELECT gameweek, status, error_message, created_at FROM execution_logs
               WHERE team_id=$1 AND execution_type='LIVE_COMMIT' ORDER BY created_at DESC LIMIT 1""", self.team_id)
        lines = []
        if gw:
            mins = int((gw["deadline_time"] - now).total_seconds() // 60)
            lines.append(f"GW{gw['gameweek_id']} deadline in {mins // 1440}d {mins % 1440 // 60}h {mins % 60}m")
        lines.append(f"Auto-execute: {'ON' if cfg and cfg['auto_execute_enabled'] else 'OFF'}")
        if cfg and cfg["has_tokens"]:
            ok = cfg["auth_last_check_ok"]
            lines.append(f"FPL login: {'OK' if ok else 'FAILING' if ok is False else 'not checked yet'}")
        else:
            lines.append("FPL login: no tokens stored")
        if last:
            lines.append(f"Last commit: GW{last['gameweek']} {last['status']}" + (f" ({last['error_message'][:120]})" if last["error_message"] else ""))
        return "\n".join(lines)

    async def handle(self, update: dict[str, Any]) -> None:
        chat_id = None
        if "callback_query" in update:
            cq = update["callback_query"]
            chat_id = str(cq.get("message", {}).get("chat", {}).get("id"))
            if chat_id != str(self.tg.chat_id):
                log.warning("ignoring callback from unauthorised chat %s", chat_id)
                return
            if cq.get("data") == "killswitch":
                await self.set_auto_execute(False)
                await self.tg.call("answerCallbackQuery", {"callback_query_id": cq["id"], "text": "Auto-execute disabled"})
                await self.reply("🛑 Kill switch engaged: autonomous execution is OFF. /enable to turn it back on.")
            return
        msg = update.get("message") or {}
        chat_id = str(msg.get("chat", {}).get("id"))
        if chat_id != str(self.tg.chat_id):
            log.warning("ignoring message from unauthorised chat %s", chat_id)
            return
        cmd = (msg.get("text") or "").strip().split()[0].split("@")[0].lower() if msg.get("text") else ""
        if cmd == "/killswitch":
            await self.set_auto_execute(False)
            await self.reply("🛑 Kill switch engaged: autonomous execution is OFF. /enable to turn it back on.")
        elif cmd == "/enable":
            verified = await self.pool.fetchval(
                "SELECT encrypted_auth_cookie IS NOT NULL AND auth_last_check_ok IS TRUE FROM system_config WHERE team_id=$1",
                self.team_id)
            if not verified:
                await self.reply("Cannot enable: the FPL login has not passed a live check (`python -m app auth set`).")
                return
            await self.set_auto_execute(True)
            await self.reply("✅ Autonomous execution is ON. The plan will be submitted at T-15m (guardrails apply).", kill_button=True)
        elif cmd == "/status":
            await self.reply(await self.status_text())
        elif cmd == "/plan":
            await self.reply(await ops.plan_summary(self.pool, self.team_id, horizon=True))
        elif cmd == "/dryrun":
            await self.reply("Refreshing data and re-planning (about a minute)…")
            task = asyncio.create_task(self._run_dry_run())
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
        elif cmd in ("/help", "/start"):
            await self.reply(HELP)

    async def _run_dry_run(self) -> None:
        try:
            await self._dry_run()
            await self.reply(await ops.plan_summary(self.pool, self.team_id, horizon=True))
        except Exception as exc:
            await self.reply(f"Dry run failed: {exc}")

    async def poll_once(self, timeout: int = 25) -> int:
        offset = int(await self.redis.get(OFFSET_KEY) or 0)
        data = await self.tg.call("getUpdates", {"offset": offset, "timeout": timeout,
                                                 "allowed_updates": ["message", "callback_query"]}, timeout=timeout + 10)
        updates = data.get("result", [])
        for u in updates:
            await self.redis.set(OFFSET_KEY, int(u["update_id"]) + 1)  # acknowledge before acting: never re-run a command
            try:
                await self.handle(u)
            except Exception:
                log.exception("telegram update failed")
        return len(updates)

    async def run_forever(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            try:
                await self.poll_once()
            except Exception as exc:
                log.warning("telegram polling error: %s", exc)
                try:
                    await asyncio.wait_for(stop.wait(), timeout=15)
                except TimeoutError:
                    pass

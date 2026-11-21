import json
import os
from datetime import datetime, timedelta, timezone

import httpx
import pytest
import respx
from cryptography.fernet import Fernet
from redis.asyncio import Redis

from app.api.main import app
from app.config import Settings, get_settings
from app.executor.service import CommitResult
from app.notify import TelegramNotifier
from app.notify.telegram_bot import TelegramBot
from app.scheduler import DeadlineScheduler

TEAM = 4242
DEADLINE = datetime(2026, 11, 21, 11, 0, tzinfo=timezone.utc)


class Notes:
    def __init__(self):
        self.messages = []

    async def send(self, text, *, urgent=False, kill_button=False):
        self.messages.append({"text": text, "urgent": urgent, "kill_button": kill_button})


class FakeExecutor:
    def __init__(self):
        self.calls = 0

    async def commit(self, team_id, trigger="scheduled", respect_kill_switch=True):
        self.calls += 1
        return CommitResult("SUCCESS", "ok", "00000000-0000-0000-0000-000000000000", 3)


@pytest.fixture
async def sched_env(pool, fpl):
    from app.ingestion import pipeline

    await pipeline.ingest_core(pool, fpl)
    await pool.execute("UPDATE gameweeks SET deadline_time = $1 WHERE gameweek_id = 3", DEADLINE)
    await pool.execute("INSERT INTO system_config (team_id, auto_execute_enabled) VALUES ($1, TRUE)", TEAM)
    calls = []

    async def step(name):
        calls.append(name)

    notes, executor = Notes(), FakeExecutor()
    s = DeadlineScheduler(Settings(database_url="", team_id=TEAM), pool, notes, executor=executor,
                          refresh=lambda: step("refresh"), predict=lambda: step("predict"), optimize=lambda: step("optimize"))
    return s, notes, executor, calls, pool


async def test_deadline_timeline_runs_each_action_once(sched_env):
    s, notes, executor, calls, pool = sched_env
    assert await s.run(DEADLINE - timedelta(hours=30)) == []
    assert await s.run(DEADLINE - timedelta(hours=20)) == ["dry_run"]
    assert await s.run(DEADLINE - timedelta(hours=19)) == [], "once per gameweek"
    assert notes.messages[-1]["kill_button"] is True and "Auto-execute is ON" in notes.messages[-1]["text"]
    assert await s.run(DEADLINE - timedelta(minutes=25)) == ["final_lock"]
    assert await s.run(DEADLINE - timedelta(minutes=14)) == ["commit"]
    assert executor.calls == 1
    statuses = {r["action"]: r["status"] for r in await pool.fetch("SELECT action, status FROM scheduled_actions")}
    assert statuses == {"dry_run": "SUCCESS", "final_lock": "SUCCESS", "commit": "SUCCESS"}
    assert calls.count("optimize") == 2


async def test_commit_skipped_when_auto_execute_off(sched_env):
    s, notes, executor, calls, pool = sched_env
    await pool.execute("UPDATE system_config SET auto_execute_enabled = FALSE")
    assert await s.run(DEADLINE - timedelta(minutes=14)) == ["final_lock", "commit"]
    assert executor.calls == 0 and calls.count("optimize") == 1, "final lock still runs first"
    assert "nothing was submitted" in notes.messages[-1]["text"]


async def test_missed_commit_window_alerts(sched_env):
    s, notes, executor, calls, pool = sched_env
    await s.run(DEADLINE - timedelta(minutes=2))
    assert await pool.fetchval("SELECT status FROM scheduled_actions WHERE action='commit'") == "MISSED"
    assert notes.messages[-1]["urgent"] is True and executor.calls == 0


def test_next_wake_points():
    s = DeadlineScheduler(Settings(database_url="", team_id=TEAM), None, Notes())
    assert s.next_wake(DEADLINE - timedelta(hours=30), DEADLINE) == DEADLINE - timedelta(hours=24)
    assert s.next_wake(DEADLINE - timedelta(hours=1), DEADLINE) == DEADLINE - timedelta(minutes=30)
    assert s.next_wake(DEADLINE - timedelta(minutes=20), DEADLINE) == DEADLINE - timedelta(minutes=15)


# ------------------------------------------------------------------ Telegram


@pytest.fixture
async def bot(pool):
    redis = Redis.from_url(os.environ.get("TEST_REDIS_URL", "redis://localhost:6379/15"))
    await redis.flushdb()
    await pool.execute(
        "INSERT INTO system_config (team_id, auto_execute_enabled, encrypted_auth_cookie, auth_last_check_ok) VALUES ($1, TRUE, 'x', TRUE)", TEAM)
    b = TelegramBot(Settings(database_url="", team_id=TEAM), pool, redis, TelegramNotifier("TKN", "777"))
    yield b
    await redis.aclose()


def _updates(*items):
    return httpx.Response(200, json={"ok": True, "result": list(items)})


async def test_killswitch_command_and_button(bot, pool):
    with respx.mock() as router:
        sent = router.post("https://api.telegram.org/botTKN/sendMessage").mock(return_value=httpx.Response(200, json={"ok": True, "result": {}}))
        router.post("https://api.telegram.org/botTKN/answerCallbackQuery").mock(return_value=httpx.Response(200, json={"ok": True, "result": True}))
        router.post("https://api.telegram.org/botTKN/getUpdates").mock(side_effect=[
            _updates({"update_id": 10, "message": {"chat": {"id": 999}, "text": "/killswitch"}}),   # stranger: ignored
            _updates({"update_id": 11, "message": {"chat": {"id": 777}, "text": "/killswitch"}}),
            _updates({"update_id": 12, "message": {"chat": {"id": 777}, "text": "/enable"}}),
            _updates({"update_id": 13, "callback_query": {"id": "cb", "data": "killswitch", "message": {"chat": {"id": 777}}}}),
        ])
        await bot.poll_once(timeout=0)
        assert await pool.fetchval("SELECT auto_execute_enabled FROM system_config") is True
        assert sent.call_count == 0
        await bot.poll_once(timeout=0)
        assert await pool.fetchval("SELECT auto_execute_enabled FROM system_config") is False
        await bot.poll_once(timeout=0)
        assert await pool.fetchval("SELECT auto_execute_enabled FROM system_config") is True
        await bot.poll_once(timeout=0)
        assert await pool.fetchval("SELECT auto_execute_enabled FROM system_config") is False
        assert int(await bot.redis.get("telegram:update_offset")) == 14
        assert "Kill switch engaged" in json.loads(sent.calls[-1].request.content)["text"]


# ------------------------------------------------------------------ control-plane API


@pytest.fixture
async def api(dsn, monkeypatch):
    monkeypatch.setenv("FPL_DATABASE_URL", dsn)
    monkeypatch.setenv("FPL_API_TOKEN", "s3cret")
    monkeypatch.setenv("FPL_SECRET_KEY", Fernet.generate_key().decode())
    monkeypatch.setenv("FPL_TEAM_ID", str(TEAM))
    get_settings.cache_clear()
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
            yield c
    get_settings.cache_clear()


async def test_control_endpoints_require_token(api):
    assert (await api.patch(f"/api/v1/config/{TEAM}", json={"auto_execute_enabled": False})).status_code == 401
    bad = await api.patch(f"/api/v1/config/{TEAM}", json={}, headers={"Authorization": "Bearer nope"})
    assert bad.status_code == 401
    assert (await api.post("/api/v1/execute/force")).status_code == 401


async def test_config_patch_and_status(api):
    h = {"Authorization": "Bearer s3cret"}
    r = await api.patch(f"/api/v1/config/{TEAM}", json={"auto_execute_enabled": True}, headers=h)
    assert r.status_code == 409, "cannot arm without tokens"
    blob = json.dumps({"access_token": "AT", "refresh_token": "RT", "expires_at": 1900000000})
    with respx.mock(assert_all_called=False) as router:
        router.get(url__regex=r".*/my-team/4242/$").mock(return_value=httpx.Response(401, json={}))
        r = await api.patch(f"/api/v1/config/{TEAM}", json={"auth_tokens": blob, "auto_execute_enabled": True}, headers=h)
    assert r.status_code == 409, "tokens that fail the live check cannot arm execution"
    with respx.mock(assert_all_called=False) as router:
        router.get(url__regex=r".*/my-team/4242/$").mock(return_value=httpx.Response(200, json={"picks": [], "chips": [], "transfers": {}}))
        r = await api.patch(f"/api/v1/config/{TEAM}", json={"auth_tokens": blob, "auto_execute_enabled": True, "max_hits_per_gw": 0}, headers=h)
    assert r.status_code == 200 and r.json() == {"auto_execute_enabled": True, "max_hits_per_gw": 0, "chip_aggression_profile": "BALANCED"}
    st = (await api.get(f"/api/v1/status/{TEAM}")).json()
    assert st["auth"]["has_tokens"] is True and st["config"]["auto_execute_enabled"] is True
    assert "AT" not in json.dumps(st), "tokens are never echoed"
    assert (await api.get(f"/api/v1/audit/{TEAM}")).json() == {"team_id": TEAM, "executions": []}

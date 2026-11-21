"""End-to-end commits against a stateful fake of FPL's authenticated API."""

import json
import os
from datetime import datetime, timedelta, timezone

import httpx
import pytest
import respx
from cryptography.fernet import Fernet
from redis.asyncio import Redis

from app.config import Settings
from app.executor.service import Executor
from app.executor.vault import AuthTokens, TokenVault

pytestmark = pytest.mark.skipif(not os.environ.get("TEST_REDIS_URL"), reason="TEST_REDIS_URL not set")
TEAM = 4242
DEADLINE = datetime(2026, 11, 21, 11, 0, tzinfo=timezone.utc)
NOW = DEADLINE - timedelta(minutes=14)
POS = ["GKP"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3 + ["FWD"]          # ids 101..116
START = list(range(101, 116))
TARGET = [p for p in START if p != 115] + [116]
XI = [101, 103, 104, 105, 106, 108, 109, 110, 111, 113, 116]
BENCH = [102, 107, 112, 114]


class RecordingNotifier:
    def __init__(self):
        self.messages = []

    async def send(self, text, *, urgent=False, kill_button=False):
        self.messages.append((text, urgent))


class FakeFPL:
    """Holds a team; applies transfers/picks like FPL does and serves my-team."""

    def __init__(self, squad, ft=1, bank=10, price_116=70):
        self.squad = list(squad)
        self.ft, self.bank = ft, bank
        self.picks = [{"element": p, "position": k, "is_captain": k == 1, "is_vice_captain": k == 2}
                      for k, p in enumerate(self.squad, start=1)]
        self.calls = []
        self.transfers_status = 200
        self.apply_picks = True
        self.price_116 = price_116
        self.chips = [{"name": "bboost", "status_for_entry": "available"}]

    def my_team(self, request):
        assert request.headers["X-API-Authorization"].startswith("Bearer ")
        self.calls.append(("GET", request.headers["X-API-Authorization"]))
        sell = lambda p: 60 if p != 116 else self.price_116  # noqa: E731
        return httpx.Response(200, json={
            "picks": [{**p, "selling_price": sell(p["element"]), "purchase_price": sell(p["element"])} for p in self.picks],
            "chips": self.chips,
            "transfers": {"cost": 4, "status": "cost", "limit": self.ft, "made": 0, "bank": self.bank, "value": 1000}})

    def transfers(self, request):
        body = json.loads(request.content)
        self.calls.append(("POST transfers", body))
        for t in body["transfers"]:
            idx = self.squad.index(t["element_out"])
            self.squad[idx] = t["element_in"]
            self.bank += t["selling_price"] - t["purchase_price"]
            for p in self.picks:
                if p["element"] == t["element_out"]:
                    p["element"] = t["element_in"]
        return httpx.Response(self.transfers_status, json={})

    def my_team_post(self, request):
        body = json.loads(request.content)
        self.calls.append(("POST picks", body))
        if self.apply_picks:
            self.picks = [{k: v for k, v in p.items()} for p in body["picks"]]
            if body["chip"]:
                for c in self.chips:
                    if c["name"] == body["chip"]:
                        c["status_for_entry"] = "active"
        return httpx.Response(200, json={})

    def posts(self):
        return [c[0] for c in self.calls if c[0].startswith("POST")]


@pytest.fixture
async def env(pool, fpl):
    from app.ingestion import pipeline

    await pipeline.ingest_core(pool, fpl)
    # A full synthetic squad + one transfer target.
    await pool.executemany(
        "INSERT INTO players (player_id, web_name, team_id, position, now_cost, status) VALUES ($1, $2, $3, $4, $5, 'a')",
        [(101 + i, f"P{101 + i}", i % 4 + 1, pos, 70 if 101 + i == 116 else 60) for i, pos in enumerate(POS)],
    )
    await pool.execute("UPDATE gameweeks SET deadline_time = $1 WHERE gameweek_id = 3", DEADLINE)
    await pool.execute(
        "INSERT INTO prediction_runs (run_timestamp, model_version, base_gameweek, horizon, params, inputs) VALUES ($1, 't', 3, 1, '{}', '{}')",
        NOW - timedelta(minutes=10))
    plan = [{"gameweek": 3, "squad": sorted(TARGET), "starting_xi": XI, "bench": BENCH, "captain": 116, "vice_captain": 113,
             "transfers_in": [116], "transfers_out": [115], "hits": 0, "free_transfers": 1, "bank": 0, "chip": None, "xp": 55.0}]
    await pool.execute(
        """INSERT INTO optimization_runs (created_at, team_id, base_gameweek, prediction_run_timestamp, inputs, params, solver, plan,
               chip_evaluation, chip_recommended, objective, baseline_objective)
           VALUES ($1, $2, 3, $3, $4, '{}', '{}', $5, '[]', NULL, 60, 50)""",
        NOW - timedelta(minutes=5), TEAM, NOW - timedelta(minutes=10), {"team_state": {"squad": START}}, plan)
    settings = Settings(database_url="", api_base_url="https://fpl.test/api", team_id=TEAM, secret_key=Fernet.generate_key().decode(),
                        http_max_retries=0)
    vault = TokenVault(settings, pool)
    # Far-future expiry: the vault checks expiry against the real clock, so a fixed date would rot.
    await vault.store(TEAM, AuthTokens("AT1", "RT1", datetime(2100, 1, 1, tzinfo=timezone.utc)))
    await pool.execute("UPDATE system_config SET auto_execute_enabled = TRUE WHERE team_id = $1", TEAM)
    redis = Redis.from_url(os.environ["TEST_REDIS_URL"])
    await redis.flushdb()
    yield settings, pool, redis
    await redis.aclose()


def mount(fake: FakeFPL, router):
    router.get(f"https://fpl.test/api/my-team/{TEAM}/").mock(side_effect=fake.my_team)
    router.post("https://fpl.test/api/transfers/").mock(side_effect=fake.transfers)
    router.post(f"https://fpl.test/api/my-team/{TEAM}/").mock(side_effect=fake.my_team_post)


async def run(env, fake, **kw):
    settings, pool, redis = env
    notes = RecordingNotifier()
    with respx.mock(assert_all_called=False) as router:
        mount(fake, router)
        refresh = router.post(settings.oidc_token_url)
        if "refresh" in kw:
            refresh.mock(return_value=kw.pop("refresh"))
        ex = Executor(settings, pool, redis, notes, clock=lambda: NOW)
        result = await ex.commit(TEAM, **kw)
    return result, notes


async def test_successful_commit_submits_exact_payloads_and_verifies(env):
    fake = FakeFPL(START)
    result, notes = await run(env, fake)
    assert result.status == "SUCCESS", result.reason
    transfers = next(c[1] for c in fake.calls if c[0] == "POST transfers")
    assert transfers == {"chip": None, "entry": TEAM, "event": 3,
                         "transfers": [{"element_in": 116, "element_out": 115, "purchase_price": 70, "selling_price": 60}]}
    picks = next(c[1] for c in fake.calls if c[0] == "POST picks")
    assert [p["element"] for p in picks["picks"]][11:] == BENCH
    assert set(fake.squad) == set(TARGET) and fake.bank == 0
    row = await env[1].fetchrow("SELECT status, details FROM execution_logs WHERE id = $1::uuid", result.execution_log_id)
    assert row["status"] == "SUCCESS" and row["details"]["verify"]["problems"] == []
    assert "submitted and verified" in notes.messages[-1][0]


async def test_kill_switch_aborts_without_touching_fpl(env):
    await env[1].execute("UPDATE system_config SET auto_execute_enabled = FALSE")
    fake = FakeFPL(START)
    result, notes = await run(env, fake)
    assert result.status == "ABORTED" and "kill_switch" in result.reason and fake.posts() == []


async def test_duplicate_execution_is_refused(env):
    fake = FakeFPL(START)
    assert (await run(env, fake))[0].status == "SUCCESS"
    n_posts = len(fake.posts())
    second, _ = await run(env, fake)
    assert second.status == "ABORTED" and "duplicate" in second.reason and len(fake.posts()) == n_posts


@pytest.mark.parametrize(("fake_kwargs", "check"), [
    ({"ft": 0}, "hit_cap"),                    # live FTs lower than planned and max hits 0
    ({"bank": 0, "price_116": 75}, "budget"),  # price rose since the plan
])
async def test_live_data_guardrails(env, fake_kwargs, check):
    await env[1].execute("UPDATE system_config SET max_hits_per_gw = 0")
    if check == "budget":
        await env[1].execute("UPDATE players SET now_cost = 75 WHERE player_id = 116")
    fake = FakeFPL(START, **fake_kwargs)
    result, notes = await run(env, fake)
    assert result.status == "ABORTED" and check in result.reason and fake.posts() == []
    assert notes.messages[-1][1] is True, "guardrail aborts are urgent alerts"


async def test_manual_changes_since_plan_abort(env):
    fake = FakeFPL([p if p != 114 else 116 for p in START])
    result, _ = await run(env, fake)
    assert result.status == "ABORTED" and "squad_parity" in result.reason


async def test_retry_after_partial_commit_skips_transfers(env):
    fake = FakeFPL(TARGET)      # transfers already applied by an earlier attempt
    result, _ = await run(env, fake)
    assert result.status == "SUCCESS" and fake.posts() == ["POST picks"]


async def test_ambiguous_transfer_error_is_resolved_by_reading_state(env):
    fake = FakeFPL(START)
    fake.transfers_status = 502   # applied, but the response was lost
    result, _ = await run(env, fake)
    assert result.status == "SUCCESS"


async def test_unconfirmed_lineup_fails_loudly(env):
    fake = FakeFPL(START)
    fake.apply_picks = False
    result, notes = await run(env, fake)
    assert result.status == "FAILED" and "captaincy" in result.reason and notes.messages[-1][1] is True


async def test_expired_token_is_refreshed(env):
    settings, pool, _ = env
    await TokenVault(settings, pool).store(TEAM, AuthTokens("OLD", "RT1", datetime(2020, 1, 1, tzinfo=timezone.utc)))
    fake = FakeFPL(START)
    result, _ = await run(env, fake, refresh=httpx.Response(200, json={"access_token": "NEW", "refresh_token": "RT2", "expires_in": 3600}))
    assert result.status == "SUCCESS"
    assert all(c[1] == "Bearer NEW" for c in fake.calls if c[0] == "GET")
    stored = await TokenVault(settings, pool).load(TEAM)
    assert (stored.access_token, stored.refresh_token) == ("NEW", "RT2")


async def test_rejected_refresh_aborts_with_alert(env):
    settings, pool, _ = env
    await TokenVault(settings, pool).store(TEAM, AuthTokens("OLD", "RT1", datetime(2020, 1, 1, tzinfo=timezone.utc)))
    fake = FakeFPL(START)
    result, notes = await run(env, fake, refresh=httpx.Response(400, json={"error": "invalid_grant"}))
    assert result.status == "ABORTED" and "refresh" in result.reason and fake.posts() == []
    assert notes.messages[-1][1] is True


async def test_concurrent_commit_is_locked_out(env):
    _, _, redis = env
    held = redis.lock("fpl_commit_lock:gw_3", timeout=30)
    await held.acquire(blocking=False)
    fake = FakeFPL(START)
    result, _ = await run(env, fake)
    await held.release()
    assert result.status == "ABORTED" and "in progress" in result.reason and fake.calls == []

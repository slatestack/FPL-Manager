"""Integration fixtures. Set TEST_DATABASE_URL (a disposable database: its public schema is
dropped and recreated per test) and optionally TEST_REDIS_URL."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator

import asyncpg
import httpx
import pytest
import respx

from app.config import Settings
from app.db.migrate import migrate
from app.db.pool import create_pool
from app.ingestion.fpl_client import FPLClient
from tests.conftest import load_fixture

TEST_DSN = os.environ.get("TEST_DATABASE_URL")
TEST_REDIS = os.environ.get("TEST_REDIS_URL")
FPL_BASE = "https://fpl.test/api"

pytestmark = pytest.mark.integration


def pytest_collection_modifyitems(items):
    for item in items:
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
            if not TEST_DSN:
                item.add_marker(pytest.mark.skip(reason="TEST_DATABASE_URL not set"))


@pytest.fixture
async def dsn() -> str:
    conn = await asyncpg.connect(TEST_DSN)
    try:
        await conn.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public;")
    finally:
        await conn.close()
    await migrate(TEST_DSN)
    return TEST_DSN


@pytest.fixture
async def pool(dsn) -> AsyncIterator[asyncpg.Pool]:
    p = await create_pool(dsn)
    yield p
    await p.close()


@pytest.fixture
def settings() -> Settings:
    return Settings(
        database_url=TEST_DSN or "",
        redis_url=TEST_REDIS or "redis://localhost:6379/15",
        api_base_url=FPL_BASE,
        team_id=4242,
        http_max_retries=1,
        understat_enabled=False,
    )


async def _nosleep(_: float) -> None:
    return None


@pytest.fixture
def fpl_api(bootstrap_payload, fixtures_payload):
    """respx router serving recorded FPL payloads; tests may mutate ``state`` between calls."""
    state = {"bootstrap": bootstrap_payload, "fixtures": fixtures_payload}
    with respx.mock(base_url=FPL_BASE, assert_all_called=False) as router:
        router.get("/bootstrap-static/").mock(side_effect=lambda req: httpx.Response(200, json=state["bootstrap"]))
        router.get("/fixtures/").mock(side_effect=lambda req: httpx.Response(200, json=state["fixtures"]))
        router.get("/element-summary/5/").mock(return_value=httpx.Response(200, json=load_fixture("element_summary_5.json")))
        router.get("/element-summary/3/").mock(return_value=httpx.Response(200, json=load_fixture("element_summary_3.json")))
        router.get(url__regex=r".*/element-summary/\d+/$").mock(return_value=httpx.Response(200, json={"history": []}))
        router.get("/entry/4242/history/").mock(return_value=httpx.Response(200, json={
            "current": [{"event": 1, "points": 60, "total_points": 60, "overall_rank": 200000, "bank": 5, "value": 1000,
                         "event_transfers": 0, "event_transfers_cost": 0, "points_on_bench": 4},
                        {"event": 2, "points": 60, "total_points": 120, "overall_rank": 150000, "bank": 5, "value": 1003,
                         "event_transfers": 1, "event_transfers_cost": 0, "points_on_bench": 2}],
            "past": [], "chips": []}))
        router.get("/entry/4242/").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": 4242, "name": "Expected Pointers", "player_first_name": "Test", "player_last_name": "Manager",
                    "current_event": 2, "summary_overall_points": 120, "summary_overall_rank": 150000,
                    "summary_event_points": 60, "last_deadline_bank": 5, "last_deadline_value": 1003,
                    "last_deadline_total_transfers": 1, "leagues": {},
                },
            )
        )
        yield state


@pytest.fixture
async def fpl(settings, fpl_api) -> AsyncIterator[FPLClient]:
    async with FPLClient(settings, sleep=_nosleep) as client:
        yield client

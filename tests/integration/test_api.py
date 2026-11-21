import httpx
import pytest

from app.api.main import app
from app.config import get_settings
from app.ingestion import pipeline
from app.ml.service import run_prediction
from tests.integration.test_predictions import NOW, _ingest_archive


@pytest.fixture
async def client(dsn, monkeypatch):
    monkeypatch.setenv("FPL_DATABASE_URL", dsn)
    monkeypatch.setenv("FPL_TEAM_ID", "4242")
    get_settings.cache_clear()
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
            yield c
    get_settings.cache_clear()


async def test_dashboard_page_and_assets_are_served(client):
    page = await client.get("/")
    assert page.status_code == 200 and "FPL Manager" in page.text
    for asset in ("/static/app.js", "/static/app.css"):
        assert (await client.get(asset)).status_code == 200


async def test_dashboard_payload(client, pool, fpl):
    await pipeline.ingest_core(pool, fpl)
    await pipeline.ingest_entry(pool, fpl, 4242)
    await _ingest_archive(pool)
    await run_prediction(pool, now=NOW)
    data = (await client.get("/api/v1/dashboard")).json()
    assert data["team"]["entry_name"] == "Expected Pointers"
    assert data["gameweeks"] == [3]
    haaland = next(p for p in data["players"] if p["name"] == "Haaland")
    assert haaland["xp"]["3"] > 0 and haaland["opp"]["3"] == ["WOL (H)"]
    assert data["optimization"] is None
    latest = (await client.get("/api/v1/predictions/latest", params={"position": "FWD"})).json()
    assert {p["position"] for p in latest["players"]} == {"FWD"}

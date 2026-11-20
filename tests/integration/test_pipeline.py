from datetime import datetime, timezone

import httpx
import pytest
import respx

from app.ingestion import pipeline
from app.ingestion.adapters.odds_api import OddsApiAdapter
from app.ingestion.adapters.understat import UnderstatAdapter
from app.ingestion.http import ResilientClient
from tests.conftest import load_fixture


async def test_core_ingestion_populates_reference_tables(pool, fpl):
    result = await pipeline.run_job(pool, "core", lambda: pipeline.ingest_core(pool, fpl))

    assert result.details["players"] == 10 and result.details["season"] == 2026
    assert await pool.fetchval("SELECT COUNT(*) FROM teams") == 4
    assert await pool.fetchval("SELECT COUNT(*) FROM players") == 10
    assert await pool.fetchval("SELECT COUNT(*) FROM fixtures") == 7
    assert await pool.fetchval("SELECT COUNT(*) FROM gameweeks") == 3
    assert await pool.fetchval("SELECT COUNT(*) FROM chip_definitions") == 4
    assert await pool.fetchval("SELECT COUNT(*) FROM raw_snapshots") == 2

    haaland = await pool.fetchrow("SELECT * FROM players WHERE player_id = 5")
    assert (haaland["position"], haaland["now_cost"], haaland["team_id"]) == ("FWD", 145, 2)
    run = await pool.fetchrow("SELECT * FROM ingestion_runs WHERE job='core'")
    assert run["status"] == "SUCCESS" and run["payload_hash"] == result.payload_hash


async def test_reingest_tracks_price_and_status_changes(pool, fpl, fpl_api):
    await pipeline.ingest_core(pool, fpl)
    # Identical payload: no spurious history rows, raw snapshot deduplicated.
    await pipeline.ingest_core(pool, fpl)
    assert await pool.fetchval("SELECT COUNT(*) FROM player_price_history") == 10
    assert await pool.fetchval("SELECT COUNT(*) FROM raw_snapshots") == 2

    haaland = next(e for e in fpl_api["bootstrap"]["elements"] if e["id"] == 5)
    haaland["now_cost"] = 146
    saka = next(e for e in fpl_api["bootstrap"]["elements"] if e["id"] == 3)
    saka.update(status="a", chance_of_playing_next_round=None, news="")
    await pipeline.ingest_core(pool, fpl)

    change = await pool.fetchrow("SELECT old_cost, new_cost FROM player_price_history WHERE player_id=5 ORDER BY id DESC LIMIT 1")
    assert (change["old_cost"], change["new_cost"]) == (145, 146)
    statuses = [r["status"] for r in await pool.fetch("SELECT status FROM player_status_history WHERE player_id=3 ORDER BY id")]
    assert statuses == ["d", "a"]
    assert await pool.fetchval("SELECT COUNT(*) FROM raw_snapshots WHERE source='bootstrap-static'") == 2


async def test_failed_job_is_audited(pool, settings):
    with respx.mock(base_url="https://fpl.test/api") as router:
        router.get("/bootstrap-static/").mock(return_value=httpx.Response(200, json={"events": []}))
        router.get("/fixtures/").mock(return_value=httpx.Response(200, json=[]))
        from app.ingestion.fpl_client import FPLClient
        async with FPLClient(settings) as client:
            with pytest.raises(Exception):
                await pipeline.run_job(pool, "core", lambda: pipeline.ingest_core(pool, client))
    run = await pool.fetchrow("SELECT status, error_message FROM ingestion_runs WHERE job='core'")
    assert run["status"] == "FAILED" and "ValidationError" in run["error_message"]
    assert await pool.fetchval("SELECT COUNT(*) FROM players") == 0


async def test_history_and_rolling_form(pool, fpl):
    await pipeline.ingest_core(pool, fpl)
    result = await pipeline.ingest_player_histories(pool, fpl, concurrency=3)
    # Only players with minutes > 0 are fetched (ids 9 and 10 have 0 minutes).
    assert result.details["players"] == 8 and result.details["failed"] == 0
    assert await pool.fetchval("SELECT COUNT(*) FROM player_gw_history") == 4

    form = await pool.fetchrow("SELECT * FROM player_rolling_form WHERE player_id=5")
    assert form["matches"] == 2 and form["minutes"] == 175
    assert float(form["xg_per90"]) == pytest.approx((1.10 + 1.65) * 90 / 175, abs=1e-3)


async def test_entry_snapshot(pool, fpl):
    await pipeline.ingest_entry(pool, fpl, 4242)
    row = await pool.fetchrow("SELECT * FROM manager_entries WHERE entry_id=4242")
    assert row["summary_overall_rank"] == 150000 and row["last_deadline_bank"] == 5


async def test_xg_enrichment_maps_players_and_respects_manual_overrides(pool, fpl):
    await pipeline.ingest_core(pool, fpl)
    with respx.mock:
        respx.get("https://understat.test/getLeagueData/EPL/2026").mock(
            return_value=httpx.Response(200, json=load_fixture("understat_league.json"))
        )
        async with ResilientClient(max_retries=0) as http:
            adapter = UnderstatAdapter(http, base_url="https://understat.test")
            result = await pipeline.ingest_xg(pool, adapter)
            assert result.details["matched"] == 4 and result.details["unmatched_sample"] == ["Some Loanee"]

            mapping = {r["external_id"]: r["player_id"] for r in await pool.fetch("SELECT * FROM player_xg_stats")}
            assert mapping == {"8260": 5, "7322": 3, "5555": 2, "1234": 8, "9999": None}

            # A manual override wins over the automatic matcher on subsequent runs.
            await pool.execute(
                "UPDATE player_id_map SET player_id=6, matched_by='manual' WHERE source='understat' AND external_id='7322'"
            )
            await pipeline.ingest_xg(pool, adapter)
    assert await pool.fetchval("SELECT player_id FROM player_xg_stats WHERE external_id='7322'") == 6
    assert await pool.fetchval("SELECT matched_by FROM player_id_map WHERE external_id='7322'") == "manual"


async def test_odds_enrichment(pool, fpl):
    await pipeline.ingest_core(pool, fpl)
    with respx.mock(base_url="https://odds.test/v4") as router:
        router.get("/sports/soccer_epl/odds").mock(return_value=httpx.Response(200, json=load_fixture("odds_api_epl.json")))
        router.get("/sports/soccer_epl/events/evt1/odds").mock(
            return_value=httpx.Response(200, json=load_fixture("odds_api_goalscorers_evt1.json"))
        )
        router.get(url__regex=r".*/events/evt[23]/odds").mock(return_value=httpx.Response(404))
        async with ResilientClient(max_retries=0) as http:
            adapter = OddsApiAdapter(http, api_key="k", base_url="https://odds.test/v4", include_goalscorers=True)
            result = await pipeline.ingest_odds(pool, adapter)

    # Liverpool v Everton has no FPL fixture in the test season.
    assert result.details["fixtures"] == 2 and result.details["unmatched_fixtures"] == ["Liverpool v Everton"]
    ars_tot = await pool.fetchrow("SELECT * FROM latest_fixture_odds WHERE fixture_id=5")
    assert ars_tot["lambda_home"] > ars_tot["lambda_away"]
    assert 0 < ars_tot["p_cs_home"] < 1 and ars_tot["p_cs_home"] > ars_tot["p_cs_away"]
    city = await pool.fetchrow("SELECT * FROM latest_fixture_odds WHERE fixture_id=6")
    assert city["p_over_2_5"] is None and city["p_cs_home"] > 0.4

    scorers = {r["player_id"]: r for r in await pool.fetch("SELECT * FROM player_goal_odds")}
    assert set(scorers) == {3, 8}  # Saka, Son; 'Unknown Kid' unmatched
    assert float(scorers[3]["median_decimal_odds"]) == 2.55

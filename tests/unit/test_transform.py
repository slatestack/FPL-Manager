from datetime import datetime, timezone
from decimal import Decimal

import pytest
from pydantic import ValidationError

from app.ingestion import schemas, transform


def test_player_rows_map_positions_and_skip_non_players(bootstrap_payload):
    bs = schemas.BootstrapStatic.model_validate(bootstrap_payload)
    rows = transform.player_rows(bs)
    by_id = {r["player_id"]: r for r in rows}

    assert 11 not in by_id, "assistant-manager element type must be skipped"
    assert len(rows) == 10
    assert {r["position"] for r in rows} == {"GKP", "DEF", "MID", "FWD"}
    assert by_id[5]["position"] == "FWD" and by_id[5]["now_cost"] == 145
    assert by_id[5]["expected_goals"] == Decimal("2.40")
    assert by_id[3]["status"] == "d" and by_id[3]["chance_of_playing"] == 75
    assert by_id[3]["news_added"] == datetime(2026, 11, 18, 10, 0, 0, 123456, tzinfo=timezone.utc)
    assert by_id[1]["news"] is None, "empty news string is stored as NULL"


def test_fixture_rows_keep_postponed_fixtures(fixtures_payload):
    rows = transform.fixture_rows([schemas.Fixture.model_validate(f) for f in fixtures_payload])
    postponed = next(r for r in rows if r["fixture_id"] == 7)
    assert postponed["gameweek_id"] is None and postponed["kickoff_time"] is None
    assert next(r for r in rows if r["fixture_id"] == 1)["team_h_score"] == 2


def test_season_start_year(bootstrap_payload):
    assert transform.season_start_year(schemas.BootstrapStatic.model_validate(bootstrap_payload)) == 2026


def test_gameweek_and_chip_rows(bootstrap_payload):
    bs = schemas.BootstrapStatic.model_validate(bootstrap_payload)
    gws = transform.gameweek_rows(bs)
    assert [g["gameweek_id"] for g in gws if g["is_next"]] == [3]
    assert {c["name"] for c in transform.chip_rows(bs)} == {"wildcard", "freehit", "bboost", "3xc"}


def test_missing_required_field_fails_validation(bootstrap_payload):
    del bootstrap_payload["elements"][0]["now_cost"]
    with pytest.raises(ValidationError):
        schemas.BootstrapStatic.model_validate(bootstrap_payload)

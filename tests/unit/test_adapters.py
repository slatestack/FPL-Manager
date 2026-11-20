import json

import httpx
import pytest
import respx

from app.ingestion.adapters import odds_api, understat
from app.ingestion.http import ResilientClient
from tests.conftest import load_fixture


def _hex_escape(text: str) -> str:
    # Mimic Understat: escape every non-alphanumeric UTF-8 byte as \xNN.
    return "".join(chr(b) if b < 128 and chr(b).isalnum() else f"\\x{b:02x}" for b in text.encode("utf-8"))


def test_understat_records_from_json():
    recs = understat.to_records(load_fixture("understat_league.json")["players"])
    haaland = next(r for r in recs if r.external_id == "8260")
    assert haaland.team_names == ("Manchester City",)
    assert haaland.npxg == 2.75 and haaland.minutes == 175 and haaland.shots == 9
    assert haaland.npxg_per90 == pytest.approx(2.75 * 90 / 175)


def test_understat_html_fallback_decodes_utf8_escapes():
    players = load_fixture("understat_league.json")["players"]
    html = f"<script>var playersData\t= JSON.parse('{_hex_escape(json.dumps(players, ensure_ascii=False))}');</script>"
    rows = understat.parse_league_html(html)
    assert any(r["player_name"] == "Gabriel Magalhães" for r in rows)


async def test_understat_adapter_falls_back_to_html():
    players = load_fixture("understat_league.json")["players"]
    html = f"var playersData = JSON.parse('{_hex_escape(json.dumps(players))}');"
    with respx.mock:
        respx.get("https://understat.test/getLeagueData/EPL/2026").mock(return_value=httpx.Response(404))
        respx.get("https://understat.test/league/EPL/2026").mock(return_value=httpx.Response(200, text=html))
        async with ResilientClient(max_retries=0) as http:
            recs = await understat.UnderstatAdapter(http, base_url="https://understat.test").fetch(2026)
    assert len(recs) == 5


def test_odds_parse_consensus_and_totals():
    events = load_fixture("odds_api_epl.json")
    rec = odds_api.parse_match_odds(events[0])
    assert rec is not None and rec.bookmaker_count == 2
    assert rec.p_home + rec.p_draw + rec.p_away == pytest.approx(1.0)
    assert rec.p_home > rec.p_away
    assert rec.p_over_2_5 is not None and 0.45 < rec.p_over_2_5 < 0.6
    assert odds_api.parse_match_odds(events[1]).p_over_2_5 is None


def test_goalscorer_parse_takes_median_per_player():
    recs = {r.player_name: r for r in odds_api.parse_goalscorer_odds(load_fixture("odds_api_goalscorers_evt1.json"))}
    assert recs["Bukayo Saka"].bookmaker_count == 2
    assert recs["Bukayo Saka"].median_decimal_odds == pytest.approx(2.55)
    assert recs["Son Heung-Min"].p_anytime_goal == pytest.approx(1 / 3.1)

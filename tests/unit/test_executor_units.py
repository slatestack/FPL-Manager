import base64
import json
from datetime import datetime, timedelta, timezone

import pytest

from app.executor.client import parse_my_team
from app.executor.guardrails import evaluate
from app.executor.payloads import pair_transfers, picks_payload, transfers_payload
from app.executor.vault import AuthTokens, parse_token_input

POS = {1: "GKP", 2: "GKP", 3: "DEF", 4: "DEF", 5: "DEF", 6: "DEF", 7: "DEF", 8: "MID", 9: "MID", 10: "MID",
       11: "MID", 12: "MID", 13: "FWD", 14: "FWD", 15: "FWD", 16: "FWD", 17: "DEF"}


def _jwt(exp: int) -> str:
    enc = lambda d: base64.urlsafe_b64encode(json.dumps(d).encode()).decode().rstrip("=")  # noqa: E731
    return f"{enc({'alg': 'none'})}.{enc({'exp': exp})}.sig"


def test_parse_browser_oidc_blob():
    blob = json.dumps({"id_token": "x", "access_token": "AT", "refresh_token": "RT", "token_type": "Bearer", "expires_at": 1795000000})
    t = parse_token_input(blob)
    assert (t.access_token, t.refresh_token) == ("AT", "RT")
    assert t.expires_at == datetime.fromtimestamp(1795000000, tz=timezone.utc)
    bare = parse_token_input(_jwt(1795000100))
    assert bare.refresh_token is None and bare.expires_at.timestamp() == 1795000100
    assert AuthTokens.from_json(t.to_json()) == t


def test_picks_payload_orders_xi_by_position_and_bench_by_priority():
    xi = [15, 3, 9, 1, 4, 5, 8, 10, 11, 13, 14]
    p = picks_payload(starting_xi=xi, bench=[2, 12, 6, 7], captain=13, vice_captain=9, chip="3xc", position=POS)
    assert p["chip"] == "3xc"
    order = [x["element"] for x in p["picks"]]
    assert order[0] == 1 and [POS[e] for e in order[:11]] == ["GKP"] + ["DEF"] * 3 + ["MID"] * 4 + ["FWD"] * 3
    assert order[11:] == [2, 12, 6, 7] and [x["position"] for x in p["picks"]] == list(range(1, 16))
    assert [x["element"] for x in p["picks"] if x["is_captain"]] == [13]
    with pytest.raises(ValueError):
        picks_payload(starting_xi=xi, bench=[12, 2, 6, 7], captain=13, vice_captain=9, chip=None, position=POS)


def test_transfers_payload_pairs_same_positions_with_live_prices():
    p = transfers_payload(entry=7, event=11, outs=[16, 7], ins=[17, 15], chip=None, position={**POS, 15: "FWD"},
                          selling_price={16: 71, 7: 44}, now_cost={17: 45, 15: 80})
    assert p == {"chip": None, "entry": 7, "event": 11, "transfers": [
        {"element_in": 17, "element_out": 7, "purchase_price": 45, "selling_price": 44},
        {"element_in": 15, "element_out": 16, "purchase_price": 80, "selling_price": 71}]}
    with pytest.raises(ValueError):
        pair_transfers([16], [17], POS)


def test_parse_my_team_free_transfers_and_chips():
    mt = parse_my_team({
        "picks": [{"element": 1, "position": 1, "selling_price": 50, "purchase_price": 50, "is_captain": True, "is_vice_captain": False}],
        "chips": [{"name": "3xc", "status_for_entry": "played"}, {"name": "3xc", "status_for_entry": "available"}],
        "transfers": {"cost": 4, "status": "cost", "limit": 3, "made": 1, "bank": 6, "value": 1000},
    })
    assert mt.free_transfers == 2 and mt.bank == 6 and mt.chip_status("3xc") == "available" and mt.captain == 1
    assert parse_my_team({"picks": [], "chips": [], "transfers": {"status": "unlimited", "limit": None}}).free_transfers is None


def _base(**over):
    now = datetime(2026, 11, 21, 10, 45, tzinfo=timezone.utc)
    kw = dict(auto_execute_enabled=True, respect_kill_switch=True, now=now, deadline=now + timedelta(minutes=15),
              cutoff=timedelta(minutes=3), plan_created_at=now - timedelta(minutes=10), plan_max_age=timedelta(minutes=45),
              plan_gameweek=11, next_gameweek=11, expected_squad={1, 2}, live_squad={1, 2}, outs=[2], ins=[3], chip=None,
              live_free_transfers=1, max_hits=1, live_bank=0, live_selling={1: 50, 2: 60}, now_cost={3: 60},
              duplicate_success=False, chip_status=None)
    kw.update(over)
    return evaluate(**kw)


@pytest.mark.parametrize(("override", "failing"), [
    ({}, []),
    ({"auto_execute_enabled": False}, ["kill_switch"]),
    ({"live_free_transfers": 0, "max_hits": 0}, ["hit_cap"]),
    ({"now_cost": {3: 61}}, ["budget"]),
    ({"duplicate_success": True}, ["duplicate"]),
    ({"live_squad": {1, 9}}, ["squad_parity"]),
    ({"now": datetime(2026, 11, 21, 10, 58, tzinfo=timezone.utc)}, ["deadline"]),
    ({"plan_created_at": datetime(2026, 11, 21, 9, 0, tzinfo=timezone.utc)}, ["plan_fresh"]),
    ({"chip": "bboost", "chip_status": "played"}, ["chip_available"]),
    ({"chip": "wildcard", "chip_status": "available", "live_free_transfers": 0, "max_hits": 0}, []),
])
def test_guardrails(override, failing):
    assert [c.name for c in _base(**override).failures] == failing


async def test_whatsapp_notifier_sends_and_never_leaks_the_key():
    import httpx
    import respx

    from app.notify import WhatsAppNotifier

    n = WhatsAppNotifier("+1 919 555 0100", "SECRETKEY", api_url="https://wa.test/whatsapp.php")
    with respx.mock() as router:
        route = router.get("https://wa.test/whatsapp.php").mock(return_value=httpx.Response(200, text="Message queued."))
        await n.send("GW12 plan", kill_button=True)
        q = dict(route.calls.last.request.url.params)
        assert q["phone"] == "+19195550100" and q["apikey"] == "SECRETKEY"
        assert q["text"].startswith("GW12 plan") and "Kill switch" in q["text"]
        router.get("https://wa.test/whatsapp.php").mock(return_value=httpx.Response(203, text="APIKey is invalid"))
        with pytest.raises(RuntimeError) as err:
            await n.send("x")
    assert "SECRETKEY" not in str(err.value)

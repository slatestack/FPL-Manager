import pytest

from app.optimizer.team_state import available_chips, derive_team_state, replay_free_transfers, selling_price


@pytest.mark.parametrize(("bought", "now", "sell"), [(50, 50, 50), (50, 53, 51), (50, 51, 50), (50, 54, 52), (84, 82, 82)])
def test_selling_price_keeps_half_of_rises_and_all_falls(bought, now, sell):
    assert selling_price(bought, now) == sell


def _gw(ev, n, cost=0):
    return {"event": ev, "event_transfers": n, "event_transfers_cost": cost}


def test_free_transfer_replay_matches_real_history():
    """The real 2026/27 history of the test account: hits in GW3 and GW8, wildcard in GW4 -> 3 FTs for GW11."""
    history = [_gw(1, 0), _gw(2, 0), _gw(3, 3, 4), _gw(4, 0), _gw(5, 0), _gw(6, 2), _gw(7, 0), _gw(8, 3, 4), _gw(9, 0), _gw(10, 0)]
    ft, warnings = replay_free_transfers(history, [{"name": "wildcard", "event": 4}], 11, 1)
    assert ft == 3 and warnings == []


def test_free_transfers_cap_at_five_and_flag_inconsistencies():
    history = [_gw(ev, 0) for ev in range(1, 12)]
    assert replay_free_transfers(history, [], 12, 1)[0] == 5
    _, warnings = replay_free_transfers([_gw(1, 0), _gw(2, 2, 0)], [], 3, 1)
    assert warnings == ["GW2: replay expected 4 hit points, FPL charged 0"]


def test_chip_windows():
    defs = [
        {"name": "wildcard", "start_event": 2, "stop_event": 19}, {"name": "wildcard", "start_event": 20, "stop_event": 38},
        {"name": "3xc", "start_event": 1, "stop_event": 19}, {"name": "3xc", "start_event": 20, "stop_event": 38},
    ]
    chips = available_chips(defs, [{"name": "wildcard", "event": 4}], 11)
    assert [(c.name, c.stop_event) for c in chips] == [("3xc", 19)]
    later = available_chips(defs, [{"name": "wildcard", "event": 4}], 21)
    assert sorted(c.name for c in later) == ["3xc", "wildcard"], "second-half chips are a fresh set"


def test_derive_team_state_prices():
    picks = {"picks": [{"element": i} for i in (1, 2)], "entry_history": {"bank": 7}}
    state = derive_team_state(
        entry_id=9, next_gameweek=5, entry={"started_event": 1},
        history={"current": [_gw(e, 0) for e in range(1, 5)], "chips": []},
        transfers=[{"element_in": 2, "element_in_cost": 60, "element_out": 3, "event": 3, "time": "t"}],
        picks=picks, chip_definitions=[], now_cost={1: 55, 2: 64}, first_gw_price={1: 50, 2: 58},
    )
    assert state.purchase_price == {1: 50, 2: 60} and state.selling_price == {1: 52, 2: 62}
    assert state.squad_value == 114 and state.bank == 7 and state.free_transfers == 4

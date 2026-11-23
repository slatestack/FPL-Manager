from collections import Counter

from app.optimizer.chip_forecast import best_lineup, forecast_chips
from app.optimizer.chips import evaluate_chips
from app.optimizer.model import solve
from app.optimizer.params import OptimizerParams
from app.optimizer.team_state import ChipWindow
from tests.unit.test_optimizer import PARAMS, league, squad_of, state_for

POS = {1: "GKP", 2: "GKP", **{i: "DEF" for i in range(3, 8)}, **{i: "MID" for i in range(8, 13)}, **{i: "FWD" for i in range(13, 16)}}


def test_best_lineup_is_legal_and_optimal():
    xp = {i: 1.0 for i in POS} | {13: 9.0, 14: 8.0, 15: 7.0, 3: 0.1, 4: 0.1}   # strong forwards, weak defence
    xi, bench = best_lineup(list(POS), xp, POS)
    c = Counter(POS[i] for i in xi)
    assert len(xi) == 11 and c["GKP"] == 1 and c["DEF"] >= 3 and c["MID"] >= 2 and c["FWD"] >= 1
    assert {13, 14, 15} <= set(xi) and c["DEF"] == 3, "plays 3 forwards and the minimum defence"
    assert POS[bench[0]] == "GKP" and len(bench) == 4


def _plan_and_state(chips, gw0=11):
    cand = league(T=3)
    cand.gameweeks = [gw0, gw0 + 1, gw0 + 2]
    state = state_for(cand, squad_of(cand), ft=1, bank=0, chips=())
    state.next_gameweek = gw0
    state.chips_available = chips
    plan = solve(cand, state, PARAMS, allow_transfers=False)
    return cand, state, plan


def _maps(cand, plan, gws, boost=None):
    xp, nf, pp = {}, {}, {}
    pos = dict(zip(cand.ids.tolist(), cand.position.tolist()))
    for i in cand.ids.tolist():
        for gw in gws:
            xp[(i, gw)], nf[(i, gw)], pp[(i, gw)] = 3.0, 1, 0.95
    for (i, gw), v in (boost or {}).items():
        xp[(i, gw)], nf[(i, gw)] = v, 2
    return xp, nf, pp, pos


def test_triple_captain_holds_for_a_later_double_gameweek():
    cand, state, plan = _plan_and_state([ChipWindow("3xc", 1, 19)])
    star = plan.gameweeks[-1].starting_xi[0]
    xp, nf, pp, pos = _maps(cand, plan, range(11, 21), {(star, 15): 12.0})
    (fc,) = forecast_chips(plan, state, xp, nf, pp, pos, list(range(11, 21)))
    assert fc.recommendation == "hold" and fc.best_gameweek == 15 and fc.best_value == 12.0
    assert next(w for w in fc.weeks if w.gameweek == 15).double


def test_uses_chip_in_best_week_once_window_is_fully_visible():
    cand, state, plan = _plan_and_state([ChipWindow("bboost", 1, 15)])
    bench_now = plan.first.bench
    xp, nf, pp, pos = _maps(cand, plan, range(11, 21), {(b, 11): 6.0 for b in bench_now})
    (fc,) = forecast_chips(plan, state, xp, nf, pp, pos, list(range(11, 21)))
    assert fc.window_visible and fc.best_gameweek == 11 and fc.recommendation == "use_now"


def test_defers_to_threshold_when_window_runs_past_forecast():
    cand, state, plan = _plan_and_state([ChipWindow("3xc", 1, 38)])
    xp, nf, pp, pos = _maps(cand, plan, range(11, 21))
    (fc,) = forecast_chips(plan, state, xp, nf, pp, pos, list(range(11, 21)))
    assert not fc.window_visible and fc.recommendation == "threshold"


def test_chip_decision_follows_the_forecast():
    cand, state, plan = _plan_and_state([ChipWindow("3xc", 1, 19)])
    cand.n_fixtures[:, 0] = 2          # this week looks like a double gameweek for everyone ...
    cand.xp[:, 0] = 7.0               # ... strong enough to pass the 12-point threshold with a captain
    base = solve(cand, state, OptimizerParams(time_limit_seconds=30))
    star = base.first.captain
    xp, nf, pp, pos = _maps(cand, base, range(11, 21), {(star, 11): 7.0, (star, 16): 20.0})
    fcs = forecast_chips(base, state, xp, nf, pp, pos, list(range(11, 21)))
    (ev,) = evaluate_chips(cand, state, OptimizerParams(time_limit_seconds=30), base, fcs)
    assert not ev.triggered and "GW16" in ev.reason, "a much better week later wins over this week's threshold"

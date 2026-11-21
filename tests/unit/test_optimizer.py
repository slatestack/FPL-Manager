"""Optimiser behaviour on a small synthetic league where the right answers are known."""

from dataclasses import replace

import numpy as np
import pytest

from app.optimizer.chips import choose_chip, evaluate_chips
from app.optimizer.model import Candidates, solve
from app.optimizer.params import OptimizerParams
from app.optimizer.team_state import ChipWindow, TeamState
from app.optimizer.validate import PlanViolation, validate_plan

POS = ["GKP"] * 4 + ["DEF"] * 10 + ["MID"] * 10 + ["FWD"] * 6
PARAMS = OptimizerParams(time_limit_seconds=30)


def league(T=3, seed=0):
    rng = np.random.default_rng(seed)
    n = len(POS)
    ids = np.arange(1, n + 1)
    team = np.array([i % 10 + 1 for i in range(n)])   # 10 clubs, 3 players each
    cost = np.where(np.array(POS) == "GKP", 45, 50).astype(float) + rng.integers(0, 30, n)
    xp = np.round(rng.uniform(1.5, 5.0, (n, T)), 2)
    return Candidates(ids=ids, position=np.array(POS), team=team, buy=cost, sell=cost.copy(), xp=xp,
                      p_play=np.full((n, T), 0.95), n_fixtures=np.ones((n, T), dtype=int), gameweeks=list(range(11, 11 + T)))


def squad_of(cand):
    """A valid starting squad: the first 2 GK, 5 DEF, 5 MID, 3 FWD by id."""
    picks = []
    for pos, k in (("GKP", 2), ("DEF", 5), ("MID", 5), ("FWD", 3)):
        picks += [int(i) for i in cand.ids[cand.position == pos][:k]]
    return picks


def state_for(cand, squad, ft=1, bank=0, chips=()):
    idx = cand.index()
    sell = {p: int(cand.sell[idx[p]]) for p in squad}
    return TeamState(entry_id=1, next_gameweek=cand.gameweeks[0], squad=squad, bank=bank, selling_price=sell,
                     purchase_price=dict(sell), free_transfers=ft, chips_available=[ChipWindow(c, 1, 19) for c in chips])


def test_plan_is_valid_and_lineup_is_best_available():
    cand = league()
    state = state_for(cand, squad_of(cand), ft=1, bank=100)
    plan = solve(cand, state, PARAMS)
    validate_plan(plan, cand, state, PARAMS)
    g = plan.first
    idx = cand.index()
    xp0 = {p: cand.xp[idx[p], 0] for p in g.squad}
    assert g.captain == max(g.starting_xi, key=xp0.get)
    outfield_bench = g.bench[1:]
    assert [xp0[p] for p in outfield_bench] == sorted((xp0[p] for p in outfield_bench), reverse=True)


def test_no_transfers_when_squad_is_already_best():
    cand = league()
    squad = squad_of(cand)
    cand.xp[:] = 0.5
    cand.xp[[cand.index()[p] for p in squad]] = 6.0
    plan = solve(cand, state_for(cand, squad, ft=1, bank=100), PARAMS)
    assert all(not g.transfers_in for g in plan.gameweeks)
    assert [g.free_transfers for g in plan.gameweeks] == [1, 2, 3], "unused transfers roll over"


def test_free_transfers_cap_at_five():
    cand = league(T=4)
    squad = squad_of(cand)
    cand.xp[:] = 0.5
    cand.xp[[cand.index()[p] for p in squad]] = 6.0
    plan = solve(cand, state_for(cand, squad, ft=4, bank=100), PARAMS)
    assert [g.free_transfers for g in plan.gameweeks] == [4, 5, 5, 5]


def test_hit_taken_only_when_worth_it_and_capped():
    cand = league()
    squad = squad_of(cand)
    idx = cand.index()
    cand.xp[:] = 2.0
    for p in squad:
        cand.xp[idx[p]] = 3.0
    star_mid, star_fwd = int(cand.ids[cand.position == "MID"][-1]), int(cand.ids[cand.position == "FWD"][-1])
    cand.xp[idx[star_mid]] = 9.0
    cand.xp[idx[star_fwd]] = 9.0
    state = state_for(cand, squad, ft=1, bank=500)
    plan = solve(cand, state, PARAMS)
    validate_plan(plan, cand, state, PARAMS)
    assert set(plan.first.transfers_in) == {star_mid, star_fwd} and plan.first.hits == 1
    no_hits = solve(cand, state, replace(PARAMS, max_hits_per_gw=0))
    assert len(no_hits.first.transfers_in) <= 1 and no_hits.first.hits == 0


def test_budget_and_club_limits_bind():
    cand = league()
    squad = squad_of(cand)
    idx = cand.index()
    cand.xp[:] = 2.0
    target = int(cand.ids[cand.position == "FWD"][-1])
    cand.xp[idx[target]] = 12.0
    cand.buy[idx[target]] = 200.0
    plan = solve(cand, state_for(cand, squad, ft=1, bank=0), PARAMS)
    assert target not in plan.first.squad, "unaffordable"
    rich = state_for(cand, squad, ft=1, bank=500)
    plan = solve(cand, rich, PARAMS)
    validate_plan(plan, cand, rich, PARAMS)
    assert target in plan.first.squad


def test_wildcard_and_free_hit():
    cand = league(seed=3)
    squad = squad_of(cand)
    state = state_for(cand, squad, ft=1, bank=50)
    wc = solve(cand, state, PARAMS, chip="wildcard")
    validate_plan(wc, cand, state, PARAMS)
    assert wc.first.hits == 0 and len(wc.first.transfers_in) > 2
    assert wc.gameweeks[1].free_transfers == 1, "wildcard keeps, but does not add, free transfers"
    fh = solve(cand, state, PARAMS, chip="freehit")
    validate_plan(fh, cand, state, PARAMS)
    g2 = fh.gameweeks[1]
    reverted = set(g2.squad) == (set(squad) - set(g2.transfers_out)) | set(g2.transfers_in)
    assert reverted and fh.first.hits == 0
    base = solve(cand, state, PARAMS)
    assert wc.objective >= base.objective - 1e-6 and fh.objective >= base.objective - 1e-6


def test_triple_captain_and_bench_boost_objectives():
    cand = league(seed=5)
    state = state_for(cand, squad_of(cand), ft=0, bank=0)
    base = solve(cand, state, PARAMS, allow_transfers=False)
    tc = solve(cand, state, PARAMS, chip="3xc", allow_transfers=False)
    idx = cand.index()
    assert tc.objective - base.objective == pytest.approx(cand.xp[idx[base.first.captain], 0], abs=1e-6)
    bb = solve(cand, state, PARAMS, chip="bboost", allow_transfers=False)
    bench_xp = sum(cand.xp[idx[p], 0] for p in base.first.bench)
    w = PARAMS.bench_weights
    eps_part = sum(PARAMS.bench_epsilon * w[k] * cand.xp[idx[p], 0] for k, p in enumerate(base.first.bench))
    assert bb.objective - base.objective == pytest.approx(bench_xp - eps_part, abs=1e-6)


def test_chip_triggers_follow_thresholds():
    cand = league(seed=7)
    squad = squad_of(cand)
    idx = cand.index()
    state = state_for(cand, squad, ft=1, bank=50, chips=("3xc", "bboost"))
    base = solve(cand, state, PARAMS)
    evals = {e.chip: e for e in evaluate_chips(cand, state, PARAMS, base)}
    assert not evals["3xc"].triggered, "no double gameweek"
    # Give the best player a double gameweek with a huge haul: Triple Captain should now fire.
    star = base.first.captain
    cand.n_fixtures[idx[star], 0] = 2
    cand.xp[idx[star], 0] = 14.0
    base = solve(cand, state, PARAMS)
    evals = evaluate_chips(cand, state, PARAMS, base)
    chosen = choose_chip(evals)
    assert chosen is not None and chosen.chip == "3xc" and chosen.gain >= 12.0


def test_validator_rejects_illegal_plans():
    cand = league()
    state = state_for(cand, squad_of(cand), ft=1, bank=100)
    plan = solve(cand, state, PARAMS)
    bad = plan.first
    bad.captain = bad.bench[1]
    with pytest.raises(PlanViolation, match="captaincy"):
        validate_plan(plan, cand, state, PARAMS)

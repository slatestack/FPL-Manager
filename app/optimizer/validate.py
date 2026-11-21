"""Independent check that a plan obeys every FPL rule (Phase 4 acceptance criteria)."""

from __future__ import annotations

from collections import Counter

from app.optimizer.model import SQUAD, XI_BOUNDS, Candidates, Plan
from app.optimizer.params import OptimizerParams
from app.optimizer.team_state import TeamState


class PlanViolation(AssertionError):
    pass


def validate_plan(plan: Plan, cand: Candidates, state: TeamState, params: OptimizerParams) -> None:
    idx = cand.index()
    pos = lambda p: cand.position[idx[p]]  # noqa: E731
    team = lambda p: cand.team[idx[p]]  # noqa: E731
    errors: list[str] = []

    def check(cond: bool, msg: str) -> None:
        if not cond:
            errors.append(msg)

    prev_squad, bank, ft = set(state.squad), state.bank, state.free_transfers
    for k, g in enumerate(plan.gameweeks):
        tag = f"GW{g.gameweek}"
        sq = set(g.squad)
        check(len(sq) == 15, f"{tag}: squad has {len(sq)} players")
        counts = Counter(pos(p) for p in sq)
        check(all(counts[p] == n for p, n in SQUAD.items()), f"{tag}: squad composition {dict(counts)}")
        clubs = Counter(team(p) for p in sq)
        check(max(clubs.values()) <= 3, f"{tag}: more than 3 players from one club {clubs.most_common(1)}")
        xi = set(g.starting_xi)
        check(len(xi) == 11 and xi <= sq, f"{tag}: starting XI invalid")
        xc = Counter(pos(p) for p in xi)
        check(all(lo <= xc[p] <= hi for p, (lo, hi) in XI_BOUNDS.items()), f"{tag}: formation {dict(xc)}")
        check(len(g.bench) == 4 and set(g.bench) == sq - xi, f"{tag}: bench {g.bench} != squad minus XI")
        check(pos(g.bench[0]) == "GKP" and all(pos(p) != "GKP" for p in g.bench[1:]), f"{tag}: bench order/GK slot")
        check(g.captain in xi and g.vice_captain in xi and g.captain != g.vice_captain, f"{tag}: captaincy")

        free_hit = g.chip == "freehit"
        base = set(state.squad) if (k == 1 and plan.chip == "freehit") else prev_squad
        if not free_hit:
            check(sq == (base - set(g.transfers_out)) | set(g.transfers_in), f"{tag}: transfers don't explain squad change")
            check(len(g.transfers_in) == len(g.transfers_out), f"{tag}: unbalanced transfers")
            start_bank = state.bank if (k == 1 and plan.chip == "freehit") else bank
            bank = start_bank + sum(cand.sell[idx[p]] for p in g.transfers_out) - sum(cand.buy[idx[p]] for p in g.transfers_in)
        else:
            budget = state.bank + sum(cand.sell[idx[p]] for p in state.squad)
            check(sum(cand.buy[idx[p]] for p in sq) <= budget + 1e-6, f"{tag}: Free Hit squad over budget")
        check(bank >= -1e-6, f"{tag}: bank negative ({bank})")

        n = len(g.transfers_in)
        check(g.free_transfers == ft, f"{tag}: plan says {g.free_transfers} FT, rules give {ft}")
        if g.chip in ("wildcard", "freehit"):
            check(g.hits == 0, f"{tag}: hits during {g.chip}")
        else:
            check(g.hits == max(0, n - ft), f"{tag}: hits {g.hits} != max(0, {n} - {ft})")
            check(g.hits <= params.max_hits_per_gw, f"{tag}: {g.hits} hits exceeds cap {params.max_hits_per_gw}")
            ft = min(params.max_free_transfers, max(0, ft - n) + 1)
        prev_squad = sq if not free_hit else set(state.squad)
    if errors:
        raise PlanViolation("; ".join(errors))

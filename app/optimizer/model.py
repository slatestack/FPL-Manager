"""Multi-period MILP (Section 6) built with OR-Tools.

Decision variables per player i and gameweek t follow Section 6.1 (squad x, starter s,
captain c, vice v, bench slot b_k, transfer in/out). Section 6.3's free-transfer rollover

    ft_{t+1} = min(5, max(1, ft_t - n_t + 1))

is not linear, so it is written as: paid hits h_t >= n_t - ft_t; unused FTs r_t = ft_t - n_t + h_t
(>= 0); a binary z_t forbids paying hits while FTs are left (h_t <= M z_t, r_t <= 5 (1 - z_t));
ft_{t+1} <= r_t + 1 and ft_{t+1} <= 5 (the objective pushes ft to its upper bound). With a
Wildcard or Free Hit in t, hits are 0 and ft_{t+1} = ft_t (FPL keeps banked transfers).

The budget is a money flow: bank_t = bank_{t-1} + sum(sell * out) - sum(buy * in) >= 0, using the
manager's real selling prices for current squad players.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp

from app.optimizer.params import OptimizerParams
from app.optimizer.team_state import TeamState

POSITIONS = ("GKP", "DEF", "MID", "FWD")
SQUAD = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
XI_BOUNDS = {"GKP": (1, 1), "DEF": (3, 5), "MID": (2, 5), "FWD": (1, 3)}
CHIPS = ("wildcard", "freehit", "bboost", "3xc")


class SolveError(RuntimeError):
    pass


@dataclass
class GameweekPlan:
    gameweek: int
    squad: list[int]
    starting_xi: list[int]
    bench: list[int]                 # [GK, outfield 1, outfield 2, outfield 3]
    captain: int
    vice_captain: int
    transfers_in: list[int]
    transfers_out: list[int]
    hits: int
    free_transfers: int              # available entering this gameweek
    bank: int
    chip: str | None
    xp: float                        # lineup xP incl. captain (and bench if Bench Boost)


@dataclass
class Plan:
    objective: float
    status: str
    solve_seconds: float
    gameweeks: list[GameweekPlan]
    chip: str | None
    n_variables: int
    n_constraints: int
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def first(self) -> GameweekPlan:
        return self.gameweeks[0]


@dataclass
class Candidates:
    """Players the solver may choose from. ``xp`` is (N, T); ``p_play`` likewise."""

    ids: np.ndarray
    position: np.ndarray
    team: np.ndarray
    buy: np.ndarray       # tenths of £m
    sell: np.ndarray
    xp: np.ndarray
    p_play: np.ndarray
    n_fixtures: np.ndarray
    gameweeks: list[int]

    @property
    def n(self) -> int:
        return len(self.ids)

    def index(self) -> dict[int, int]:
        return {int(p): i for i, p in enumerate(self.ids)}


def build_candidates(pred: pd.DataFrame, players: pd.DataFrame, state: TeamState, params: OptimizerParams) -> Candidates:
    """``pred``: player_id, target_gw, xp_total, prob_play, n_fixtures. ``players``: player_id, position, team_id, now_cost."""
    gws = sorted(pred["target_gw"].unique())
    xp = pred.pivot(index="player_id", columns="target_gw", values="xp_total").reindex(columns=gws).fillna(0.0).astype(float)
    pp = pred.pivot(index="player_id", columns="target_gw", values="prob_play").reindex(columns=gws).fillna(0.0).astype(float)
    nf = pred.pivot(index="player_id", columns="target_gw", values="n_fixtures").reindex(columns=gws).fillna(0).astype(int)
    info = players.set_index("player_id")
    disc = xp.to_numpy() @ (params.gamma ** np.arange(len(gws)))
    score = pd.Series(disc, index=xp.index)
    keep = set(state.squad)
    for pos, k in params.pool_per_position.items():
        ids = info.index[info["position"] == pos].intersection(score.index)
        s = score.loc[ids]
        keep |= set(s.nlargest(k).index)
        value = (s / info.loc[ids, "now_cost"].astype(float)).nlargest(params.pool_value_per_position)
        keep |= set(value.index)
    ids = np.array(sorted(keep))
    sub = info.loc[ids]
    buy = sub["now_cost"].to_numpy(dtype=float)
    sell = np.array([state.selling_price.get(int(p), int(c)) for p, c in zip(ids, buy)], dtype=float)
    return Candidates(
        ids=ids, position=sub["position"].to_numpy(), team=sub["team_id"].to_numpy(), buy=buy, sell=sell,
        xp=xp.reindex(ids).fillna(0.0).to_numpy(), p_play=pp.reindex(ids).fillna(0.0).to_numpy(),
        n_fixtures=nf.reindex(ids).fillna(0).to_numpy(), gameweeks=gws,
    )


def solve(
    cand: Candidates,
    state: TeamState,
    params: OptimizerParams,
    *,
    chip: str | None = None,
    allow_transfers: bool = True,
) -> Plan:
    """Solve the horizon plan; ``chip`` is played in the first gameweek (Section 6.4 counterfactuals)."""
    if chip not in (None, *CHIPS):
        raise ValueError(f"unknown chip {chip}")
    solver = pywraplp.Solver.CreateSolver(params.solver)
    if solver is None:
        raise SolveError(f"solver {params.solver} unavailable")
    N, T, gws = cand.n, len(cand.gameweeks), cand.gameweeks
    idx = cand.index()
    x0 = np.zeros(N)
    for p in state.squad:
        if p not in idx:
            raise SolveError(f"squad player {p} missing from candidates")
        x0[idx[p]] = 1
    gk = cand.position == "GKP"
    inf = solver.infinity()
    B = lambda name: solver.BoolVar(name)  # noqa: E731

    x = [[B(f"x_{i}_{t}") for t in range(T)] for i in range(N)]
    s = [[B(f"s_{i}_{t}") for t in range(T)] for i in range(N)]
    c = [[B(f"c_{i}_{t}") for t in range(T)] for i in range(N)]
    v = [[B(f"v_{i}_{t}") for t in range(T)] for i in range(N)]
    b = [[[B(f"b_{i}_{t}_{k}") for k in range(4)] for t in range(T)] for i in range(N)]
    pin = [[B(f"in_{i}_{t}") for t in range(T)] for i in range(N)]
    pout = [[B(f"out_{i}_{t}") for t in range(T)] for i in range(N)]
    h = [solver.IntVar(0, params.max_hits_per_gw, f"h_{t}") for t in range(T)]
    ft = [solver.IntVar(0, params.max_free_transfers, f"ft_{t}") for t in range(T + 1)]
    r = [solver.IntVar(0, params.max_free_transfers, f"r_{t}") for t in range(T)]
    z = [B(f"z_{t}") for t in range(T)]
    bank = [solver.NumVar(0, inf, f"bank_{t}") for t in range(T)]

    C = solver.Add
    C(ft[0] == state.free_transfers)
    for t in range(T):
        chip_t = chip if t == 0 else None
        free_moves = chip_t in ("wildcard", "freehit")
        C(sum(x[i][t] for i in range(N)) == 15)
        for pos in POSITIONS:
            members = [i for i in range(N) if cand.position[i] == pos]
            C(sum(x[i][t] for i in members) == SQUAD[pos])
            lo, hi = XI_BOUNDS[pos]
            C(sum(s[i][t] for i in members) >= lo)
            C(sum(s[i][t] for i in members) <= hi)
        for club in np.unique(cand.team):
            C(sum(x[i][t] for i in range(N) if cand.team[i] == club) <= 3)
        C(sum(s[i][t] for i in range(N)) == 11)
        C(sum(c[i][t] for i in range(N)) == 1)
        C(sum(v[i][t] for i in range(N)) == 1)
        for k in range(4):
            C(sum(b[i][t][k] for i in range(N)) == 1)
        for i in range(N):
            C(s[i][t] + sum(b[i][t][k] for k in range(4)) == x[i][t])
            C(c[i][t] <= s[i][t])
            C(v[i][t] <= s[i][t])
            C(c[i][t] + v[i][t] <= 1)
            if gk[i]:
                for k in (1, 2, 3):
                    C(b[i][t][k] == 0)
            else:
                C(b[i][t][0] == 0)
            C(pin[i][t] + pout[i][t] <= 1)
            # Squad continuity; a Free Hit squad (t=0) reverts to the original squad at t=1.
            if t == 0:
                prev = x0[i]
            elif t == 1 and chip == "freehit":
                prev = x0[i]
            else:
                prev = x[i][t - 1]
            if not (t == 0 and chip == "freehit"):
                C(x[i][t] == prev + pin[i][t] - pout[i][t])
            else:
                # Free Hit: any squad affordable from the current selling value + bank.
                C(pin[i][t] >= x[i][t] - x0[i])
                C(pout[i][t] >= x0[i] - x[i][t])
        n_t = sum(pin[i][t] for i in range(N))
        if not allow_transfers:
            C(n_t == 0)
        # Money.
        if t == 0 and chip == "freehit":
            C(sum(cand.buy[i] * x[i][t] for i in range(N)) <= state.bank + float(np.dot(cand.sell, x0)))
            C(bank[t] == state.bank)
        else:
            prev_bank = state.bank if t == 0 or (t == 1 and chip == "freehit") else bank[t - 1]
            C(bank[t] == prev_bank + sum(cand.sell[i] * pout[i][t] - cand.buy[i] * pin[i][t] for i in range(N)))
        # Free transfers and hits.
        if free_moves:
            C(h[t] == 0)
            C(ft[t + 1] == ft[t])
            C(r[t] == 0)
            C(z[t] == 0)
        else:
            C(h[t] >= n_t - ft[t])
            C(r[t] == ft[t] - n_t + h[t])
            C(h[t] <= params.max_hits_per_gw * z[t])
            C(r[t] <= params.max_free_transfers * (1 - z[t]))
            C(ft[t + 1] <= r[t] + 1)

    obj = solver.Objective()
    w = params.bench_weights
    for t in range(T):
        g = params.gamma**t
        cap_mult = 2.0 if (chip == "3xc" and t == 0) else 1.0
        bench_full = chip == "bboost" and t == 0
        for i in range(N):
            xp = float(cand.xp[i, t])
            obj.SetCoefficient(s[i][t], g * xp)
            obj.SetCoefficient(c[i][t], g * xp * cap_mult)
            obj.SetCoefficient(v[i][t], g * xp * params.vice_weight)
            for k in range(4):
                obj.SetCoefficient(b[i][t][k], g * xp * (1.0 if bench_full else params.bench_epsilon * w[k]))
        obj.SetCoefficient(h[t], -g * params.hit_cost)
    obj.SetCoefficient(ft[T], params.ft_terminal_value * params.gamma**T)
    for t in range(1, T):  # tie-break so reported FT counts sit at their (rule-implied) maximum
        obj.SetCoefficient(ft[t], 1e-3)
    obj.SetMaximization()

    solver.SetTimeLimit(params.time_limit_seconds * 1000)
    mp = pywraplp.MPSolverParameters()
    mp.SetDoubleParam(pywraplp.MPSolverParameters.RELATIVE_MIP_GAP, params.relative_gap)
    status = solver.Solve(mp)
    names = {pywraplp.Solver.OPTIMAL: "OPTIMAL", pywraplp.Solver.FEASIBLE: "FEASIBLE"}
    if status not in names:
        raise SolveError(f"solver returned status {status} (chip={chip})")

    val = lambda var: var.solution_value()  # noqa: E731
    plans = []
    for t in range(T):
        sq = [int(cand.ids[i]) for i in range(N) if val(x[i][t]) > 0.5]
        xi = [int(cand.ids[i]) for i in range(N) if val(s[i][t]) > 0.5]
        bench = [next(int(cand.ids[i]) for i in range(N) if val(b[i][t][k]) > 0.5) for k in range(4)]
        cap = next(int(cand.ids[i]) for i in range(N) if val(c[i][t]) > 0.5)
        vice = next(int(cand.ids[i]) for i in range(N) if val(v[i][t]) > 0.5)
        ins = [int(cand.ids[i]) for i in range(N) if val(pin[i][t]) > 0.5]
        outs = [int(cand.ids[i]) for i in range(N) if val(pout[i][t]) > 0.5]
        xp_t = sum(cand.xp[idx[p], t] for p in xi) + cand.xp[idx[cap], t] * (2.0 if chip == "3xc" and t == 0 else 1.0)
        if chip == "bboost" and t == 0:
            xp_t += sum(cand.xp[idx[p], t] for p in bench)
        plans.append(
            GameweekPlan(
                gameweek=int(gws[t]), squad=sorted(sq), starting_xi=xi, bench=bench, captain=cap, vice_captain=vice,
                transfers_in=ins, transfers_out=outs, hits=int(round(val(h[t]))), free_transfers=int(round(val(ft[t]))),
                bank=int(round(val(bank[t]))), chip=chip if t == 0 else None, xp=float(xp_t),
            )
        )
    return Plan(
        objective=float(solver.Objective().Value()), status=names[status], solve_seconds=solver.WallTime() / 1000.0,
        gameweeks=plans, chip=chip, n_variables=solver.NumVariables(), n_constraints=solver.NumConstraints(),
    )

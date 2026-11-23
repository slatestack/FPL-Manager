"""Section 6.4: evaluate every available chip as a counterfactual solve and apply the trigger rules.

Only chips playable in the next gameweek are considered (one chip per gameweek). Thresholds are
scaled by ``system_config.chip_aggression_profile``. A chip in the final gameweek of its window
is used whenever it adds value at all (when ``use_expiring_chips``), since holding it is worth
nothing after that.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.optimizer.model import Candidates, Plan, SolveError, solve
from app.optimizer.params import OptimizerParams
from app.optimizer.team_state import TeamState


@dataclass
class ChipEvaluation:
    chip: str
    gain: float
    triggered: bool
    reason: str
    expiring: bool
    details: dict[str, Any] = field(default_factory=dict)
    plan: Plan | None = None

    def summary(self) -> dict[str, Any]:
        return {"chip": self.chip, "gain": round(self.gain, 2), "triggered": self.triggered, "reason": self.reason,
                "expiring": self.expiring, **self.details}


def _unavailable(cand: Candidates, squad: list[int], t: int = 0) -> int:
    idx = cand.index()
    return sum(1 for p in squad if cand.p_play[idx[p], t] < 0.5)


def evaluate_chips(cand: Candidates, state: TeamState, params: OptimizerParams, base: Plan,
                   forecasts: list | None = None) -> list[ChipEvaluation]:
    """``forecasts`` (from chip_forecast) adds timing: hold when a later week is clearly better,
    use when this is the best week of a fully forecast window."""
    timing = {f.chip: f for f in (forecasts or [])}
    scale = params.threshold_scale
    idx = cand.index()
    out: list[ChipEvaluation] = []
    for window in state.chips_available:
        chip = window.name
        expiring = window.stop_event == state.next_gameweek
        try:
            plan = solve(cand, state, params, chip=chip)
        except SolveError as exc:
            out.append(ChipEvaluation(chip, 0.0, False, f"solve failed: {exc}", expiring))
            continue
        gain = plan.objective - base.objective
        g = plan.first
        details: dict[str, Any] = {}
        if chip == "3xc":
            dgw = int(cand.n_fixtures[idx[g.captain], 0]) >= 2
            details = {"captain": g.captain, "captain_double_gw": dgw}
            ok = dgw and gain >= params.tc_min_gain * scale
            reason = f"captain {'has' if dgw else 'has no'} double GW, gain {gain:.1f} vs {params.tc_min_gain * scale:.1f}"
        elif chip == "bboost":
            bench_xp = float(sum(cand.xp[idx[p], 0] for p in g.bench))
            min_play = float(min(cand.p_play[idx[p], 0] for p in g.bench))
            details = {"bench_xp": round(bench_xp, 2), "bench_min_p_play": round(min_play, 3)}
            ok = min_play >= params.bb_min_bench_play and bench_xp >= params.bb_min_bench_xp * scale
            reason = f"bench xP {bench_xp:.1f} vs {params.bb_min_bench_xp * scale:.1f}, min P(play) {min_play:.2f}"
        elif chip == "freehit":
            fielded = int(sum(1 for p in state.squad if cand.n_fixtures[idx[p], 0] > 0 and cand.p_play[idx[p], 0] >= 0.5))
            details = {"current_squad_fielded": fielded}
            ok = fielded < params.fh_max_fielded and gain >= params.fh_min_gain * scale
            reason = f"{fielded} squad players fielded (Free Hit needs < {params.fh_max_fielded}), gain {gain:.1f} vs {params.fh_min_gain * scale:.1f}"
        else:  # wildcard
            unavailable = _unavailable(cand, state.squad)
            details = {"unavailable_assets": unavailable}
            ok = unavailable >= params.wc_min_unavailable and gain >= params.wc_min_gain * scale
            reason = f"{unavailable} unavailable assets (>= {params.wc_min_unavailable} needed), gain {gain:.1f} vs {params.wc_min_gain * scale:.1f}"
        fc = timing.get(chip)
        if fc is not None and fc.recommendation == "hold":
            ok, reason = False, f"forecast: {fc.reason}"
        elif fc is not None and fc.recommendation == "use_now" and gain > 0:
            ok, reason = True, f"forecast: {fc.reason}"
        if fc is not None:
            details["forecast_best_gameweek"] = fc.best_gameweek
        if not ok and expiring and params.use_expiring_chips and gain > 0:
            ok, reason = True, f"expires this gameweek; gain {gain:.1f} > 0"
        out.append(ChipEvaluation(chip, float(gain), bool(ok), reason, expiring, details, plan))
    return out


def choose_chip(evals: list[ChipEvaluation]) -> ChipEvaluation | None:
    triggered = [e for e in evals if e.triggered]
    return max(triggered, key=lambda e: e.gain) if triggered else None

"""Forecast the value of Triple Captain and Bench Boost in every remaining week of their window.

For each upcoming gameweek the squad is the one the optimiser plans to hold that week (beyond
its horizon, the last planned squad). From that squad the best legal lineup is picked, and:

* Triple Captain is worth the captain's expected points once more (the extra multiple);
* Bench Boost is worth the expected points of the four bench players.

Timing rule (used by the chip decision):

* hold if a later week in the window is clearly better than this one;
* once the remaining window is fully forecast and this week is the best of it, use the chip
  even without a double gameweek: an unused chip is worth nothing after its window closes;
* otherwise fall back to the Section 6.4 thresholds.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from app.optimizer.model import Plan
from app.optimizer.team_state import TeamState

FORECAST_CHIPS = ("3xc", "bboost")
FORMATIONS = [(d, m, f) for d in range(3, 6) for m in range(2, 6) for f in range(1, 4) if d + m + f == 10]
CLEARLY_BETTER = 1.0          # a later week must beat this week by this many points (and 5%) to hold


@dataclass
class ChipWeek:
    gameweek: int
    value: float
    detail: str                 # captain / bench summary
    double: bool                # captain (TC) or a bench player (BB) has two fixtures
    blank_players: int          # squad players without a fixture that week


@dataclass
class ChipForecast:
    chip: str
    window_end: int
    weeks: list[ChipWeek] = field(default_factory=list)
    best_gameweek: int | None = None
    best_value: float = 0.0
    now_value: float = 0.0
    window_visible: bool = False
    recommendation: str = "threshold"   # "use_now" | "hold" | "threshold" (defer to Section 6.4)
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def best_lineup(squad: Sequence[int], xp: Mapping[int, float], position: Mapping[int, str]) -> tuple[list[int], list[int]]:
    """Exact best starting XI (legal FPL formation) and bench in substitution order."""
    by_pos = {p: sorted((i for i in squad if position[i] == p), key=lambda i: -xp.get(i, 0.0)) for p in ("GKP", "DEF", "MID", "FWD")}
    best, best_xi = -1.0, []
    for d, m, f in FORMATIONS:
        if len(by_pos["DEF"]) < d or len(by_pos["MID"]) < m or len(by_pos["FWD"]) < f:
            continue
        xi = by_pos["GKP"][:1] + by_pos["DEF"][:d] + by_pos["MID"][:m] + by_pos["FWD"][:f]
        total = sum(xp.get(i, 0.0) for i in xi)
        if total > best:
            best, best_xi = total, xi
    rest = [i for i in squad if i not in best_xi]
    bench = [i for i in rest if position[i] == "GKP"] + sorted((i for i in rest if position[i] != "GKP"), key=lambda i: -xp.get(i, 0.0))
    return best_xi, bench


def forecast_chips(
    plan: Plan,
    state: TeamState,
    xp: Mapping[tuple[int, int], float],
    n_fixtures: Mapping[tuple[int, int], int],
    p_play: Mapping[tuple[int, int], float],
    position: Mapping[int, str],
    gameweeks: Sequence[int],
    names: Mapping[int, str] | None = None,
) -> list[ChipForecast]:
    """``xp`` etc. are keyed by (player, gameweek) and cover ``gameweeks`` (possibly beyond the plan)."""
    names = names or {}
    planned = {g.gameweek: g.squad for g in plan.gameweeks}
    last_squad = plan.gameweeks[-1].squad
    out = []
    for window in state.chips_available:
        if window.name not in FORECAST_CHIPS:
            continue
        fc = ChipForecast(chip=window.name, window_end=window.stop_event)
        weeks = [int(g) for g in gameweeks if state.next_gameweek <= g <= window.stop_event]   # plain ints: stored as JSON
        for gw in weeks:
            squad = planned.get(gw, last_squad)
            wk = {i: float(xp.get((i, gw), 0.0)) for i in squad}
            xi, bench = best_lineup(squad, wk, position)
            blanks = sum(1 for i in squad if n_fixtures.get((i, gw), 0) == 0)
            if window.name == "3xc":
                cap = max(xi, key=lambda i: wk[i])
                fc.weeks.append(ChipWeek(gw, round(wk[cap], 2), f"captain {names.get(cap, cap)}",
                                         bool(n_fixtures.get((cap, gw), 0) >= 2), int(blanks)))
            else:
                value = sum(wk[i] for i in bench)
                shaky = [names.get(i, i) for i in bench if p_play.get((i, gw), 0.0) < 0.85]
                detail = "bench " + ", ".join(str(names.get(i, i)) for i in bench)
                if shaky:
                    detail += f" (doubtful: {', '.join(map(str, shaky))})"
                fc.weeks.append(ChipWeek(gw, round(float(value), 2), detail,
                                         bool(any(n_fixtures.get((i, gw), 0) >= 2 for i in bench)), int(blanks)))
        if not fc.weeks:
            continue
        best = max(fc.weeks, key=lambda w: w.value)
        fc.best_gameweek, fc.best_value = best.gameweek, best.value
        fc.now_value = fc.weeks[0].value if fc.weeks[0].gameweek == state.next_gameweek else 0.0
        fc.window_visible = max(gameweeks) >= window.stop_event
        later_better = best.gameweek != state.next_gameweek and best.value > max(fc.now_value * 1.05, fc.now_value + CLEARLY_BETTER)
        if later_better:
            fc.recommendation = "hold"
            fc.reason = f"GW{best.gameweek} is forecast better ({best.value:.1f} vs {fc.now_value:.1f} now)"
        elif fc.window_visible and fc.now_value >= best.value - 1e-9 and fc.now_value > 0:
            fc.recommendation = "use_now"
            fc.reason = (f"best remaining week of the window (ends GW{window.stop_event}): "
                         f"{fc.now_value:.1f} expected points")
        else:
            fc.recommendation = "threshold"
            fc.reason = ("no clearly better week in the forecast, but the window runs past it; "
                         "the standard threshold decides this week")
        out.append(fc)
    return out

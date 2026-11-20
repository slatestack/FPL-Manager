"""Bookmaker odds -> probabilities.

* Proportional overround removal turns a bookmaker's decimal prices into a fair
  probability distribution.
* An independent-Poisson score model is fitted to the consensus 1X2 (and Over/Under
  2.5 when available) probabilities to recover each side's implied expected goals
  ``lambda``; the clean-sheet probability for a team is then ``exp(-lambda_opponent)``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import median

MAX_GOALS = 12


def implied_probabilities(decimal_odds: Sequence[float]) -> list[float]:
    """De-vig a complete market (e.g. home/draw/away) by proportional normalisation."""
    if not decimal_odds or any(o <= 1.0 for o in decimal_odds):
        raise ValueError(f"decimal odds must all be > 1.0: {decimal_odds}")
    raw = [1.0 / o for o in decimal_odds]
    book = sum(raw)
    return [r / book for r in raw]


def overround(decimal_odds: Sequence[float]) -> float:
    return sum(1.0 / o for o in decimal_odds) - 1.0


def poisson_pmf(lam: float, max_goals: int = MAX_GOALS) -> list[float]:
    out = [math.exp(-lam)]
    for k in range(1, max_goals + 1):
        out.append(out[-1] * lam / k)
    return out


def outcome_probabilities(lambda_home: float, lambda_away: float) -> tuple[float, float, float, float]:
    """Return (P(home), P(draw), P(away), P(over 2.5)) under independent Poisson goals."""
    ph = poisson_pmf(lambda_home)
    pa = poisson_pmf(lambda_away)
    home = draw = away = 0.0
    for i, pi in enumerate(ph):
        for j, pj in enumerate(pa):
            p = pi * pj
            if i > j:
                home += p
            elif i == j:
                draw += p
            else:
                away += p
    total = lambda_home + lambda_away
    under = math.exp(-total) * (1 + total + total * total / 2)
    return home, draw, away, 1.0 - under


@dataclass(frozen=True)
class GoalRateFit:
    lambda_home: float
    lambda_away: float
    error: float

    @property
    def p_cs_home(self) -> float:
        return math.exp(-self.lambda_away)

    @property
    def p_cs_away(self) -> float:
        return math.exp(-self.lambda_home)


def fit_goal_rates(p_home: float, p_draw: float, p_away: float, p_over_2_5: float | None = None) -> GoalRateFit:
    """Least-squares fit of (lambda_home, lambda_away) via deterministic coarse-to-fine grid search."""

    def loss(lh: float, la: float) -> float:
        h, d, a, o = outcome_probabilities(lh, la)
        err = (h - p_home) ** 2 + (d - p_draw) ** 2 + (a - p_away) ** 2
        if p_over_2_5 is not None:
            err += (o - p_over_2_5) ** 2
        return err

    lo, hi, step = 0.05, 5.0, 0.25
    best = (float("inf"), 1.3, 1.1)
    center_h = center_a = None
    for _ in range(5):
        if center_h is None:
            grid_h = _frange(lo, hi, step)
            grid_a = grid_h
        else:
            grid_h = _frange(max(lo, center_h - 2 * step), min(hi, center_h + 2 * step), step)
            grid_a = _frange(max(lo, center_a - 2 * step), min(hi, center_a + 2 * step), step)
        for lh in grid_h:
            for la in grid_a:
                e = loss(lh, la)
                if e < best[0]:
                    best = (e, lh, la)
        center_h, center_a = best[1], best[2]
        step /= 4
    return GoalRateFit(lambda_home=round(best[1], 4), lambda_away=round(best[2], 4), error=best[0])


def _frange(start: float, stop: float, step: float) -> list[float]:
    n = int(round((stop - start) / step))
    return [start + i * step for i in range(n + 1)]


def consensus(values: Sequence[Sequence[float]]) -> list[float]:
    """Average per-bookmaker fair probability vectors and renormalise."""
    if not values:
        raise ValueError("no bookmaker probabilities to combine")
    n = len(values[0])
    avg = [sum(v[i] for v in values) / len(values) for i in range(n)]
    s = sum(avg)
    return [a / s for a in avg]


def median_price(prices: Sequence[float]) -> float:
    return float(median(prices))

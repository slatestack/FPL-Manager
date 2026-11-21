"""Team attack/defence ratings: weighted Poisson regression with ridge priors.

    log E[goals_t vs o] = base + home * is_home + att[t] - def[o]

fitted to a blend of team xG and actual goals (a non-integer target is fine for the Poisson
quasi-likelihood), with exponential time decay and an extra down-weight for past seasons.
Teams are keyed by stable FPL team ``code``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from app.ml.params import ModelParams
from app.ml.weights import recency_weights


@dataclass
class TeamRatings:
    base: float
    home: float
    att: dict[int, float] = field(default_factory=dict)
    dfn: dict[int, float] = field(default_factory=dict)
    avg_rate: float = 1.35

    def rates(self, home_code: int, away_code: int) -> tuple[float, float]:
        ah, dh = self.att.get(home_code, 0.0), self.dfn.get(home_code, 0.0)
        aa, da = self.att.get(away_code, 0.0), self.dfn.get(away_code, 0.0)
        return (
            float(np.exp(self.base + self.home + ah - da)),
            float(np.exp(self.base + aa - dh)),
        )


def promoted_teams(fixtures: pd.DataFrame, current_idx: int) -> set[int]:
    cur = fixtures[fixtures["season_idx"] == current_idx]
    prev = fixtures[fixtures["season_idx"] == current_idx - 1]
    if prev.empty:
        return set()
    cur_teams = set(cur["home_code"]) | set(cur["away_code"])
    return cur_teams - (set(prev["home_code"]) | set(prev["away_code"]))


def fit_team_ratings(
    fixtures: pd.DataFrame, as_of: pd.Timestamp, current_idx: int, params: ModelParams, extra_teams: set[int] = frozenset()
) -> TeamRatings:
    done = fixtures[(fixtures["kickoff"] < as_of) & fixtures["home_goals"].notna()]
    done = done[done["season_idx"] >= current_idx - 2]
    w = recency_weights(done["kickoff"], done["season_idx"], as_of, current_idx,
                        params.team_half_life_days, params.team_prev_season_factor)

    def target(xg: pd.Series, goals: pd.Series) -> np.ndarray:
        xg_v, g_v = xg.to_numpy(dtype=float), goals.to_numpy(dtype=float)
        return np.where(np.isnan(xg_v), g_v, params.team_xg_weight * xg_v + (1 - params.team_xg_weight) * g_v)

    team = np.concatenate([done["home_code"].to_numpy(), done["away_code"].to_numpy()])
    opp = np.concatenate([done["away_code"].to_numpy(), done["home_code"].to_numpy()])
    is_home = np.concatenate([np.ones(len(done)), np.zeros(len(done))])
    y = np.concatenate([target(done["home_xg"], done["home_goals"]), target(done["away_xg"], done["away_goals"])])
    wt = np.concatenate([w, w])

    codes = sorted(set(team) | set(opp) | set(extra_teams) | (set(fixtures.loc[fixtures["season_idx"] == current_idx, "home_code"])))
    idx = {c: i for i, c in enumerate(codes)}
    n = len(codes)
    ti = np.array([idx[c] for c in team], dtype=int)
    oi = np.array([idx[c] for c in opp], dtype=int)

    promoted = promoted_teams(fixtures, current_idx)
    prior = np.array([params.promoted_prior if c in promoted else 0.0 for c in codes])
    r = params.team_ridge

    def f(theta: np.ndarray) -> tuple[float, np.ndarray]:
        base, home = theta[0], theta[1]
        att, dfn = theta[2 : 2 + n], theta[2 + n :]
        eta = base + home * is_home + att[ti] - dfn[oi]
        mu = np.exp(eta)
        nll = np.sum(wt * (mu - y * eta)) + r * np.sum((att - prior) ** 2) + r * np.sum((dfn - prior) ** 2)
        g = wt * (mu - y)
        grad = np.zeros_like(theta)
        grad[0] = g.sum()
        grad[1] = (g * is_home).sum()
        np.add.at(grad, 2 + ti, g)
        np.add.at(grad, 2 + n + oi, -g)
        grad[2 : 2 + n] += 2 * r * (att - prior)
        grad[2 + n :] += 2 * r * (dfn - prior)
        return nll, grad

    theta0 = np.zeros(2 + 2 * n)
    theta0[0] = np.log(max(np.average(y, weights=wt) if len(y) else 1.35, 0.1))
    res = minimize(f, theta0, jac=True, method="L-BFGS-B")
    th = res.x
    ratings = TeamRatings(
        base=float(th[0]),
        home=float(th[1]),
        att={c: float(th[2 + i]) for c, i in idx.items()},
        dfn={c: float(th[2 + n + i]) for c, i in idx.items()},
    )
    ratings.avg_rate = float(np.exp(ratings.base + ratings.home / 2))
    return ratings

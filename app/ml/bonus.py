"""Bonus points.

A LightGBM regressor learns E[bonus | realised match stats] (goals, assists, clean sheet,
saves, defensive actions, cards, scoreline, the player's baseline BPS rate). Because bonus is
non-linear in those stats (3/2/1 for the top three BPS in the match), the expected bonus for
a future fixture is computed by averaging the model over simulated matches drawn from the
same distributions the rest of the engine uses, with a fixed seed for determinism.
"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

from app.ml.minutes import POS_CODE, _lgb_params
from app.ml.params import ModelParams

FEATURES = [
    "pos", "minutes", "goals_scored", "assists", "clean_sheets", "goals_conceded", "saves", "dc",
    "yellow_cards", "red_cards", "team_goals", "opp_goals", "bps90_prior",
]


def prior_bps90(matches: pd.DataFrame, k90: float = 3.0) -> pd.Series:
    """Leak-free baseline BPS/90 for each row: uses only the player's earlier matches."""
    pos_mean = (matches.groupby("position")["bps"].sum() / (matches.groupby("position")["minutes"].sum() / 90)).to_dict()
    g = matches.groupby("player_code", sort=False)
    cum_bps = g["bps"].cumsum() - matches["bps"]
    cum_n90 = (g["minutes"].cumsum() - matches["minutes"]) / 90.0
    prior = matches["position"].map(pos_mean)
    return (cum_bps + k90 * prior) / (cum_n90 + k90)


@dataclass
class BonusModel:
    booster: lgb.Booster
    n_train: int

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.clip(self.booster.predict(X[FEATURES]), 0.0, 3.0)


def training_rows(matches: pd.DataFrame, cutoff: pd.Timestamp, params: ModelParams) -> pd.DataFrame:
    m = matches.copy()
    m["bps90_prior"] = prior_bps90(m)
    m = m[(m["kickoff"] < cutoff) & (m["minutes"] > 0) & (m["season_idx"] >= params.bonus_min_season)]
    m = m[m["team_goals"].notna()]
    m["pos"] = m["position"].map(POS_CODE)
    m["dc"] = m["defensive_contribution"].fillna(0.0)
    return m


def train_bonus_model(rows: pd.DataFrame, params: ModelParams) -> BonusModel:
    p = _lgb_params(params, "regression", params.bonus_leaves, params.bonus_min_leaf, params.bonus_learning_rate)
    ds = lgb.Dataset(rows[FEATURES], label=rows["bonus"], categorical_feature=["pos"], free_raw_data=True)
    return BonusModel(booster=lgb.train(p, ds, num_boost_round=params.bonus_rounds), n_train=len(rows))


def expected_bonus(pf: pd.DataFrame, model: BonusModel, dc_dispersion: dict[str, float], n_sims: int, seed: int) -> np.ndarray:
    """E[bonus] per player-fixture row (play probability included).

    ``pf`` columns: position, pos, scen_p (n,3) via p_s60/p_slt/p_sub, m_s60/m_slt/m_sub,
    lam_team, lam_opp, share_g, share_a, saves_mu90, dc90, yc90, rc90, bps90.
    """
    n = len(pf)
    if n == 0:
        return np.zeros(0)
    rng = np.random.default_rng(seed)
    probs = pf[["p_s60", "p_slt", "p_sub"]].to_numpy(dtype=float)
    cum = np.cumsum(probs, axis=1)
    u = rng.random((n, n_sims))
    scen = np.where(u < cum[:, [0]], 0, np.where(u < cum[:, [1]], 1, np.where(u < cum[:, [2]], 2, 3)))
    mins_tab = pf[["m_s60", "m_slt", "m_sub"]].to_numpy(dtype=float)
    plays = scen < 3
    rows, sims = np.nonzero(plays)
    if len(rows) == 0:
        return np.zeros(n)
    minutes = mins_tab[rows, scen[rows, sims]]
    f = minutes / 90.0

    def col(c: str) -> np.ndarray:
        return pf[c].to_numpy(dtype=float)[rows]

    tg = rng.poisson(col("lam_team"))
    og = rng.poisson(col("lam_opp"))
    qg = np.clip(col("share_g") * f, 0.0, 0.95)
    goals = rng.binomial(tg, qg)
    qa = np.clip(col("share_a") * f / np.maximum(1.0 - qg, 0.05), 0.0, 0.95)
    assists = rng.binomial(tg - goals, qa)
    conceded = rng.binomial(og, np.clip(f, 0, 1))
    cs = ((conceded == 0) & (minutes >= 60)).astype(float)
    pos_names = pf["position"].to_numpy()[rows]
    is_gk = pos_names == "GKP"
    saves = np.where(is_gk, rng.poisson(col("saves_mu90") * f), 0)
    dc_mu = col("dc90") * f
    disp = np.array([dc_dispersion.get(p, np.inf) for p in pos_names])
    finite = np.isfinite(disp) & (dc_mu > 0)
    dc = np.where(is_gk, 0, rng.poisson(dc_mu)).astype(float)
    if finite.any():
        lam = rng.gamma(disp[finite], dc_mu[finite] / disp[finite])
        dc[finite] = rng.poisson(lam)
    yc = (rng.random(len(rows)) < np.clip(col("yc90") * f, 0, 1)).astype(float)
    rc = (rng.random(len(rows)) < np.clip(col("rc90") * f, 0, 1)).astype(float)
    X = pd.DataFrame(
        {
            "pos": pf["pos"].to_numpy()[rows], "minutes": minutes, "goals_scored": goals, "assists": assists,
            "clean_sheets": cs, "goals_conceded": conceded, "saves": saves, "dc": dc,
            "yellow_cards": yc, "red_cards": rc, "team_goals": tg, "opp_goals": og, "bps90_prior": col("bps90"),
        }
    )
    b = model.predict(X)
    total = np.zeros(n)
    np.add.at(total, rows, b)
    return total / n_sims

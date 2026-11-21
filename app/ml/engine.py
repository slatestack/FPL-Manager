"""The xP engine: combine team goal rates, playing time and player rates into expected points.

For every player and upcoming fixture, playing time is a mixture of three scenarios:
start & play 60+, start & off before 60, or come off the bench. Within each scenario all
non-bonus components have closed forms (Poisson / negative binomial), summed with the
scenario probabilities. Bonus comes from ``bonus.expected_bonus``. Gameweek xP is the sum
over that gameweek's fixtures (double gameweeks add, blanks are zero).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson

from app.ml import bonus as bonus_mod
from app.ml import minutes as minutes_mod
from app.ml import rates as rates_mod
from app.ml.data import ModelData
from app.ml.params import ModelParams
from app.ml.team_strength import TeamRatings, fit_team_ratings


@dataclass
class TrainedModels:
    minutes: minutes_mod.MinutesModel
    bonus: bonus_mod.BonusModel


@dataclass
class EngineResult:
    by_fixture: pd.DataFrame
    by_gameweek: pd.DataFrame
    fixtures: pd.DataFrame
    diagnostics: dict[str, Any] = field(default_factory=dict)


def train_models(data: ModelData, anchors: pd.DataFrame, cutoff: pd.Timestamp, params: ModelParams) -> TrainedModels:
    table = minutes_mod.training_table(data.matches, anchors, cutoff, params.horizon)
    mm = minutes_mod.train_minutes_model(table, params)
    bm = bonus_mod.train_bonus_model(bonus_mod.training_rows(data.matches, cutoff, params), params)
    return TrainedModels(minutes=mm, bonus=bm)


def expected_floor_div(mu: np.ndarray, n: int, terms: int = 25) -> np.ndarray:
    """E[floor(X / n)] for X ~ Poisson(mu) = sum_{j>=1} P(X >= j*n)."""
    j = np.arange(1, terms + 1) * n
    return poisson.sf(j[None, :] - 1, np.asarray(mu, dtype=float)[:, None]).sum(axis=1)


def prob_at_least(mu: np.ndarray, thr: np.ndarray, size: np.ndarray) -> np.ndarray:
    """P(X >= thr) for X ~ NegBin(mean mu, size) (Poisson when size is inf)."""
    mu = np.asarray(mu, dtype=float)
    out = np.zeros_like(mu)
    pois = ~np.isfinite(size)
    out[pois] = poisson.sf(thr[pois] - 1, mu[pois])
    nb = ~pois
    if nb.any():
        s = size[nb]
        out[nb] = nbinom.sf(thr[nb] - 1, s, s / (s + mu[nb]))
    return out


def fixture_rates(targets: pd.DataFrame, ratings: TeamRatings, odds: pd.DataFrame | None, odds_weight: float) -> pd.DataFrame:
    t = targets.copy()
    lam = np.array([ratings.rates(h, a) for h, a in zip(t["home_code"], t["away_code"])]) if len(t) else np.zeros((0, 2))
    t["lh_model"], t["la_model"] = lam[:, 0], lam[:, 1]
    t["odds_w"] = 0.0
    t["lh"], t["la"] = t["lh_model"], t["la_model"]
    if odds is not None and len(odds) and "fixture_id" in t:
        t = t.merge(odds[["fixture_id", "odds_lh", "odds_la"]], on="fixture_id", how="left")
        has = t["odds_lh"].notna()
        t.loc[has, "odds_w"] = odds_weight
        t["lh"] = np.where(has, odds_weight * t["odds_lh"] + (1 - odds_weight) * t["lh_model"], t["lh_model"])
        t["la"] = np.where(has, odds_weight * t["odds_la"] + (1 - odds_weight) * t["la_model"], t["la_model"])
    return t


def player_fixture_frame(players: pd.DataFrame, fx: pd.DataFrame) -> pd.DataFrame:
    """Cross players with their team's target fixtures; k = 1-based order of the team's fixtures."""
    home = fx.rename(columns={"home_code": "team_code", "away_code": "opp_code", "lh": "lam_team", "la": "lam_opp"})
    home["was_home"] = True
    away = fx.rename(columns={"away_code": "team_code", "home_code": "opp_code", "la": "lam_team", "lh": "lam_opp"})
    away["was_home"] = False
    tf = pd.concat([home, away], ignore_index=True).sort_values(["team_code", "kickoff", "fixture_uid"])
    tf["k"] = tf.groupby("team_code").cumcount() + 1
    return players.merge(tf[["team_code", "opp_code", "fixture_uid", "gameweek", "kickoff", "was_home", "lam_team", "lam_opp", "k"]],
                         on="team_code", how="inner")


def predict(
    data: ModelData,
    players: pd.DataFrame,
    targets: pd.DataFrame,
    as_of: pd.Timestamp,
    params: ModelParams,
    models: TrainedModels,
    anchors: pd.DataFrame,
    *,
    avail: pd.DataFrame | None = None,
    odds: pd.DataFrame | None = None,
    gameweeks: list[int] | None = None,
) -> EngineResult:
    """``players``: key, code, team_code, position, price. ``targets``: fixture_uid, gameweek,
    home_code, away_code, kickoff (+ fixture_id for odds). ``avail``: key, gameweek, p_avail."""
    sc = data.scoring
    m = data.matches
    cur = data.current_idx

    ratings = fit_team_ratings(data.fixtures, as_of, cur, params, extra_teams=set(players["team_code"]))
    fx = fixture_rates(targets, ratings, odds, params.odds_weight)
    league = rates_mod.league_constants(m, data.fixtures, as_of)

    pl = players.set_index("code", drop=False)
    rates = rates_mod.player_rates(m, as_of, cur, pl["price"], pl["position"], league, params)

    pf = player_fixture_frame(players, fx)
    if pf.empty:
        gws = gameweeks or []
        return EngineResult(pf, aggregate_gameweeks(pf.assign(p_play=[], p_start=[], p_clean_sheet=[]), players, gws), fx)
    pf = pf.join(rates, on="code")
    rows_before = m.groupby("player_code").size()
    feats = minutes_mod.inference_features(anchors, rows_before, pf[["code", "position", "price"]], cur)
    feats["k"] = pf["k"].clip(upper=params.horizon).to_numpy()
    p_start, p_app = models.minutes.predict(feats)
    pf["p_start_raw"], pf["p_app_raw"] = p_start, p_app

    if avail is not None:
        pf = pf.merge(avail, on=["key", "gameweek"], how="left")
        pf["p_avail"] = pf["p_avail"].fillna(1.0)
    else:
        pf["p_avail"] = 1.0

    pos = pf["position"]
    pf["pos"] = pos.map(minutes_mod.POS_CODE)
    ps = pf["p_avail"] * pf["p_start_raw"]
    pa = pf["p_avail"] * pf["p_app_raw"]
    pf["p_s60"] = ps * pf["p60"]
    pf["p_slt"] = ps * (1 - pf["p60"])
    pf["p_sub"] = (pa - ps).clip(lower=0)
    pf["p_play"] = pf["p_s60"] + pf["p_slt"] + pf["p_sub"]
    pf["m_s60"] = pos.map(league.minutes_start60)
    pf["m_slt"] = pos.map(league.minutes_start_lt60)
    pf["m_sub"] = pos.map(league.minutes_sub)
    pf["saves_mu90"] = pf["saves_per_xga"] * pf["lam_opp"]

    g_pts = pos.map(sc.goals_scored).to_numpy(dtype=float)
    cs_pts = pos.map(sc.clean_sheets).to_numpy(dtype=float)
    gc_pts = pos.map(sc.goals_conceded).to_numpy(dtype=float)
    dc_pts = pos.map(sc.defensive_contribution).to_numpy(dtype=float)
    dc_thr = pos.map(sc.dc_threshold).to_numpy(dtype=float)
    dc_size = pos.map(league.dc_dispersion).to_numpy(dtype=float)
    is_gk = (pos == "GKP").to_numpy()
    lam_t, lam_o = pf["lam_team"].to_numpy(), pf["lam_opp"].to_numpy()

    comp = {c: np.zeros(len(pf)) for c in ("app", "goals", "assists", "cs", "gc", "saves", "dc", "cards", "misc", "minutes", "xg", "xa", "pcs")}
    for scen, long in (("s60", True), ("slt", False), ("sub", False)):
        pr = pf[f"p_{scen}"].to_numpy()
        mins = pf[f"m_{scen}"].to_numpy(dtype=float)
        f = mins / 90.0
        comp["minutes"] += pr * mins
        comp["app"] += pr * (sc.long_play if long else sc.short_play)
        eg = pf["share_g"].to_numpy() * lam_t * f
        ea = pf["share_a"].to_numpy() * lam_t * f
        comp["xg"] += pr * eg
        comp["xa"] += pr * ea
        comp["goals"] += pr * eg * g_pts
        comp["assists"] += pr * ea * sc.assists
        if long:
            pcs = np.exp(-lam_o * f)
            comp["pcs"] += pr * pcs
            comp["cs"] += pr * pcs * cs_pts
        comp["gc"] += pr * expected_floor_div(lam_o * f, sc.goals_conceded_per) * gc_pts
        sv = np.where(is_gk, expected_floor_div(pf["saves_mu90"].to_numpy() * f, sc.saves_per) * sc.saves, 0.0)
        comp["saves"] += pr * sv
        dcp = np.where(is_gk, 0.0, prob_at_least(pf["dc90"].to_numpy() * f, dc_thr, dc_size)) * dc_pts
        comp["dc"] += pr * dcp
        comp["cards"] += pr * f * (pf["yc90"].to_numpy() * sc.yellow_cards + pf["rc90"].to_numpy() * sc.red_cards)
        comp["misc"] += pr * f * (
            pf["og90"].to_numpy() * sc.own_goals
            + pf["pm90"].to_numpy() * sc.penalties_missed
            + np.where(is_gk, pf["ps90"].to_numpy() * sc.penalties_saved, 0.0)
        )

    pf["xp_bonus"] = bonus_mod.expected_bonus(pf, models.bonus, league.dc_dispersion, params.bonus_simulations, params.seed)
    pf["xp_appearance"] = comp["app"]
    pf["xp_attacking"] = comp["goals"] + comp["assists"]
    pf["xp_defensive"] = comp["cs"] + comp["gc"] + comp["saves"] + comp["dc"]
    pf["xp_deductions"] = comp["cards"] + comp["misc"]
    pf["xp"] = pf["xp_appearance"] + pf["xp_attacking"] + pf["xp_defensive"] + pf["xp_deductions"] + pf["xp_bonus"]
    pf["expected_minutes"] = comp["minutes"]
    pf["expected_goals"] = comp["xg"]
    pf["expected_assists"] = comp["xa"]
    pf["p_clean_sheet"] = comp["pcs"]
    pf["p_start"] = ps

    # Explicit list: a gameweek in which nobody plays has no fixtures but still needs (zero) rows.
    gws = gameweeks if gameweeks is not None else sorted(targets["gameweek"].dropna().unique())
    gw = aggregate_gameweeks(pf, players, gws)
    diag = {
        "league": league.as_dict(),
        "team_home_advantage": float(np.exp(ratings.home)),
        "team_avg_rate": ratings.avg_rate,
        "minutes_train_rows": models.minutes.n_train,
        "bonus_train_rows": models.bonus.n_train,
    }
    return EngineResult(by_fixture=pf, by_gameweek=gw, fixtures=fx, diagnostics=diag)


SUM_COLS = ["xp", "xp_appearance", "xp_attacking", "xp_defensive", "xp_bonus", "xp_deductions",
            "expected_minutes", "expected_goals", "expected_assists"]


def aggregate_gameweeks(pf: pd.DataFrame, players: pd.DataFrame, gameweeks: list[int]) -> pd.DataFrame:
    """Per (player, gameweek): sums over fixtures; blanks are explicit zero rows."""
    pf = pf.reindex(columns=list(dict.fromkeys(["key", "gameweek", "p_play", "p_start", "p_clean_sheet", *SUM_COLS])))
    g = pf.groupby(["key", "gameweek"])
    out = g[SUM_COLS].sum()
    out["n_fixtures"] = g.size()
    out["prob_play"] = 1 - g["p_play"].apply(lambda s: float(np.prod(1 - s.to_numpy())))
    out["p_start"] = 1 - g["p_start"].apply(lambda s: float(np.prod(1 - s.to_numpy())))
    out["p_clean_sheet"] = g["p_clean_sheet"].mean()
    full = pd.MultiIndex.from_product([players["key"].unique(), gameweeks], names=["key", "gameweek"])
    out = out.reindex(full)
    out["n_fixtures"] = out["n_fixtures"].fillna(0).astype(int)
    return out.fillna(0.0).reset_index()

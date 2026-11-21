"""Per-player rates estimated from recency-weighted history with empirical-Bayes shrinkage.

Attacking output is modelled as a *share of the team's goal expectation while on the pitch*:

    share_g = sum w * (a*xG + (1-a)*goals) / sum w * team_xG_match * minutes/90

so a player's expected goals in a future fixture are ``share_g * lambda_team * minutes/90``.
This carries over when a player changes club and lets the fixture enter only via
``lambda_team``. Shares shrink toward a price-dependent position prior, so new signings
and low-minute players get sensible values. Count rates (saves, defensive contributions,
cards, bps) are per-90 rates shrunk toward position means.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.ml.params import ModelParams
from app.ml.weights import recency_weights

POSITIONS = ("GKP", "DEF", "MID", "FWD")


@dataclass
class LeagueConstants:
    assist_calibration: float                      # FPL assists per unit of xA
    minutes_start60: dict[str, float]              # E[minutes | start, >=60]
    minutes_start_lt60: dict[str, float]           # E[minutes | start, <60]
    minutes_sub: dict[str, float]                  # E[minutes | sub appearance]
    p60_given_start: dict[str, float]
    dc_dispersion: dict[str, float]                # negative-binomial size parameter (inf = Poisson)
    saves_per90: float
    avg_team_rate: float                           # league mean goals per team per match
    share_prior: dict[str, tuple[float, float, float]] = field(default_factory=dict)  # pos -> (a, b, floor) for goals
    assist_prior: dict[str, tuple[float, float, float]] = field(default_factory=dict)

    def as_dict(self) -> dict:
        from dataclasses import asdict

        d = asdict(self)
        d["dc_dispersion"] = {k: (None if np.isinf(v) else v) for k, v in self.dc_dispersion.items()}
        return d


def _price_prior(num: pd.Series, den: pd.Series, price: pd.Series, pos: pd.Series) -> dict[str, tuple[float, float, float]]:
    """Per position weighted LS of observed share on price (in £m): share ~ a + b*price."""
    out = {}
    for p in POSITIONS:
        m = (pos == p) & (den > 0.5) & price.notna()
        if m.sum() < 10:
            mean = float(num[pos == p].sum() / max(den[pos == p].sum(), 1e-9))
            out[p] = (mean, 0.0, 0.25 * mean)
            continue
        y = (num[m] / den[m]).to_numpy()
        x = price[m].to_numpy() / 10.0
        w = den[m].to_numpy()
        X = np.column_stack([np.ones_like(x), x])
        beta = np.linalg.lstsq(X * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]
        mean = float(np.average(y, weights=w))
        out[p] = (float(beta[0]), float(beta[1]), 0.25 * mean)
    return out


def apply_price_prior(prior: dict[str, tuple[float, float, float]], pos: pd.Series, price: pd.Series) -> np.ndarray:
    a = pos.map(lambda p: prior[p][0]).to_numpy(dtype=float)
    b = pos.map(lambda p: prior[p][1]).to_numpy(dtype=float)
    floor = pos.map(lambda p: prior[p][2]).to_numpy(dtype=float)
    return np.maximum(a + b * price.to_numpy(dtype=float) / 10.0, floor)


def _dc_dispersion(m: pd.DataFrame) -> dict[str, float]:
    out = {}
    full = m[(m["minutes"] >= 85) & m["defensive_contribution"].notna()]
    for p in POSITIONS:
        g = full[full["position"] == p].groupby("player_code")["defensive_contribution"]
        stats = g.agg(["mean", "var", "count"])
        stats = stats[stats["count"] >= 5]
        excess = (stats["var"] - stats["mean"]).clip(lower=0)
        if p == "GKP" or excess.sum() <= 1e-9:
            out[p] = float("inf")
        else:
            out[p] = float((stats["mean"] ** 2).sum() / excess.sum())
    return out


def league_constants(m: pd.DataFrame, fixtures: pd.DataFrame, as_of: pd.Timestamp) -> LeagueConstants:
    played = m[m["minutes"] > 0]
    xa = played["expected_assists"].sum()
    calib = float(played["assists"].sum() / xa) if xa > 0 else 1.0
    start = m["starts"].fillna((m["minutes"] >= 60).astype(float)) > 0
    s60 = m[start & (m["minutes"] >= 60)].groupby("position")["minutes"].mean()
    slt = m[start & (m["minutes"] < 60) & (m["minutes"] > 0)].groupby("position")["minutes"].mean()
    sub = m[~start & (m["minutes"] > 0)].groupby("position")["minutes"].mean()
    starts = m[start]
    p60 = (starts["minutes"] >= 60).groupby(starts["position"]).mean()
    gk = played[played["position"] == "GKP"]
    sv90 = float(gk["saves"].sum() / max(gk["minutes"].sum() / 90, 1e-9)) if len(gk) else 2.5
    recent = fixtures[(fixtures["kickoff"] < as_of) & fixtures["home_goals"].notna()].tail(760)
    avg = float(np.nanmean(np.concatenate([recent["home_goals"], recent["away_goals"]]))) if len(recent) else 1.35
    return LeagueConstants(
        assist_calibration=calib,
        minutes_start60={p: float(s60.get(p, 88.0)) for p in POSITIONS},
        minutes_start_lt60={p: float(slt.get(p, 50.0)) for p in POSITIONS},
        minutes_sub={p: float(sub.get(p, 18.0)) for p in POSITIONS},
        p60_given_start={p: float(p60.get(p, 0.8)) for p in POSITIONS},
        dc_dispersion=_dc_dispersion(m),
        saves_per90=sv90,
        avg_team_rate=avg,
    )


def player_rates(
    m: pd.DataFrame, as_of: pd.Timestamp, current_idx: int, prices: pd.Series, positions: pd.Series,
    league: LeagueConstants, params: ModelParams,
) -> pd.DataFrame:
    """Rates for every code in ``positions`` (index = player code)."""
    played = m[m["minutes"] > 0].copy()
    played["w"] = recency_weights(played["kickoff"], played["season_idx"], as_of, current_idx,
                                  params.rate_half_life_days, params.rate_prev_season_factor)
    played["n90"] = played["minutes"] / 90.0
    xg = played["expected_goals"].fillna(played["goals_scored"])
    xa = played["expected_assists"].fillna(played["assists"] / max(league.assist_calibration, 1e-9))
    played["g_num"] = params.goal_xg_weight * xg + (1 - params.goal_xg_weight) * played["goals_scored"]
    played["a_num"] = params.assist_xa_weight * xa * league.assist_calibration + (1 - params.assist_xa_weight) * played["assists"]
    team_xg = played["team_xg"].fillna(played["team_goals"]).fillna(league.avg_team_rate)
    played["den"] = team_xg * played["n90"]
    # Opponent xG faced while on the pitch: saves are modelled per unit of it, so a keeper's
    # rate doesn't re-count how often his own team gets outshot.
    opp_xg = played["opp_xg"].fillna(played["opp_goals"]).fillna(league.avg_team_rate)
    played["xga"] = opp_xg * played["n90"]
    has_dc = played["defensive_contribution"].notna()

    def wsum(col: pd.Series, mask: pd.Series | None = None) -> pd.Series:
        v = played["w"] * col
        if mask is not None:
            v = v.where(mask, 0.0)
        return v.groupby(played["player_code"]).sum()

    agg = pd.DataFrame(
        {
            "g_num": wsum(played["g_num"]),
            "a_num": wsum(played["a_num"]),
            "den": wsum(played["den"]),
            "n90": wsum(played["n90"]),
            "saves": wsum(played["saves"]),
            "xga": wsum(played["xga"]),
            "dc": wsum(played["defensive_contribution"].fillna(0), has_dc),
            "dc_n90": wsum(played["n90"], has_dc),
            "yc": wsum(played["yellow_cards"]),
            "rc": wsum(played["red_cards"]),
            "og": wsum(played["own_goals"]),
            "pm": wsum(played["penalties_missed"]),
            "ps": wsum(played["penalties_saved"]),
            "bps": wsum(played["bps"]),
        }
    )
    start = played["starts"].fillna((played["minutes"] >= 60).astype(float)) > 0
    st = played[start]
    agg["n_start"] = st.groupby("player_code")["w"].sum()
    agg["n_start60"] = st[st["minutes"] >= 60].groupby("player_code")["w"].sum()

    # Price-dependent priors, fitted on players observed in the window (price at their last match).
    last_price = played.groupby("player_code")["value"].last()
    last_pos = played.groupby("player_code")["position"].last()
    league.share_prior = _price_prior(agg["g_num"], agg["den"], last_price.reindex(agg.index), last_pos.reindex(agg.index))
    league.assist_prior = _price_prior(agg["a_num"], agg["den"], last_price.reindex(agg.index), last_pos.reindex(agg.index))

    # Position means for per-90 rates.
    by_pos = played.groupby("position")
    pos_rate = {
        c: (by_pos.apply(lambda g, c=c: (g["w"] * g[c]).sum() / max((g["w"] * g["n90"]).sum(), 1e-9), include_groups=False))
        for c in ("saves", "yellow_cards", "red_cards", "own_goals", "penalties_missed", "penalties_saved", "bps")
    }
    dc_rows = played[has_dc]
    dc_pos = dc_rows.groupby("position").apply(
        lambda g: (g["w"] * g["defensive_contribution"]).sum() / max((g["w"] * g["n90"]).sum(), 1e-9), include_groups=False
    )

    out = pd.DataFrame(index=positions.index)
    a = agg.reindex(out.index).fillna(0.0)
    pos = positions
    k_s, k90, kd = params.share_prior_strength, params.per90_prior_strength, params.discipline_prior_strength
    prior_g = apply_price_prior(league.share_prior, pos, prices)
    prior_a = apply_price_prior(league.assist_prior, pos, prices)
    out["share_g"] = (a["g_num"] + k_s * prior_g) / (a["den"] + k_s)
    out["share_a"] = (a["a_num"] + k_s * prior_a) / (a["den"] + k_s)

    def per90(num: str, n90: pd.Series, rate_map, k: float) -> np.ndarray:
        prior = pos.map(lambda p: float(rate_map.get(p, 0.0))).to_numpy(dtype=float)
        return ((a[num] + k * prior) / (n90 + k)).to_numpy()

    gk = played[played["position"] == "GKP"]
    saves_per_xga = float((gk["w"] * gk["saves"]).sum() / max((gk["w"] * gk["xga"]).sum(), 1e-9)) if len(gk) else 2.0
    k_xga = k90 * league.avg_team_rate
    out["saves_per_xga"] = ((a["saves"] + k_xga * saves_per_xga) / (a["xga"] + k_xga)).to_numpy()
    out["dc90"] = per90("dc", a["dc_n90"], dc_pos, k90)
    out["yc90"] = per90("yc", a["n90"], pos_rate["yellow_cards"], kd)
    out["rc90"] = per90("rc", a["n90"], pos_rate["red_cards"], kd * 4)
    out["og90"] = per90("og", a["n90"], pos_rate["own_goals"], kd * 4)
    out["pm90"] = per90("pm", a["n90"], pos_rate["penalties_missed"], kd * 2)
    out["ps90"] = per90("ps", a["n90"], pos_rate["penalties_saved"], kd * 2)
    out["bps90"] = per90("bps", a["n90"], pos_rate["bps"], k90)
    p60_prior = pos.map(league.p60_given_start).to_numpy(dtype=float)
    kp = params.p60_prior_strength
    out["p60"] = (a["n_start60"] + kp * p60_prior) / (a["n_start"] + kp)
    out["n90_observed"] = a["n90"]
    return out

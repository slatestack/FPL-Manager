"""Walk-forward backtest: for each gameweek, fit on data strictly before its first kickoff,
predict that gameweek, and compare with what actually happened.

No historical injury news is available, so the model runs with availability = 1 for everyone,
exactly like the form baselines it is compared with. Metrics are therefore reported both on
all registered players and conditional on the player appearing.
"""

from __future__ import annotations

import logging
import time
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from app.ml import engine
from app.ml.data import ModelData, season_start
from app.ml.minutes import anchor_features
from app.ml.params import ModelParams

log = logging.getLogger(__name__)


def _baselines(m: pd.DataFrame, cutoff: pd.Timestamp, season_idx: int) -> pd.DataFrame:
    """Season-to-date points per team match, points per appearance, last-5 form (per player code)."""
    past = m[(m["kickoff"] < cutoff) & (m["season_idx"] == season_idx)]
    g = past.groupby("player_code")
    out = pd.DataFrame({"ppm": g["total_points"].mean()})
    app = past[past["minutes"] > 0].groupby("player_code")["total_points"].mean()
    out["ppa"] = app
    out["form5"] = g["total_points"].apply(lambda s: s.tail(5).mean())
    return out


def run_backtest(
    data: ModelData, season: str, params: ModelParams | None = None, *, first_gw: int = 6, last_gw: int = 38, retrain_every: int = 4
) -> dict[str, Any]:
    params = params or ModelParams()
    season_idx = season_start(season)
    allm = data.matches
    anchors = anchor_features(allm)
    fx = data.fixtures[data.fixtures["season"] == season]
    results = []
    models = None
    t0 = time.monotonic()
    for i, gw in enumerate(range(first_gw, last_gw + 1)):
        targets = fx[fx["gameweek"] == gw][["fixture_uid", "gameweek", "home_code", "away_code", "kickoff"]]
        if targets.empty:
            continue
        cutoff = targets["kickoff"].min()
        cut = replace(data.before(cutoff), current_season=season)
        if models is None or i % retrain_every == 0:
            models = engine.train_models(cut, anchors, cutoff, params)
        # Universe: players registered for the gameweek (archive lists every squad player each GW).
        reg = allm[(allm["season_idx"] == season_idx) & (allm["gameweek"] == gw)]
        actual = reg.groupby("player_code").agg(
            points=("total_points", "sum"), minutes=("minutes", "sum"), started=("starts", "max"),
            team_code=("team_code", "first"), position=("position", "first"), price=("value", "first"),
        )
        players = actual[["team_code", "position", "price"]].reset_index().rename(columns={"player_code": "code"})
        players["key"] = players["code"]
        res = engine.predict(cut, players, targets, cutoff, replace(params, horizon=params.horizon), models, anchors)
        pred = res.by_gameweek.set_index("key")
        df = actual.join(pred[["xp", "prob_play", "p_start", "xp_bonus", "expected_goals"]], how="inner")
        df = df.join(_baselines(allm, cutoff, season_idx))
        df["gameweek"] = gw
        results.append(df.reset_index())
        log.info("backtest %s GW%d: %d players (%.0fs)", season, gw, len(df), time.monotonic() - t0)
    all_df = pd.concat(results, ignore_index=True)
    return {"season": season, "params": params.as_dict(), "metrics": evaluate(all_df), "rows": len(all_df)}, all_df


def _score(pred: pd.Series, y: pd.Series) -> dict[str, float]:
    ok = pred.notna()
    p, t = pred[ok].to_numpy(dtype=float), y[ok].to_numpy(dtype=float)
    return {
        "mae": float(np.mean(np.abs(p - t))),
        "rmse": float(np.sqrt(np.mean((p - t) ** 2))),
        "spearman": float(spearmanr(p, t).statistic),
        "bias": float(np.mean(p - t)),
        "n": int(ok.sum()),
    }


def evaluate(df: pd.DataFrame) -> dict[str, Any]:
    df = df.copy()
    for c in ("ppm", "form5", "ppa"):
        df[c] = df[c].fillna(0.0)
    out: dict[str, Any] = {}
    out["all_players"] = {name: _score(df[col], df["points"]) for name, col in
                          (("model", "xp"), ("season_ppm", "ppm"), ("form_last5", "form5"))}
    played = df[df["minutes"] > 0].copy()
    played["xp_cond"] = played["xp"] / played["prob_play"].clip(lower=0.05)
    out["when_played"] = {name: _score(played[col], played["points"]) for name, col in
                          (("model", "xp_cond"), ("season_ppa", "ppa"), ("form_last5", "form5"))}

    # Top-N selection quality: mean actual points of each method's top 30 per gameweek.
    top = {}
    for name, col in (("model", "xp"), ("season_ppm", "ppm"), ("form_last5", "form5")):
        top[name] = float(df.sort_values(col, ascending=False).groupby("gameweek").head(30)["points"].mean())
    out["top30_mean_actual_points"] = top

    # Calibration of playing-time probabilities.
    df["started_f"] = (df["started"] > 0).astype(float)
    bins = pd.cut(df["p_start"], [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0001], include_lowest=True)
    cal = df.groupby(bins, observed=True).agg(pred=("p_start", "mean"), actual=("started_f", "mean"), n=("p_start", "size"))
    out["p_start_calibration"] = [
        {"bin": str(b), "predicted": round(r.pred, 3), "actual": round(r.actual, 3), "n": int(r.n)} for b, r in cal.iterrows()
    ]
    out["totals"] = {
        "expected_points": float(df["xp"].sum()), "actual_points": float(df["points"].sum()),
        "expected_bonus": float(df["xp_bonus"].sum()),
    }
    return out

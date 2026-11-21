"""Playing-time model.

Two LightGBM classifiers predict, for a player's k-th upcoming team fixture (k = 1..horizon),
P(start) and P(appear). Features summarise the player's recent selection pattern and price.
Features at "anchor" j use only rows strictly before j, so one feature table serves both
training (anchors inside the history) and inference / backtests (anchor = rows before cutoff)
without look-ahead.
"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

from app.ml.params import ModelParams

POS_CODE = {"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}
FEATURES = [
    "pos", "k", "n_prev", "s_last1", "s_mean3", "s_mean5", "s_mean10", "a_mean5", "m_mean5", "m_mean10",
    "gap", "season_break", "price",
]


def _window_mean(x: np.ndarray, j: np.ndarray, w: int) -> np.ndarray:
    c = np.concatenate([[0.0], np.cumsum(x)])
    lo = np.maximum(j - w, 0)
    cnt = j - lo
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, (c[j] - c[lo]) / cnt, np.nan)


def anchor_features(matches: pd.DataFrame) -> pd.DataFrame:
    """One row per (player, anchor j) for j = 0..n, describing rows [0, j).

    ``row_pos`` = j; ``last_season_idx`` is the season of row j-1 (for season-break flags).
    """
    out = []
    for code, g in matches.groupby("player_code", sort=False):
        n = len(g)
        start = g["starts"].fillna((g["minutes"] >= 60).astype(float)).to_numpy(dtype=float)
        app = (g["minutes"].to_numpy() > 0).astype(float)
        mins = g["minutes"].to_numpy(dtype=float) / 90.0
        season = g["season_idx"].to_numpy()
        j = np.arange(n + 1)
        last_start = np.full(n + 1, -1)
        ls = -1
        for i in range(n):
            if start[i] > 0:
                ls = i
            last_start[i + 1] = ls
        gap = np.where(last_start >= 0, j - 1 - last_start, 20).astype(float)
        gap[0] = np.nan
        out.append(
            pd.DataFrame(
                {
                    "player_code": code,
                    "row_pos": j,
                    "n_prev": np.minimum(j, 60).astype(float),
                    "s_last1": np.concatenate([[np.nan], start]),
                    "s_mean3": _window_mean(start, j, 3),
                    "s_mean5": _window_mean(start, j, 5),
                    "s_mean10": _window_mean(start, j, 10),
                    "a_mean5": _window_mean(app, j, 5),
                    "m_mean5": _window_mean(mins, j, 5),
                    "m_mean10": _window_mean(mins, j, 10),
                    "gap": np.minimum(gap, 20),
                    "last_season_idx": np.concatenate([[np.nan], season]).astype(float),
                }
            )
        )
    return pd.concat(out, ignore_index=True)


def training_table(matches: pd.DataFrame, anchors: pd.DataFrame, cutoff: pd.Timestamp, horizon: int) -> pd.DataFrame:
    """Rows (anchor j, target r = j + k - 1) for k in 1..horizon with target kickoff < cutoff."""
    m = matches[["player_code", "position", "kickoff", "season_idx", "minutes", "starts", "value"]].copy()
    m["row_pos"] = m.groupby("player_code", sort=False).cumcount()
    m["start"] = m["starts"].fillna((m["minutes"] >= 60).astype(float))
    m["app"] = (m["minutes"] > 0).astype(float)
    m = m[m["kickoff"] < cutoff]
    a = anchors.set_index(["player_code", "row_pos"])
    price_at = m.set_index(["player_code", "row_pos"])["value"]
    parts = []
    for k in range(1, horizon + 1):
        t = m.copy()
        t["anchor"] = t["row_pos"] - (k - 1)
        t = t[t["anchor"] >= 0]
        feats = a.reindex(pd.MultiIndex.from_arrays([t["player_code"], t["anchor"]])).reset_index(drop=True)
        t = t.reset_index(drop=True)
        t = pd.concat([t, feats.drop(columns=[], errors="ignore")], axis=1)
        t["k"] = k
        # Price known at the anchor (= price of the first upcoming match).
        t["price"] = price_at.reindex(pd.MultiIndex.from_arrays([t["player_code"], t["anchor"]])).to_numpy()
        t["season_break"] = np.where(t["last_season_idx"].isna(), np.nan, (t["season_idx"] != t["last_season_idx"]).astype(float))
        parts.append(t)
    out = pd.concat(parts, ignore_index=True)
    out["pos"] = out["position"].map(POS_CODE)
    return out


@dataclass
class MinutesModel:
    start: lgb.Booster
    appear: lgb.Booster
    n_train: int

    def predict(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        ps = self.start.predict(X[FEATURES])
        pa = self.appear.predict(X[FEATURES])
        return ps, np.maximum(pa, ps)


def _lgb_params(params: ModelParams, objective: str, leaves: int, min_leaf: int, lr: float) -> dict:
    return {
        "objective": objective,
        "learning_rate": lr,
        "num_leaves": leaves,
        "min_data_in_leaf": min_leaf,
        "feature_fraction": 0.9,
        "lambda_l2": 1.0,
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": params.threads,
        "seed": params.seed,
        "verbose": -1,
    }


def train_minutes_model(table: pd.DataFrame, params: ModelParams) -> MinutesModel:
    p = _lgb_params(params, "binary", params.minutes_leaves, params.minutes_min_leaf, params.minutes_learning_rate)
    boosters = []
    for target in ("start", "app"):
        ds = lgb.Dataset(table[FEATURES], label=table[target], categorical_feature=["pos"], free_raw_data=True)
        boosters.append(lgb.train(p, ds, num_boost_round=params.minutes_rounds))
    return MinutesModel(start=boosters[0], appear=boosters[1], n_train=len(table))


def inference_features(
    anchors: pd.DataFrame, rows_before: pd.Series, players: pd.DataFrame, current_idx: int
) -> pd.DataFrame:
    """Features at anchor = number of rows before the cutoff, for each player in ``players``.

    ``players`` needs ``code``, ``position`` and ``price``; players without any history get
    ``n_prev = 0`` and NaN elsewhere, exactly like a debut in the training data.
    """
    j = players["code"].map(rows_before).fillna(0).astype(int)
    a = anchors.set_index(["player_code", "row_pos"])
    f = a.reindex(pd.MultiIndex.from_arrays([players["code"], j])).reset_index(drop=True)
    f["n_prev"] = f["n_prev"].fillna(0.0)
    f.index = players.index
    f["pos"] = players["position"].map(POS_CODE)
    f["price"] = players["price"].astype(float)
    f["season_break"] = np.where(f["last_season_idx"].isna(), np.nan, (current_idx != f["last_season_idx"]).astype(float))
    return f

"""Replay a past season as if the system had managed a team from gameweek 1.

Each gameweek the simulator does what the live system does before a deadline:
train on data available before the first kickoff, predict the next ``horizon`` gameweeks,
run the optimiser (with chip evaluation) on the simulated team's state, and apply the
next-gameweek decision. It then scores that decision with the real points, applying
auto-subs, vice-captaincy, chips, hits, FPL selling prices and the free-transfer rules.

Limitations (all make the simulation *pessimistic*): there is no historical injury or
suspension news, so the model cannot see that a flagged player will miss a match, and
there are no historical bookmaker odds.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, replace
from typing import Any

import pandas as pd

from app.ml import engine
from app.ml.data import ModelData, season_start
from app.ml.minutes import anchor_features
from app.ml.params import ModelParams
from app.optimizer.chips import choose_chip, evaluate_chips
from app.optimizer.model import build_candidates, solve
from app.optimizer.params import OptimizerParams
from app.optimizer.team_state import MAX_FREE_TRANSFERS, ChipWindow, TeamState, selling_price
from app.optimizer.validate import validate_plan
from app.sim.scoring import score_gameweek

log = logging.getLogger(__name__)

# 2025/26 onwards: every chip twice, once per half of the season.
CHIP_WINDOWS = [
    ChipWindow("wildcard", 2, 19), ChipWindow("wildcard", 20, 38),
    ChipWindow("freehit", 2, 19), ChipWindow("freehit", 20, 38),
    ChipWindow("bboost", 1, 19), ChipWindow("bboost", 20, 38),
    ChipWindow("3xc", 1, 19), ChipWindow("3xc", 20, 38),
]
STARTING_BUDGET = 1000


@dataclass
class SimConfig:
    season: str = "2025-26"
    first_gw: int = 1
    last_gw: int = 38
    horizon: int = 5
    retrain_every: int = 4
    transfers: bool = True       # False = keep the GW1 squad all season (baseline)
    chips: bool = True
    label: str = "system"

    def __post_init__(self) -> None:
        if not self.transfers:
            self.chips = False  # chip solves may transfer; keep the baseline a pure "no moves" team


@dataclass
class GameweekRecord:
    gameweek: int
    points: int
    hits: int
    chip: str | None
    transfers: list[tuple[str, str]]
    captain: str
    captain_points: int
    predicted_xp: float
    autosubs: int
    bank: int
    free_transfers: int
    seconds: float


@dataclass
class SimResult:
    config: SimConfig
    gameweeks: list[GameweekRecord] = field(default_factory=list)

    @property
    def total(self) -> int:
        return sum(g.points for g in self.gameweeks)

    def summary(self) -> dict[str, Any]:
        gws = self.gameweeks
        return {
            "label": self.config.label,
            "total_points": self.total,
            "gameweeks": len(gws),
            "average": round(self.total / max(len(gws), 1), 1),
            "transfers": sum(len(g.transfers) for g in gws),
            "hits": sum(g.hits for g in gws),
            "chips": {g.gameweek: g.chip for g in gws if g.chip},
            "predicted_total": round(sum(g.predicted_xp for g in gws), 1),
        }


def _gw_frame(m: pd.DataFrame, season_idx: int, gw: int) -> pd.DataFrame:
    return m[(m["season_idx"] == season_idx) & (m["gameweek"] == gw)]


def simulate(data: ModelData, cfg: SimConfig, params: ModelParams | None = None,
             opt_params: OptimizerParams | None = None, *, progress=None) -> SimResult:
    params = params or ModelParams()
    opt_params = replace(opt_params or OptimizerParams(), time_limit_seconds=60)
    sidx = season_start(cfg.season)
    allm = data.matches
    anchors = anchor_features(allm)
    fixtures = data.fixtures[data.fixtures["season"] == cfg.season]
    web_name: dict[int, str] = getattr(data, "web_names", {}) or {}

    squad: list[int] = []
    purchase: dict[int, int] = {}
    bank = STARTING_BUDGET
    ft = 1
    chips_left = list(CHIP_WINDOWS) if cfg.chips else []
    last_seen: dict[int, dict[str, Any]] = {}
    fh_restore: tuple[list[int], dict[int, int], int] | None = None
    models = None
    result = SimResult(cfg)

    for i, gw in enumerate(range(cfg.first_gw, cfg.last_gw + 1)):
        t0 = time.monotonic()
        if fh_restore is not None:  # Free Hit week is over: the old squad comes back
            squad, purchase, bank = fh_restore
            fh_restore = None
        gws = [g for g in range(gw, min(gw + cfg.horizon, cfg.last_gw + 1))]
        targets = fixtures[fixtures["gameweek"].isin(gws)][["fixture_uid", "gameweek", "home_code", "away_code", "kickoff"]]
        this_gw = fixtures[fixtures["gameweek"] == gw]
        if this_gw.empty:
            continue
        cutoff = this_gw["kickoff"].min()
        cut = replace(data.before(cutoff), current_season=cfg.season)
        if models is None or i % cfg.retrain_every == 0:
            models = engine.train_models(cut, anchors, cutoff, params)

        # Players registered this gameweek, at their price this gameweek.
        rows = _gw_frame(allm, sidx, gw).drop_duplicates("player_code")
        for r in rows.itertuples():
            last_seen[r.player_code] = {"team_code": r.team_code, "position": r.position, "price": int(r.value)}
        universe = {r.player_code: last_seen[r.player_code] for r in rows.itertuples()}
        for p in squad:  # a squad player who left the league can still be sold
            universe.setdefault(p, last_seen[p])
        players = pd.DataFrame([{"key": c, "code": c, **v} for c, v in universe.items()])
        res = engine.predict(cut, players, targets, cutoff, params, models, anchors, gameweeks=gws)
        pred = res.by_gameweek.rename(columns={"key": "player_id", "gameweek": "target_gw", "xp": "xp_total"})
        cand_players = players.rename(columns={"code": "player_id", "team_code": "team_id", "price": "now_cost"})

        price = {c: v["price"] for c, v in universe.items()}
        state = TeamState(
            entry_id=0, next_gameweek=gw, squad=list(squad), bank=bank,
            selling_price={p: selling_price(purchase[p], price[p]) for p in squad},
            purchase_price=dict(purchase), free_transfers=ft,
            chips_available=[w for w in chips_left if w.start_event <= gw <= w.stop_event],
        )
        cand = build_candidates(pred, cand_players, state, opt_params)
        if not squad:   # gameweek 1: pick the squad from scratch with the whole budget
            plan = solve(cand, replace(state, chips_available=[]), opt_params, chip="wildcard")
            plan.chip = None
            plan.first.chip = None
        else:
            plan = solve(cand, state, opt_params, allow_transfers=cfg.transfers)
            validate_plan(plan, cand, state, opt_params)
            if cfg.chips and cfg.transfers and state.chips_available:
                chosen = choose_chip(evaluate_chips(cand, state, opt_params, plan))
                if chosen is not None:
                    plan = chosen.plan
        g = plan.first

        # Apply the decision.
        if g.chip == "freehit":
            fh_restore = (list(squad), dict(purchase), bank)
        sell_total = sum(selling_price(purchase[p], price[p]) for p in g.transfers_out if p in purchase)
        buy_total = sum(price[p] for p in g.transfers_in)
        bank = bank + sell_total - buy_total
        for p in g.transfers_out:
            purchase.pop(p, None)
        for p in g.transfers_in:
            purchase[p] = price[p]
        squad = list(g.squad)
        if g.chip in ("wildcard", "freehit"):
            pass  # banked free transfers are kept
        elif gw == cfg.first_gw and not state.squad:
            ft = 1
        else:
            ft = min(MAX_FREE_TRANSFERS, max(0, ft - len(g.transfers_in)) + 1)
        if g.chip:
            chips_left = [w for w in chips_left if not (w.name == g.chip and w.start_event <= gw <= w.stop_event)]
        if bank < 0:
            raise AssertionError(f"GW{gw}: negative bank {bank}")

        # Score it with what actually happened.
        actual = _gw_frame(allm, sidx, gw).groupby("player_code").agg(points=("total_points", "sum"), minutes=("minutes", "sum"))
        pts = actual["points"].astype(int).to_dict()
        mins = actual["minutes"].astype(int).to_dict()
        pos = {c: universe[c]["position"] for c in universe}
        sc = score_gameweek(g.starting_xi, g.bench, g.captain, g.vice_captain, pts, mins, pos, chip=g.chip, hits=g.hits)
        rec = GameweekRecord(
            gameweek=gw, points=sc.points, hits=g.hits, chip=g.chip,
            transfers=[(web_name.get(o, str(o)), web_name.get(n, str(n))) for o, n in zip(g.transfers_out, g.transfers_in)]
            if state.squad else [],
            captain=web_name.get(g.captain, str(g.captain)), captain_points=pts.get(g.captain, 0),
            predicted_xp=float(g.xp) - 4 * g.hits, autosubs=len(sc.autosubs), bank=bank, free_transfers=ft,
            seconds=round(time.monotonic() - t0, 1),
        )
        result.gameweeks.append(rec)
        if progress:
            progress(rec, result)
    return result


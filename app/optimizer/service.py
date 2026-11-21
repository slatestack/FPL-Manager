"""Optimiser run: latest xP + team state -> plan (+ chip decision), validated and logged.

Each run writes an ``optimization_runs`` row (inputs, params, solver statistics, the whole
horizon plan and chip analysis) and an ``execution_logs`` row of type ``DRY_RUN`` holding the
next-gameweek decision and its payload hash. Phase 5 executes from the same structure.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import asyncpg
import pandas as pd

from app.optimizer.chips import ChipEvaluation, choose_chip, evaluate_chips
from app.optimizer.model import Candidates, Plan, build_candidates, solve
from app.optimizer.params import OptimizerParams
from app.optimizer.team_state import TeamState, fetch_team_state
from app.optimizer.validate import validate_plan

log = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    run_id: str
    execution_log_id: str
    plan: Plan
    baseline: Plan
    chip: ChipEvaluation | None
    chip_evaluations: list[ChipEvaluation]
    state: TeamState
    names: dict[int, dict[str, Any]]
    prediction_run: Any
    team_short: dict[int, str] = field(default_factory=dict)


def decision_payload(plan: Plan) -> dict[str, Any]:
    g = plan.first
    return {
        "gameweek": g.gameweek, "transfers_in": sorted(g.transfers_in), "transfers_out": sorted(g.transfers_out),
        "starting_xi": sorted(g.starting_xi), "bench_order": g.bench, "captain": g.captain,
        "vice_captain": g.vice_captain, "chip": plan.chip,
    }


def payload_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def optimise(cand: Candidates, state: TeamState, params: OptimizerParams) -> tuple[Plan, Plan, ChipEvaluation | None, list[ChipEvaluation]]:
    """Pure: base plan, do-nothing baseline, chip counterfactuals, final choice (all validated)."""
    base = solve(cand, state, params)
    validate_plan(base, cand, state, params)
    baseline = solve(cand, state, params, allow_transfers=False)
    evals = evaluate_chips(cand, state, params, base)
    chosen = choose_chip(evals)
    plan = chosen.plan if chosen is not None else base
    validate_plan(plan, cand, state, params)
    return plan, baseline, chosen, evals


async def _config(pool: asyncpg.Pool, team_id: int) -> tuple[int, str]:
    row = await pool.fetchrow("SELECT max_hits_per_gw, chip_aggression_profile FROM system_config WHERE team_id = $1", team_id)
    return (row["max_hits_per_gw"], row["chip_aggression_profile"]) if row else (1, "BALANCED")


async def run_optimization(
    pool: asyncpg.Pool, fpl, team_id: int, *, params: OptimizerParams | None = None, free_transfers: int | None = None
) -> OptimizationResult:
    run = await pool.fetchrow("SELECT run_timestamp, base_gameweek FROM prediction_runs ORDER BY run_timestamp DESC LIMIT 1")
    if run is None:
        raise RuntimeError("no xP prediction run found; run `python -m app predict` first")
    max_hits, aggression = await _config(pool, team_id)
    params = replace(params or OptimizerParams(), max_hits_per_gw=max_hits, aggression=aggression)

    pred = pd.DataFrame([dict(r) for r in await pool.fetch(
        """SELECT player_id, target_gw, xp_total::float AS xp_total, prob_play::float AS prob_play, n_fixtures
           FROM player_predictions WHERE run_timestamp = $1""", run["run_timestamp"])])
    players = pd.DataFrame([dict(r) for r in await pool.fetch(
        "SELECT player_id, web_name, position, team_id, now_cost FROM players")])
    state, raw = await fetch_team_state(fpl, pool, team_id, int(run["base_gameweek"]), free_transfers=free_transfers)
    for w in state.warnings:
        log.warning("team state: %s", w)

    cand = build_candidates(pred, players, state, params)
    plan, baseline, chosen, evals = await asyncio.to_thread(optimise, cand, state, params)

    payload = decision_payload(plan)
    digest = payload_hash(payload)
    g = plan.first
    solver_stats = {
        "solver": params.solver, "status": plan.status, "objective": plan.objective, "seconds": plan.solve_seconds,
        "variables": plan.n_variables, "constraints": plan.n_constraints, "candidates": cand.n,
        "baseline_objective": baseline.objective,
        "chip_solves": {e.chip: (e.plan.solve_seconds if e.plan else None) for e in evals},
    }
    async with pool.acquire() as conn, conn.transaction():
        exec_id = await conn.fetchval(
            """INSERT INTO execution_logs (team_id, gameweek, execution_type, payload_hash, transfers_in, transfers_out,
                   starting_xi, bench_order, captain_id, vice_captain_id, chip_activated, projected_xp_gain, status)
               VALUES ($1, $2, 'DRY_RUN', $3, $4, $5, $6, $7, $8, $9, $10, $11, 'SUCCESS') RETURNING id""",
            team_id, g.gameweek, digest, g.transfers_in, g.transfers_out, g.starting_xi, g.bench, g.captain,
            g.vice_captain, plan.chip, round(plan.objective - baseline.objective, 2),
        )
        run_id = await conn.fetchval(
            """INSERT INTO optimization_runs (team_id, base_gameweek, prediction_run_timestamp, inputs, params, solver, plan,
                   chip_evaluation, chip_recommended, objective, baseline_objective, execution_log_id)
               VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12) RETURNING id""",
            team_id, g.gameweek, run["run_timestamp"], {"team_state": state.as_dict(), "raw": raw},
            json.loads(json.dumps(params.as_dict())), solver_stats, [asdict(gw) for gw in plan.gameweeks],
            [e.summary() for e in evals], plan.chip, round(plan.objective, 3), round(baseline.objective, 3), exec_id,
        )
    names = {int(r.player_id): {"name": r.web_name, "position": r.position, "team_id": int(r.team_id), "cost": int(r.now_cost)}
             for r in players.itertuples()}
    return OptimizationResult(str(run_id), str(exec_id), plan, baseline, chosen, evals, state, names, run["run_timestamp"])

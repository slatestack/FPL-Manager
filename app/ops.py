"""Composite operations shared by the CLI, the scheduler, the Telegram bot and the API."""

from __future__ import annotations

from contextlib import AsyncExitStack
from typing import Any

import asyncpg

from app.config import Settings
from app.ingestion import pipeline
from app.ingestion.factory import build_sources


async def refresh_data(pool: asyncpg.Pool, settings: Settings, *, goalscorers: bool = True) -> None:
    """Latest bootstrap/fixtures and odds (team news, prices, lineup leaks). With ``goalscorers``,
    also the goalscorer odds for the coming gameweek's matches (1 odds credit per match)."""
    async with AsyncExitStack() as stack:
        fpl, _, odds = await build_sources(settings, stack)
        await pipeline.run_job(pool, "core", lambda: pipeline.ingest_core(pool, fpl))
        if odds is not None:
            until = None
            if goalscorers:
                until = await pool.fetchval(
                    """SELECT MAX(f.kickoff_time) + INTERVAL '1 hour' FROM fixtures f
                       WHERE f.gameweek_id = (SELECT gameweek_id FROM gameweeks WHERE deadline_time > NOW()
                                              ORDER BY deadline_time LIMIT 1)""")
            try:
                await pipeline.run_job(pool, "odds", lambda: pipeline.ingest_odds(pool, odds, goalscorers_until=until))
            except Exception:  # odds are an enrichment; never block a deadline action on them
                pass


async def predict(pool: asyncpg.Pool) -> pipeline.JobResult:
    from app.ml.service import run_prediction

    async def job() -> pipeline.JobResult:
        s = await run_prediction(pool)
        return pipeline.JobResult(rows_written=s.players * len(s.gameweeks),
                                  details={"run_timestamp": s.run_timestamp.isoformat(), "gameweeks": s.gameweeks, "seconds": s.seconds})

    return await pipeline.run_job(pool, "predict", job)


async def optimize(pool: asyncpg.Pool, settings: Settings, team_id: int, free_transfers: int | None = None):
    from app.optimizer.service import run_optimization

    async with AsyncExitStack() as stack:
        fpl, _, _ = await build_sources(settings, stack)
        holder: dict[str, Any] = {}

        async def job() -> pipeline.JobResult:
            r = await run_optimization(pool, fpl, team_id, free_transfers=free_transfers)
            holder["result"] = r
            return pipeline.JobResult(rows_written=1, details={"run_id": r.run_id, "objective": round(r.plan.objective, 2)})

        await pipeline.run_job(pool, "optimize", job)
    return holder["result"]


async def plan_summary(pool: asyncpg.Pool, team_id: int, *, horizon: bool = False) -> str:
    """Human-readable summary of the latest optimisation run (Telegram / alerts)."""
    row = await pool.fetchrow(
        "SELECT * FROM optimization_runs WHERE team_id = $1 ORDER BY created_at DESC LIMIT 1", team_id)
    if row is None:
        return "No plan yet."
    from app.executor.payloads import pair_transfers

    rows = await pool.fetch("SELECT player_id, web_name, position FROM players")
    names = {r["player_id"]: r["web_name"] for r in rows}
    position = {r["player_id"]: r["position"] for r in rows}
    nm = lambda p: names.get(p, str(p))  # noqa: E731

    def pairs(outs: list[int], ins: list[int]) -> list[tuple[int, int]]:
        try:
            return pair_transfers(outs, ins, position)
        except (ValueError, KeyError):  # e.g. a Free Hit squad: fall back to listing order
            return list(zip(outs, ins))
    chip_label = {"3xc": "Triple Captain", "bboost": "Bench Boost", "freehit": "Free Hit", "wildcard": "Wildcard"}
    lines = []
    gws = row["plan"] if horizon else row["plan"][:1]
    for k, g in enumerate(gws):
        moves = ", ".join(f"{nm(o)} → {nm(i)}" for o, i in pairs(g["transfers_out"], g["transfers_in"])) or "no transfers"
        hit = f" (−{g['hits'] * 4})" if g["hits"] else ""
        chip = f" + {chip_label.get(g['chip'], g['chip'])}" if g.get("chip") else ""
        lines.append(f"GW{g['gameweek']}: {moves}{hit}{chip}; C {nm(g['captain'])}, V {nm(g['vice_captain'])}; xP {g['xp']:.1f}")
        if k == 0:
            bench = ", ".join(nm(p) for p in g["bench"])
            lines.append(f"  Bench: {bench}")
    gain = float(row["objective"]) - float(row["baseline_objective"])
    lines.append(f"Plan gain vs no moves: +{gain:.1f} xP over the horizon.")
    return "\n".join(lines)

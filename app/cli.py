"""Command line entry point.

    python -m app migrate                 # apply SQL migrations
    python -m app ingest core             # bootstrap-static + fixtures
    python -m app ingest history|entry|xg|odds|all
    python -m app ingest archive          # past seasons used as model training data
    python -m app predict                 # xP for the next 5 gameweeks -> player_predictions
    python -m app backtest --season 2025-26
    python -m app optimize [--refresh]    # MILP plan for your team (dry run, nothing is submitted)
    python -m app poll                    # long-running deadline-aware worker (ingestion + xP)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import sys
from contextlib import AsyncExitStack

from redis.asyncio import Redis

from app.config import Settings, get_settings
from app.db.migrate import migrate
from app.db.pool import create_pool
from app.ingestion import pipeline
from app.ingestion.factory import build_archive, build_sources
from app.ingestion.worker import IngestionWorker

log = logging.getLogger("app")

JOBS = ("core", "history", "entry", "xg", "odds", "archive")


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # httpx logs full request URLs at INFO, which would include API keys passed as query params.
    logging.getLogger("httpx").setLevel(logging.WARNING)


async def _ingest(settings: Settings, which: str) -> int:
    jobs = JOBS if which == "all" else (which,)
    pool = await create_pool(settings.database_url)
    failures = 0
    try:
        async with AsyncExitStack() as stack:
            fpl, xg, odds = await build_sources(settings, stack)
            for job in jobs:
                if job == "core":
                    fn = lambda: pipeline.ingest_core(pool, fpl)  # noqa: E731
                elif job == "history":
                    fn = lambda: pipeline.ingest_player_histories(  # noqa: E731
                        pool, fpl, concurrency=settings.history_concurrency,
                        only_with_minutes=settings.history_only_with_minutes,
                    )
                elif job == "entry":
                    if not settings.team_id:
                        log.warning("FPL_TEAM_ID not set; skipping entry")
                        continue
                    team_id = settings.team_id
                    fn = lambda: pipeline.ingest_entry(pool, fpl, team_id)  # noqa: E731
                elif job == "xg":
                    if xg is None:
                        log.warning("understat disabled; skipping xg")
                        continue
                    fn = lambda: pipeline.ingest_xg(pool, xg)  # noqa: E731
                elif job == "archive":
                    adapter = await build_archive(settings, stack)
                    seasons = settings.archive_season_list
                    fn = lambda: pipeline.ingest_archive(pool, adapter, seasons)  # noqa: E731
                else:
                    if odds is None:
                        log.warning("FPL_ODDS_API_KEY not set; skipping odds")
                        continue
                    fn = lambda: pipeline.ingest_odds(pool, odds)  # noqa: E731
                try:
                    result = await pipeline.run_job(pool, job, fn)
                    print(f"{job}: {result.rows_written} rows  {result.details}")
                except Exception as exc:
                    failures += 1
                    log.error("%s failed: %s", job, exc)
    finally:
        await pool.close()
    return 1 if failures else 0


async def _predict(settings: Settings) -> int:
    from app.ml.service import run_prediction

    pool = await create_pool(settings.database_url)
    try:
        result = await pipeline.run_job(pool, "predict", lambda: _predict_job(pool))
    finally:
        await pool.close()
    s = result.details
    print(f"xP run {s['run_timestamp']}: GW{s['gameweeks'][0]}-{s['gameweeks'][-1]}, {s['players']} players, {s['seconds']}s")
    print(f"Top projected for GW{s['gameweeks'][0]}:")
    for row in s["top"]:
        print(f"  {row['web_name']:<18} {row['position']}  xP {row['xp']:5.2f}  P(play) {row['prob_play']:.2f}")
    return 0


async def _predict_job(pool) -> pipeline.JobResult:
    from app.ml.service import run_prediction

    s = await run_prediction(pool)
    return pipeline.JobResult(
        rows_written=s.players * len(s.gameweeks),
        details={"run_timestamp": s.run_timestamp.isoformat(), "gameweeks": s.gameweeks, "players": s.players,
                 "seconds": s.seconds, "top": s.top},
    )


async def _backtest(settings: Settings, season: str, first: int, last: int, save: bool) -> int:
    import asyncpg

    from app.ml import MODEL_VERSION
    from app.ml.backtest import run_backtest
    from app.ml.data import load_model_data

    conn = await asyncpg.connect(settings.database_url)
    try:
        data = await load_model_data(conn)
        summary, _ = await asyncio.to_thread(run_backtest, data, season, first_gw=first, last_gw=last)
        m = summary["metrics"]
        for section in ("all_players", "when_played"):
            print(section)
            for name, v in m[section].items():
                print(f"  {name:<12} MAE {v['mae']:.3f}  RMSE {v['rmse']:.3f}  Spearman {v['spearman']:.3f}  bias {v['bias']:+.3f}  n={v['n']}")
        print("top-30 mean actual points:", {k: round(v, 2) for k, v in m["top30_mean_actual_points"].items()})
        print("P(start) calibration:", [(c["predicted"], c["actual"]) for c in m["p_start_calibration"]])
        print("total xP vs actual:", round(m["totals"]["expected_points"]), "vs", round(m["totals"]["actual_points"]))
        if save:
            await conn.execute(
                "INSERT INTO model_backtests (season, model_version, params, metrics) VALUES ($1, $2, $3::jsonb, $4::jsonb)",
                season, MODEL_VERSION, json.dumps(summary["params"]), json.dumps(m),
            )
    finally:
        await conn.close()
    return 0


def format_plan(result) -> str:
    n = result.names
    teams = result.team_short
    label = lambda p: f"{n[p]['name']} ({teams.get(n[p]['team_id'], '?')} {n[p]['position']} £{n[p]['cost'] / 10:.1f})"  # noqa: E731
    st, plan = result.state, result.plan
    lines = [
        f"Team {st.entry_id} | GW{plan.first.gameweek} | bank £{st.bank / 10:.1f}m | squad selling value £{st.squad_value / 10:.1f}m"
        f" | free transfers {st.free_transfers} | chips: {', '.join(c.name for c in st.chips_available) or 'none'}",
        f"Plan value {plan.objective:.1f} discounted xP vs {result.baseline.objective:.1f} with no transfers "
        f"(+{plan.objective - result.baseline.objective:.1f}); solver {plan.status} in {plan.solve_seconds:.1f}s",
    ]
    lines += [f"  warning: {w}" for w in st.warnings]
    lines.append("")
    for g in plan.gameweeks:
        moves = ", ".join(f"{n[o]['name']} -> {n[i]['name']}" for o, i in zip(
            sorted(g.transfers_out, key=lambda p: n[p]['position']), sorted(g.transfers_in, key=lambda p: n[p]['position']))) or "no transfers"
        hit = f", -{g.hits * 4} hit" if g.hits else ""
        chip = f" [{g.chip}]" if g.chip else ""
        lines.append(f"GW{g.gameweek}{chip}: {moves}{hit} | FT {g.free_transfers} | captain {n[g.captain]['name']} | xP {g.xp:.1f}")
    g = plan.first
    order = {"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}
    lines += ["", f"GW{g.gameweek} lineup:"]
    for p in sorted(g.starting_xi, key=lambda p: order[n[p]["position"]]):
        tag = " (C)" if p == g.captain else " (V)" if p == g.vice_captain else ""
        lines.append(f"  {label(p)}{tag}")
    lines.append("  Bench: " + ", ".join(f"{k}. {n[p]['name']}" for k, p in enumerate(g.bench, start=1)))
    lines += ["", "Chips:"]
    lines += [f"  {e.chip:<9} gain {e.gain:+6.2f}  {'USE' if e.triggered else 'hold'}  ({e.reason})" for e in result.chip_evaluations]
    if not result.chip_evaluations:
        lines.append("  none available")
    lines.append(f"\nLogged as optimization run {result.run_id} (execution_logs DRY_RUN {result.execution_log_id}).")
    return "\n".join(lines)


async def _optimize(settings: Settings, refresh: bool, free_transfers: int | None) -> int:
    from app.optimizer.service import run_optimization

    if not settings.team_id:
        log.error("set FPL_TEAM_ID to your FPL team id")
        return 2
    pool = await create_pool(settings.database_url)
    try:
        if refresh:
            async with AsyncExitStack() as stack:
                fpl, _, odds = await build_sources(settings, stack)
                await pipeline.run_job(pool, "core", lambda: pipeline.ingest_core(pool, fpl))
                if odds is not None:
                    await pipeline.run_job(pool, "odds", lambda: pipeline.ingest_odds(pool, odds))
            await pipeline.run_job(pool, "predict", lambda: _predict_job(pool))
        async with AsyncExitStack() as stack:
            fpl, _, _ = await build_sources(settings, stack)
            result = await pipeline.run_job(pool, "optimize", lambda: _optimize_job(pool, fpl, settings.team_id, free_transfers))
        print(format_plan(result.details["result"]))
    finally:
        await pool.close()
    return 0


async def _optimize_job(pool, fpl, team_id: int, free_transfers: int | None) -> pipeline.JobResult:
    from app.optimizer.service import run_optimization

    r = await run_optimization(pool, fpl, team_id, free_transfers=free_transfers)
    r.team_short = {row["team_id"]: row["short_name"] for row in await pool.fetch("SELECT team_id, short_name FROM teams")}
    job = pipeline.JobResult(rows_written=1, details={"run_id": r.run_id, "objective": round(r.plan.objective, 2)})
    job.details["result"] = r
    return job


async def _poll(settings: Settings) -> int:
    pool = await create_pool(settings.database_url)
    redis = Redis.from_url(settings.redis_url)
    try:
        async with AsyncExitStack() as stack:
            fpl, xg, odds = await build_sources(settings, stack)
            archive = await build_archive(settings, stack)
            worker = IngestionWorker(settings, pool, redis, fpl, xg_adapter=xg, odds_adapter=odds, archive_adapter=archive)
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                try:
                    loop.add_signal_handler(sig, worker.stop)
                except NotImplementedError:  # pragma: no cover - Windows
                    pass
            await worker.run_forever()
    finally:
        await redis.aclose()
        await pool.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fpl-manager")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("migrate", help="apply database migrations")
    ingest = sub.add_parser("ingest", help="run ingestion job(s) once")
    ingest.add_argument("job", choices=(*JOBS, "all"))
    sub.add_parser("poll", help="run the deadline-aware ingestion worker")
    sub.add_parser("predict", help="compute xP for the next gameweeks and store them")
    bt = sub.add_parser("backtest", help="walk-forward evaluation of the xP model on a past season")
    bt.add_argument("--season", default="2025-26")
    bt.add_argument("--from-gw", type=int, default=6)
    bt.add_argument("--to-gw", type=int, default=38)
    bt.add_argument("--save", action="store_true", help="store metrics in model_backtests")
    op = sub.add_parser("optimize", help="plan transfers, lineup, captaincy and chips for FPL_TEAM_ID (dry run)")
    op.add_argument("--refresh", action="store_true", help="re-ingest FPL data and odds and recompute xP first")
    op.add_argument("--free-transfers", type=int, default=None, help="override the reconstructed free transfer count")
    args = parser.parse_args(argv)

    _configure_logging(args.verbose)
    settings = get_settings()
    if args.command == "migrate":
        applied = asyncio.run(migrate(settings.database_url))
        print("applied: " + (", ".join(applied) if applied else "nothing (up to date)"))
        return 0
    if args.command == "ingest":
        return asyncio.run(_ingest(settings, args.job))
    if args.command == "optimize":
        return asyncio.run(_optimize(settings, args.refresh, args.free_transfers))
    if args.command == "predict":
        return asyncio.run(_predict(settings))
    if args.command == "backtest":
        return asyncio.run(_backtest(settings, args.season, args.from_gw, args.to_gw, args.save))
    return asyncio.run(_poll(settings))


if __name__ == "__main__":
    sys.exit(main())

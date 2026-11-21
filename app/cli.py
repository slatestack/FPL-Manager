"""Command line entry point.

    python -m app migrate                 # apply SQL migrations
    python -m app ingest core             # bootstrap-static + fixtures
    python -m app ingest history|entry|xg|odds|all
    python -m app ingest archive          # past seasons used as model training data
    python -m app predict                 # xP for the next 5 gameweeks -> player_predictions
    python -m app backtest --season 2025-26
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
    args = parser.parse_args(argv)

    _configure_logging(args.verbose)
    settings = get_settings()
    if args.command == "migrate":
        applied = asyncio.run(migrate(settings.database_url))
        print("applied: " + (", ".join(applied) if applied else "nothing (up to date)"))
        return 0
    if args.command == "ingest":
        return asyncio.run(_ingest(settings, args.job))
    if args.command == "predict":
        return asyncio.run(_predict(settings))
    if args.command == "backtest":
        return asyncio.run(_backtest(settings, args.season, args.from_gw, args.to_gw, args.save))
    return asyncio.run(_poll(settings))


if __name__ == "__main__":
    sys.exit(main())

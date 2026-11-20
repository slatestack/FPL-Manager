"""Command line entry point.

    python -m app migrate                 # apply SQL migrations
    python -m app ingest core             # bootstrap-static + fixtures
    python -m app ingest history|entry|xg|odds|all
    python -m app poll                    # long-running deadline-aware ingestion worker
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from contextlib import AsyncExitStack

from redis.asyncio import Redis

from app.config import Settings, get_settings
from app.db.migrate import migrate
from app.db.pool import create_pool
from app.ingestion import pipeline
from app.ingestion.factory import build_sources
from app.ingestion.worker import IngestionWorker

log = logging.getLogger("app")

JOBS = ("core", "history", "entry", "xg", "odds")


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


async def _poll(settings: Settings) -> int:
    pool = await create_pool(settings.database_url)
    redis = Redis.from_url(settings.redis_url)
    try:
        async with AsyncExitStack() as stack:
            fpl, xg, odds = await build_sources(settings, stack)
            worker = IngestionWorker(settings, pool, redis, fpl, xg_adapter=xg, odds_adapter=odds)
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
    args = parser.parse_args(argv)

    _configure_logging(args.verbose)
    settings = get_settings()
    if args.command == "migrate":
        applied = asyncio.run(migrate(settings.database_url))
        print("applied: " + (", ".join(applied) if applied else "nothing (up to date)"))
        return 0
    if args.command == "ingest":
        return asyncio.run(_ingest(settings, args.job))
    return asyncio.run(_poll(settings))


if __name__ == "__main__":
    sys.exit(main())

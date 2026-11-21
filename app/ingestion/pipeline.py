"""Ingestion jobs: fetch -> validate -> transform -> upsert, each audited in ``ingestion_runs``."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import asyncpg

from app.db.upsert import upsert_rows
from app.ingestion import odds_math, schemas, transform
from app.ingestion.adapters.archive import FPLArchiveAdapter
from app.ingestion.adapters.base import OddsAdapter, OddsSnapshot, XGAdapter
from app.ingestion.fpl_client import FPLClient
from app.ingestion.names import PlayerResolver, TeamResolver

log = logging.getLogger(__name__)


@dataclass
class JobResult:
    rows_written: int = 0
    payload_hash: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


def payload_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


async def run_job(
    pool: asyncpg.Pool, job: str, fn: Callable[[], Awaitable[JobResult]]
) -> JobResult:
    """Execute ``fn`` and record its outcome in ``ingestion_runs``. Re-raises failures."""
    run_id = await pool.fetchval("INSERT INTO ingestion_runs (job, status) VALUES ($1, 'RUNNING') RETURNING id", job)
    try:
        result = await fn()
    except Exception as exc:
        await pool.execute(
            "UPDATE ingestion_runs SET status='FAILED', finished_at=NOW(), error_message=$2 WHERE id=$1",
            run_id,
            f"{type(exc).__name__}: {exc}"[:4000],
        )
        raise
    await pool.execute(
        """UPDATE ingestion_runs SET status='SUCCESS', finished_at=NOW(), rows_written=$2, payload_hash=$3, details=$4
           WHERE id=$1""",
        run_id,
        result.rows_written,
        result.payload_hash,
        {k: v for k, v in result.details.items() if k != "result"},
    )
    log.info("job %s ok: %d rows %s", job, result.rows_written, result.details)
    return result


async def record_skipped(pool: asyncpg.Pool, job: str, reason: str) -> None:
    await pool.execute(
        "INSERT INTO ingestion_runs (job, status, finished_at, details) VALUES ($1, 'SKIPPED', NOW(), $2)",
        job,
        {"reason": reason},
    )


async def store_raw_snapshot(conn: asyncpg.Connection, source: str, payload: Any, digest: str) -> None:
    await conn.execute(
        """INSERT INTO raw_snapshots (source, payload_hash, payload) VALUES ($1, $2, $3)
           ON CONFLICT (source, payload_hash) DO UPDATE SET last_seen_at = NOW()""",
        source,
        digest,
        payload,
    )


async def prune_raw_snapshots(pool: asyncpg.Pool, retention_days: int) -> int:
    status = await pool.execute(
        "DELETE FROM raw_snapshots WHERE last_seen_at < NOW() - make_interval(days => $1)", retention_days
    )
    return int(status.split()[-1])


# ----------------------------------------------------------------------------- core FPL data


async def ingest_core(pool: asyncpg.Pool, fpl: FPLClient) -> JobResult:
    """bootstrap-static + fixtures -> teams, gameweeks, chips, players, fixtures (one transaction)."""
    raw_bootstrap, raw_fixtures = await asyncio.gather(fpl.bootstrap_static(), fpl.fixtures())
    bootstrap = schemas.BootstrapStatic.model_validate(raw_bootstrap)
    fixtures = [schemas.Fixture.model_validate(f) for f in raw_fixtures]

    teams = transform.team_rows(bootstrap)
    gameweeks = transform.gameweek_rows(bootstrap)
    chips = transform.chip_rows(bootstrap)
    players = transform.player_rows(bootstrap)
    fixture_rows = transform.fixture_rows(fixtures)

    b_hash, f_hash = payload_hash(raw_bootstrap), payload_hash(raw_fixtures)
    async with pool.acquire() as conn, conn.transaction():
        await store_raw_snapshot(conn, "bootstrap-static", raw_bootstrap, b_hash)
        await store_raw_snapshot(conn, "fixtures", raw_fixtures, f_hash)
        n = await upsert_rows(conn, "teams", teams, ["team_id"])
        n += await upsert_rows(conn, "gameweeks", gameweeks, ["gameweek_id"])
        n += await upsert_rows(conn, "chip_definitions", chips, ["chip_id"])
        n += await upsert_rows(conn, "players", players, ["player_id"])
        n += await upsert_rows(conn, "fixtures", fixture_rows, ["fixture_id"])

    return JobResult(
        rows_written=n,
        payload_hash=b_hash,
        details={
            "teams": len(teams),
            "gameweeks": len(gameweeks),
            "chips": len(chips),
            "players": len(players),
            "fixtures": len(fixture_rows),
            "season": transform.season_start_year(bootstrap),
            "fixtures_hash": f_hash,
        },
    )


async def ingest_player_histories(
    pool: asyncpg.Pool,
    fpl: FPLClient,
    *,
    concurrency: int = 6,
    only_with_minutes: bool = True,
    player_ids: list[int] | None = None,
) -> JobResult:
    """element-summary/{id} -> player_gw_history for every (active) player."""
    if player_ids is None:
        query = "SELECT player_id FROM players" + (" WHERE COALESCE(minutes, 0) > 0" if only_with_minutes else "")
        player_ids = [r["player_id"] for r in await pool.fetch(query + " ORDER BY player_id")]

    sem = asyncio.Semaphore(concurrency)
    failures: list[int] = []
    written = 0

    async def one(pid: int) -> None:
        nonlocal written
        async with sem:
            try:
                summary = schemas.ElementSummary.model_validate(await fpl.element_summary(pid))
            except Exception as exc:  # one bad player must not sink the batch
                log.warning("element-summary %s failed: %s", pid, exc)
                failures.append(pid)
                return
        rows = transform.history_rows(summary)
        async with pool.acquire() as conn, conn.transaction():
            n = await upsert_rows(conn, "player_gw_history", rows, ["player_id", "fixture_id"])
        # Not `written += await ...`: that reads `written` before the await and loses
        # increments made by concurrent tasks in the meantime.
        written += n

    await asyncio.gather(*(one(pid) for pid in player_ids))
    if player_ids and len(failures) == len(player_ids):
        raise RuntimeError(f"all {len(player_ids)} element-summary requests failed")
    return JobResult(
        rows_written=written,
        details={"players": len(player_ids), "failed": len(failures), "failed_ids": sorted(failures)[:50]},
    )


async def ingest_entry(pool: asyncpg.Pool, fpl: FPLClient, team_id: int) -> JobResult:
    entry = schemas.EntryProfile.model_validate(await fpl.entry(team_id))
    history = await fpl.entry_history(team_id)
    hist_rows = [
        {
            "entry_id": team_id, "gameweek": int(h["event"]), "points": int(h["points"]),
            "total_points": h.get("total_points"), "overall_rank": h.get("overall_rank"), "bank": h.get("bank"),
            "value": h.get("value"), "event_transfers": h.get("event_transfers"),
            "event_transfers_cost": h.get("event_transfers_cost"), "points_on_bench": h.get("points_on_bench"),
        }
        for h in history.get("current", [])
    ]
    async with pool.acquire() as conn:
        n = await upsert_rows(conn, "manager_entries", [transform.entry_row(entry)], ["entry_id"])
        n += await upsert_rows(conn, "manager_history", hist_rows, ["entry_id", "gameweek"])
    return JobResult(rows_written=n, details={"entry_id": entry.id, "current_event": entry.current_event})


# ----------------------------------------------------------------------------- enrichment


async def current_season(pool: asyncpg.Pool) -> int:
    first_deadline = await pool.fetchval("SELECT deadline_time FROM gameweeks ORDER BY gameweek_id LIMIT 1")
    if first_deadline is None:
        raise RuntimeError("no gameweeks ingested yet; run core ingestion first")
    return first_deadline.year if first_deadline.month >= 7 else first_deadline.year - 1


async def _resolvers(conn: asyncpg.Connection) -> tuple[TeamResolver, PlayerResolver]:
    teams = [dict(r) for r in await conn.fetch("SELECT team_id, name, short_name FROM teams")]
    players = [
        dict(r)
        for r in await conn.fetch("SELECT player_id, team_id, web_name, first_name, second_name FROM players")
    ]
    if not teams or not players:
        raise RuntimeError("teams/players are empty; run core ingestion first")
    return TeamResolver(teams), PlayerResolver(players)


async def ingest_xg(pool: asyncpg.Pool, adapter: XGAdapter, season: int | None = None) -> JobResult:
    season = season if season is not None else await current_season(pool)
    records = await adapter.fetch(season)
    async with pool.acquire() as conn, conn.transaction():
        team_resolver, player_resolver = await _resolvers(conn)
        known = {
            r["external_id"]: (r["player_id"], r["matched_by"])
            for r in await conn.fetch("SELECT external_id, player_id, matched_by FROM player_id_map WHERE source=$1", adapter.source)
        }
        stat_rows, map_rows, unmatched = [], [], []
        for rec in records:
            manual = known.get(rec.external_id)
            if manual and manual[1] == "manual":
                player_id = manual[0]
            else:
                team_ids = [t for t in (team_resolver.resolve(n) for n in rec.team_names) if t is not None]
                match = player_resolver.resolve(rec.name, team_ids)
                player_id = match.player_id if match else None
                if match:
                    map_rows.append((adapter.source, rec.external_id, match.player_id, rec.name, match.confidence, match.matched_by))
                else:
                    unmatched.append(rec.name)
            stat_rows.append(
                {
                    "source": adapter.source,
                    "season": season,
                    "external_id": rec.external_id,
                    "player_id": player_id,
                    "external_name": rec.name,
                    "team_name": ",".join(rec.team_names)[:128] or None,
                    "games": rec.games,
                    "minutes": rec.minutes,
                    "goals": rec.goals,
                    "npg": rec.npg,
                    "assists": rec.assists,
                    "shots": rec.shots,
                    "key_passes": rec.key_passes,
                    "xg": rec.xg,
                    "npxg": rec.npxg,
                    "xa": rec.xa,
                    "xg_chain": rec.xg_chain,
                    "xg_buildup": rec.xg_buildup,
                    "npxg_per90": round(rec.npxg_per90, 3) if rec.npxg_per90 is not None else None,
                    "xa_per90": round(rec.xa_per90, 3) if rec.xa_per90 is not None else None,
                }
            )
        if map_rows:
            await conn.executemany(
                """INSERT INTO player_id_map (source, external_id, player_id, external_name, confidence, matched_by)
                   VALUES ($1, $2, $3, $4, $5, $6)
                   ON CONFLICT (source, external_id) DO UPDATE
                   SET player_id=EXCLUDED.player_id, external_name=EXCLUDED.external_name,
                       confidence=EXCLUDED.confidence, matched_by=EXCLUDED.matched_by, updated_at=NOW()
                   WHERE player_id_map.matched_by <> 'manual'""",
                map_rows,
            )
        n = await upsert_rows(conn, "player_xg_stats", stat_rows, ["source", "season", "external_id"])
    return JobResult(
        rows_written=n,
        details={
            "source": adapter.source,
            "season": season,
            "records": len(records),
            "matched": len(records) - len(unmatched),
            "unmatched": len(unmatched),
            "unmatched_sample": sorted(unmatched)[:25],
        },
    )


async def _fixture_lookup(conn: asyncpg.Connection) -> list[asyncpg.Record]:
    return await conn.fetch(
        "SELECT fixture_id, team_h, team_a, kickoff_time FROM fixtures WHERE NOT finished AND kickoff_time IS NOT NULL"
    )


def _match_fixture(
    fixtures: list[asyncpg.Record], home_id: int | None, away_id: int | None, commence: datetime
) -> int | None:
    if home_id is None or away_id is None:
        return None
    window = timedelta(days=3)
    candidates = [
        f for f in fixtures
        if f["team_h"] == home_id and f["team_a"] == away_id and abs(f["kickoff_time"] - commence) <= window
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda f: abs(f["kickoff_time"] - commence))["fixture_id"]


async def ingest_odds(pool: asyncpg.Pool, adapter: OddsAdapter) -> JobResult:
    snapshot: OddsSnapshot = await adapter.fetch()
    async with pool.acquire() as conn, conn.transaction():
        team_resolver, player_resolver = await _resolvers(conn)
        fixtures = await _fixture_lookup(conn)

        fixture_rows, unmatched_fixtures = [], []
        for rec in snapshot.fixtures:
            home_id, away_id = team_resolver.resolve(rec.home_team), team_resolver.resolve(rec.away_team)
            fixture_id = _match_fixture(fixtures, home_id, away_id, rec.commence_time)
            if fixture_id is None:
                unmatched_fixtures.append(f"{rec.home_team} v {rec.away_team}")
                continue
            fit = odds_math.fit_goal_rates(rec.p_home, rec.p_draw, rec.p_away, rec.p_over_2_5)
            fixture_rows.append(
                {
                    "fixture_id": fixture_id,
                    "source": snapshot.source,
                    "captured_at": snapshot.captured_at,
                    "bookmaker_count": rec.bookmaker_count,
                    "p_home": round(rec.p_home, 4),
                    "p_draw": round(rec.p_draw, 4),
                    "p_away": round(rec.p_away, 4),
                    "p_over_2_5": round(rec.p_over_2_5, 4) if rec.p_over_2_5 is not None else None,
                    "lambda_home": round(fit.lambda_home, 3),
                    "lambda_away": round(fit.lambda_away, 3),
                    "p_cs_home": round(fit.p_cs_home, 4),
                    "p_cs_away": round(fit.p_cs_away, 4),
                }
            )

        scorer_rows, unmatched_players = [], []
        for rec in snapshot.goalscorers:
            home_id, away_id = team_resolver.resolve(rec.home_team), team_resolver.resolve(rec.away_team)
            fixture_id = _match_fixture(fixtures, home_id, away_id, rec.commence_time)
            match = player_resolver.resolve(rec.player_name, [home_id, away_id]) if fixture_id else None
            if match is None:
                unmatched_players.append(rec.player_name)
                continue
            scorer_rows.append(
                {
                    "player_id": match.player_id,
                    "fixture_id": fixture_id,
                    "source": snapshot.source,
                    "captured_at": snapshot.captured_at,
                    "bookmaker_count": rec.bookmaker_count,
                    "median_decimal_odds": round(rec.median_decimal_odds, 2),
                    "p_anytime_goal": round(rec.p_anytime_goal, 4),
                }
            )
        # Defensive: a player resolved twice for one fixture keeps the first occurrence.
        seen: set[tuple[int, int]] = set()
        scorer_rows = [r for r in scorer_rows if not ((r["player_id"], r["fixture_id"]) in seen or seen.add((r["player_id"], r["fixture_id"])))]

        n = await upsert_rows(conn, "fixture_odds", fixture_rows, ["fixture_id", "source", "captured_at"], touch_updated_at=False)
        n += await upsert_rows(
            conn, "player_goal_odds", scorer_rows, ["player_id", "fixture_id", "source", "captured_at"], touch_updated_at=False
        )
    return JobResult(
        rows_written=n,
        details={
            "source": snapshot.source,
            "fixtures": len(fixture_rows),
            "unmatched_fixtures": unmatched_fixtures[:25],
            "goalscorers": len(scorer_rows),
            "unmatched_goalscorers": len(unmatched_players),
        },
    )


# ----------------------------------------------------------------------------- past seasons (training data)


async def ingest_archive(pool: asyncpg.Pool, adapter: FPLArchiveAdapter, seasons: list[str]) -> JobResult:
    details: dict[str, Any] = {}
    total = 0
    for season in seasons:
        data = await adapter.fetch(season)
        async with pool.acquire() as conn, conn.transaction():
            n = await upsert_rows(conn, "archive_players", data.players, ["season", "element"])
            n += await upsert_rows(conn, "archive_fixtures", data.fixtures, ["season", "fixture_id"])
            # Chunked: a season is ~30k rows.
            for i in range(0, len(data.matches), 5000):
                n += await upsert_rows(
                    conn, "archive_player_matches", data.matches[i : i + 5000], ["season", "element", "fixture_id"]
                )
        details[season] = {"players": len(data.players), "fixtures": len(data.fixtures), "matches": len(data.matches)}
        total += n
    return JobResult(rows_written=total, details=details)

# FPL Manager

Autonomous Fantasy Premier League manager, built in phases from the system architecture blueprint.

| Phase | Scope | Status |
| --- | --- | --- |
| 1 | Foundation: project layout, Docker Compose (PostgreSQL 16 + Redis), Section 8 schema | ✅ |
| 2 | Ingestion & feature pipeline: FPL fetchers, xG + odds enrichment adapters, deadline-aware polling | ✅ |
| 3 | xP prediction model | — |
| 4 | MILP squad / chip optimiser | — |
| 5 | Authenticated executor & guardrails | — |
| 6 | Scheduler, control-plane API, alerts | — |

## Layout

```
app/
  config.py            settings from FPL_* environment variables
  cli.py               `python -m app migrate | ingest <job> | poll`
  db/                  asyncpg pool, migration runner, generic upsert
  ingestion/
    http.py            retrying HTTP client (exponential backoff + jitter, Retry-After, 429/5xx)
    fpl_client.py      bootstrap-static, fixtures, element-summary, entry
    schemas.py         Pydantic validation of upstream payloads
    transform.py       payload -> row mapping (pure)
    pipeline.py        jobs: core, history, entry, xg, odds (audited in ingestion_runs)
    schedule.py        Section 4.2 polling windows (pure)
    worker.py          long-running deadline-aware worker with Redis job locks
    names.py           external team/player name -> FPL id resolution
    odds_math.py       de-vig, consensus, Poisson goal-rate fit -> clean-sheet probability
    adapters/          Understat (xG/xA) and The Odds API (1X2, O/U 2.5, anytime scorer)
  api/main.py          FastAPI: /healthz, /api/v1/ingestion/status
  ml/ optimizer/ executor/   placeholders for phases 3-5
migrations/            0001 = Section 8 DDL verbatim, 0002 = ingestion tables
```

## Running

```bash
cp .env.example .env
docker compose up --build
```

This starts Postgres and Redis, applies migrations, runs the ingestion worker and serves the API on `:8000`.

Without Docker:

```bash
python -m venv .venv && . .venv/bin/activate && pip install -e ".[dev]"
```

```bash
python -m app migrate
```

```bash
python -m app ingest all
```

```bash
python -m app poll
```

## Data model (Phase 2)

* **Core (every poll):** `teams`, `gameweeks`, `chip_definitions`, `players` (Section 8 columns plus additive
  feature columns), `fixtures`. Written in one transaction; rows only rewritten when something changed.
* **Change tracking:** a trigger on `players` appends to `player_price_history` and `player_status_history`.
* **History:** `player_gw_history` from `element-summary` (one row per player per fixture, so double
  gameweeks keep both matches). View `player_rolling_form` gives last-5 xG/xA per 90, minutes and BPS.
* **Enrichment:** `player_xg_stats` (Understat season npxG/xA/shots/key passes) with `player_id_map`
  (automatic matches; set `matched_by='manual'` to pin a mapping). `fixture_odds` stores consensus
  de-vigged 1X2 / O/U 2.5 probabilities, fitted Poisson goal rates and clean-sheet probabilities;
  `player_goal_odds` stores anytime-scorer odds. Both are append-only snapshots (`latest_fixture_odds` view).
* **Audit:** `ingestion_runs` logs every job (status, rows, hash, errors); `raw_snapshots` keeps
  deduplicated raw upstream payloads for `FPL_RAW_SNAPSHOT_RETENTION_DAYS` (default 14).

## Polling cadence (Section 4.2)

| Window | Core poll | Odds (if key set) |
| --- | --- | --- |
| > 24h before deadline | 6h | daily |
| T-24h … T-2h | 30 min | every `FPL_ODDS_MIN_INTERVAL_MINUTES` (30) |
| T-2h … T-10m | 2 min | every 30 min |
| T-10m … T | sleep until just after the deadline | — |

Sleeps are clipped to window boundaries, so the worker never sleeps past T-24h or T-2h. Player history and
Understat refresh every 12h (history also as soon as FPL marks a gameweek `data_checked`); the entry profile every 6h.

## Tests

```bash
pytest
```

Unit tests need nothing. Integration tests run when these point at a disposable database (its `public` schema is dropped) and a Redis DB (it is flushed):

```bash
TEST_DATABASE_URL=postgresql://fpl:fpl@localhost/fpl_test TEST_REDIS_URL=redis://localhost:6379/15 pytest
```

## Notes

* Anytime-goalscorer probabilities are raw `1 / median odds` and still include bookmaker margin; there's
  no complete market to de-vig. Phase 3 should calibrate them.
* Understat's site layout has changed before. The adapter tries the JSON endpoint
  (`/getLeagueData/EPL/{season}`) first and falls back to the `playersData` blob embedded in the league page.

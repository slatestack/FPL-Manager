# FPL Manager

Autonomous Fantasy Premier League manager, built in phases from the system architecture blueprint.

| Phase | Scope | Status |
| --- | --- | --- |
| 1 | Foundation: project layout, Docker Compose (PostgreSQL 16 + Redis), Section 8 schema | ✅ |
| 2 | Ingestion & feature pipeline: FPL fetchers, xG + odds enrichment adapters, deadline-aware polling | ✅ |
| 3 | xP prediction model (Section 5), validated by walk-forward backtest | ✅ |
| 4 | MILP squad, transfer, lineup, captaincy and chip optimiser (Section 6) | ✅ |
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
  ml/
    scoring.py         FPL scoring rules (values read live from game_config.scoring)
    data.py            unified match/fixture frames across seasons (keyed by stable codes)
    team_strength.py   Poisson attack/defence ratings -> team goal rates (blended with odds)
    minutes.py         LightGBM P(start) / P(appear) by horizon
    rates.py           shrunk per-player shares of team xG/xA, saves, defensive actions, cards
    bonus.py           LightGBM bonus model, evaluated over simulated matches
    availability.py    injury/suspension flags and return dates -> P(available) per GW
    engine.py          combines everything into per-fixture and per-GW xP
    service.py         live run -> prediction_runs / player_predictions / fixture_predictions
    backtest.py        walk-forward evaluation against form baselines
  optimizer/
    team_state.py      squad, bank, real selling prices, free transfers, chips (from public endpoints)
    model.py           multi-period MILP (OR-Tools; CBC by default, SCIP/HiGHS supported)
    chips.py           Section 6.4 counterfactual solves and trigger rules
    validate.py        independent rule check of every plan before it is stored
    service.py         run -> optimization_runs + execution_logs (DRY_RUN)
  api/main.py          FastAPI: health, ingestion status, predictions, optimize dry-run/latest
  executor/            placeholder for phase 5
migrations/            0001 = Section 8 DDL verbatim, 0002 = ingestion, 0003 = archive + predictions, 0004 = optimiser runs
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

## Expected points model (Phase 3)

`python -m app predict` (and the worker, automatically) writes xP for every player for the next
5 gameweeks to `player_predictions`, with the components the spec names (`xp_attacking`,
`xp_defensive`, `xp_bonus`) plus appearance, deductions, P(start), expected minutes, goals,
assists and clean-sheet probability. Each run is recorded in `prediction_runs` with all
parameters and fingerprints of the input data, and is deterministic: the same inputs give
identical output.

How it works, per player and upcoming fixture:

1. **Team goal rates.** Poisson attack/defence ratings fitted to a blend of xG and goals, with
   recency weighting (last season down-weighted). Where bookmaker odds exist, their implied
   goal rates are blended in (70% odds). This drives goals, assists, clean sheets, goals
   conceded and saves for both sides.
2. **Playing time.** LightGBM classifiers trained on 2+ seasons predict P(start) and P(appear)
   for the k-th upcoming match from recent selection pattern and price. These are multiplied by
   availability from FPL status, chance of playing, and return dates parsed from the news.
3. **Points.** Three scenarios (start and play 60+, start and go off earlier, come off the bench).
   Within each one, goals and assists come from the player's shrunk share of team xG/xA × the
   team rate. Clean sheets, goals conceded (−1 per 2), saves (1 per 3, per opponent xG faced)
   and defensive contributions (DEF ≥ 10, MID/FWD ≥ 12, negative binomial) have exact
   closed forms. Cards, own goals and penalty misses/saves use per-90 rates.
4. **Bonus.** LightGBM learns bonus from realised match stats, and is averaged over 300
   simulated matches per fixture.
5. **Gameweeks.** Double gameweeks sum their fixtures; blank gameweeks are explicit zero rows.

The scoring rules were verified by reproducing FPL's `total_points` exactly for every 2026/27
match played so far (3,084 of 3,084).

### Backtest (2025/26, GW6–38, each gameweek predicted only from earlier data)

`python -m app backtest --season 2025-26`

| | Model | Season pts/match | Last-5 form |
| --- | --- | --- | --- |
| RMSE, all registered players | **1.92** | 2.05 | 2.10 |
| RMSE, players who played | **2.87** | 3.01 | 3.21 |
| Rank correlation, players who played | **0.356** | 0.275 | 0.273 |
| Mean actual points of each method's top 30 per GW | **4.43** | 3.87 | 3.73 |

Total predicted points are within 1.4% of actual, and P(start) is calibrated across all deciles.
No historical injury news exists, so the backtest runs every method with availability = 1. Live
runs also use FPL's injury flags. Results barely move when the main constants change (±0.002 RMSE),
so the defaults are not tuned to this one season.

## Optimiser (Phase 4)

`python -m app optimize` (add `--refresh` to re-ingest and recompute xP first), or
`POST /api/v1/optimize/dry-run`, plans the next 5 gameweeks for `FPL_TEAM_ID`. Nothing is
submitted to FPL; execution is Phase 5.

* **Model:** Section 6 MILP: squad, starting XI, ordered bench, captain and vice for every
  gameweek, plus transfers in each week. Free transfers bank up to 5. The objective is
  discounted xP (γ = 0.85) minus 4 per hit, plus small bench and vice-captain terms.
  The spec's `min/max` rollover formula is rewritten in solver-compatible form (see
  `model.py`). Hits are also capped at `system_config.max_hits_per_gw`, so plans never exceed
  what the Phase 5 guardrail allows.
* **Money:** a bank balance carried week to week, using your real selling prices (FPL keeps half
  of a rise). Purchase prices come from your public transfer history.
* **Team state:** free transfers are replayed from your history and cross-checked against the
  hit points FPL actually charged. Chip windows come from `bootstrap-static`. Phase 5's
  authenticated `my-team` call will replace these reconstructions.
* **Chips:** each available chip is solved as a counterfactual and used only if it meets the
  Section 6.4 thresholds, scaled by `chip_aggression_profile`. A chip in the last gameweek
  of its window is used if it adds anything at all (configurable).
* **Safety and audit:** every plan is re-checked by an independent validator (squad and formation rules,
  ≤ 3 per club, budget, free-transfer rollover, hit cap) before it is stored. Each run logs inputs, parameters,
  solver statistics, the full plan and chip analysis to `optimization_runs`, and the next-gameweek
  decision to `execution_logs` as a `DRY_RUN` with its payload hash.

A run solves 5 MILPs (plan, do-nothing baseline, three chips): about 25–45 seconds in total.
CBC, SCIP and HiGHS reach the identical optimum on live data.

## Polling cadence (Section 4.2)

| Window | Core poll | Odds (if key set) |
| --- | --- | --- |
| > 24h before deadline | 6h | daily |
| T-24h … T-2h | 30 min | every `FPL_ODDS_MIN_INTERVAL_MINUTES` (30) |
| T-2h … T-10m | 2 min | every 30 min |
| T-10m … T | sleep until just after the deadline | — |

Sleeps are clipped to window boundaries, so the worker never sleeps past T-24h or T-2h. Player history and
Understat refresh every 12h (history also as soon as FPL marks a gameweek `data_checked`); the entry profile every 6h.
xP is recomputed every 6h, every 30 min inside T-24h, and every 10 min inside T-2h. Past seasons (training data)
are refreshed weekly.

## Tests

```bash
pytest
```

Unit tests need nothing. Integration tests run when these point at a disposable database (its `public` schema is dropped) and a Redis DB (it is flushed):

```bash
TEST_DATABASE_URL=postgresql://fpl:fpl@localhost/fpl_test TEST_REDIS_URL=redis://localhost:6379/15 pytest
```

## Notes

* Model limitations: no historical injury data exists for training, so a player just back from injury can look
  less likely to start than he is until he plays again. Understat data is stored but not yet used by the model
  (FPL's own per-match xG is used instead).
* Anytime-goalscorer probabilities are raw `1 / median odds` and still include bookmaker margin; there's
  no complete market to de-vig. Phase 3 should calibrate them.
* Understat's site layout has changed before. The adapter tries the JSON endpoint
  (`/getLeagueData/EPL/{season}`) first and falls back to the `playersData` blob embedded in the league page.

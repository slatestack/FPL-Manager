# FPL Manager

Autonomous Fantasy Premier League manager, built in phases from the system architecture blueprint.

| Phase | Scope | Status |
| --- | --- | --- |
| 1 | Foundation: project layout, Docker Compose (PostgreSQL 16 + Redis), Section 8 schema | ✅ |
| 2 | Ingestion & feature pipeline: FPL fetchers, xG + odds enrichment adapters, deadline-aware polling | ✅ |
| 3 | xP prediction model (Section 5), validated by walk-forward backtest | ✅ |
| 4 | MILP squad, transfer, lineup, captaincy and chip optimiser (Section 6) | ✅ |
| 5 | Authenticated executor, token vault, guardrails, post-commit verification (Section 7) | ✅ |
| 6 | Deadline scheduler, Telegram alerts + kill switch, control-plane API, dashboard (Sections 7.2, 9) | ✅ |
| 7 | Full-season simulation of the decision system | ✅ |
| 8 | Bookmaker goalscorer odds, chip forecasting (Triple Captain, Bench Boost) | ✅ |

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
  api/main.py          FastAPI: dashboard + health, ingestion status, predictions, optimize dry-run/latest
  api/static/          the web dashboard (served at /)
  executor/
    vault.py           encrypted FPL token store + OIDC refresh (no password ever handled)
    client.py          authenticated my-team / transfers / lineup calls (writes never blindly retried)
    payloads.py        plan -> FPL request bodies
    guardrails.py      Section 7.3 circuit breakers (+ deadline, freshness, squad parity, live FT checks)
    service.py         commit pipeline (lock, guardrails, POSTs, verification) and keep-alive
  scheduler.py         T-24h dry run, T-30m final lock, T-15m commit (once per gameweek)
  notify/              Telegram / webhook alerts; telegram_bot.py handles /killswitch etc.
  ops.py               shared refresh / predict / optimise / summary operations
migrations/            0001 = Section 8 DDL verbatim, 0002 = ingestion, 0003 = archive + predictions, 0004 = optimiser runs
```

## Dashboard

Open `http://localhost:8000/` (the `api` service in Docker Compose, or `uvicorn app.api.main:app`). It shows:
- the next deadline with a live countdown
- your planned GW lineup on a pitch, with new signings, captaincy and bench order
- the five-week transfer plan
- the chip analysis
- a sortable, filterable table of every player's projections and fixtures
- the status of each pipeline job

**Run dry-run** re-solves the plan on the latest projections. Nothing is ever submitted to FPL
from the dashboard. It is plain HTML/CSS/JS served by FastAPI, with no build step or external
assets, and supports dark mode and phones.

## Run it on your own computer

Everything (database, worker, dashboard) runs in Docker on one computer. Nothing is reachable
from other machines: the dashboard listens on `localhost` only.

**You need:** Docker Desktop ([download](https://www.docker.com/products/docker-desktop/)), set to
*start when you sign in*, and a copy of this repository (`git clone`, or GitHub's *Download ZIP*).

**Install** from the repository folder:

* macOS / Linux: `./scripts/install.sh`
* Windows (PowerShell): `powershell -ExecutionPolicy Bypass -File scripts\install.ps1`

The installer:
1. asks for your team ID and, optionally, your Odds API key and WhatsApp number for alerts
2. generates the secret keys into `.env` (readable only by you)
3. builds and starts everything
4. walks you through signing in to FPL (a normal sign-in link; paste back the address you land on)
5. asks whether to turn on automatic execution

Re-running it is safe: it keeps your `.env`.

**Keep it running around deadlines.** The computer must be on and awake from about 30 minutes
before each deadline. On macOS, turn on *Prevent automatic sleeping when the display is off*
(Battery or Energy settings, when on power). On Windows, set *Sleep* to *Never* when plugged in.
If the computer was asleep through the commit window, you get an alert, and nothing is submitted
late.

**Everyday commands** (from the repository folder):

| What | Command |
| --- | --- |
| Dashboard | open http://localhost:8000 |
| Turn automatic execution off / on | `docker compose exec ingestion-worker python -m app config --auto-execute off` (or `on`) |
| Check the FPL sign-in | `docker compose exec ingestion-worker python -m app auth status` |
| Sign in again (every ~180 days; you get reminders) | `docker compose exec ingestion-worker python -m app auth login` |
| See the current plan | `docker compose exec ingestion-worker python -m app optimize` |
| Rehearse the next submission (sends nothing) | `docker compose exec ingestion-worker python -m app execute --check` |
| Logs | `docker compose logs -f ingestion-worker` |
| Update to a new version | `git pull` then `docker compose up -d --build` |
| Stop everything | `docker compose down` (your data is kept) |

Your data lives in Docker volumes (`pgdata`, `redisdata`) and your settings in `.env`. Back up
`.env` if you move machines; the stored FPL sign-in can only be decrypted with its `FPL_SECRET_KEY`.

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
4. **Bookmaker goalscorer odds.** For the coming gameweek, anytime-scorer prices from UK
   bookmakers (including the Betfair exchange) are converted to scoring rates. Each team is scaled
   so its rates, weighted by P(play), add up to the team's expected goals, which removes the
   bookmaker margin. The result is blended 60/40 with the model's share of team goals. Odds are
   fetched at the T−24h and T−30m re-plans only (~1 credit per match). A reserve guard stops
   spending before the monthly credit allowance runs out.
5. **Bonus.** LightGBM learns bonus from realised match stats, and is averaged over 300
   simulated matches per fixture.
6. **Gameweeks.** Double gameweeks sum their fixtures; blank gameweeks are explicit zero rows.

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

## Autonomous execution (Phase 5)

Execution is **off by default**. A fresh install never touches your team until you arm it, and arming
requires an FPL login that has passed a live check.

FPL signs in through an OAuth/OIDC flow with bot protection, so the system never sees your
password. It uses the token pair your logged-in browser already holds and renews it itself.

1. **Encryption key:** `python -m app auth keygen`, then put the output in `.env` as `FPL_SECRET_KEY`.
2. **Import tokens:** log in at fantasy.premierleague.com. Open DevTools → Application → Local
   Storage → `https://fantasy.premierleague.com` and copy the value of the key starting with
   `oidc.user:https://account.premierleague.com`. Then run `python -m app auth set` and paste it
   (input is hidden). Tokens are stored encrypted, and `auth status` checks them against FPL.
3. **Arm:** `python -m app config --auto-execute on`. Alternatives: `/enable` in Telegram, or the
   dashboard. Disarm with `--auto-execute off`, `/killswitch`, the Telegram button or the dashboard.

What a commit does: it takes a per-gameweek Redis lock and loads the latest plan, then runs the
guardrails on live data. Transfers go first (`POST transfers/`), then lineup, captaincy and chip
(`POST my-team/{id}/`). Finally it re-reads `my-team` and checks that it matches the plan exactly.
Every step is logged in `execution_logs` (`details` holds the guardrail report, request bodies and
FPL responses).

| Guardrail | Aborts when |
| --- | --- |
| Kill switch | `system_config.auto_execute_enabled` is false |
| Hit cap | hits at the **live** free-transfer count exceed `max_hits_per_gw` (Wildcard/Free Hit exempt) |
| Budget drift | live selling prices + current prices leave the bank negative |
| Duplicate | an identical payload already succeeded for this gameweek |
| Deadline | less than 3 minutes to the deadline, or the plan is for another gameweek |
| Plan freshness | the plan is more than 45 minutes old |
| Squad parity | your live squad differs from the plan's starting squad (you changed it by hand) |
| Chip | the planned chip is not available |

Writes are never blindly retried. If a POST fails ambiguously, the live state decides: a commit whose
transfers already went through skips them on retry. Unconfirmed lineups and every abort raise an
urgent alert. When tokens are stored, the optimiser also uses the exact `my-team` state
(selling prices, free transfers, chip status) instead of the public reconstruction.

## Scheduler, alerts and control plane (Phase 6)

`python -m app poll` runs everything: ingestion, xP, a login keep-alive every 4 h, alerts and these
deadline actions (each once per gameweek, tracked in `scheduled_actions`). The worker
wakes up exactly at each boundary:

| When | Action |
| --- | --- |
| T−24h | refresh data, xP, optimise; send the plan (with a kill-switch button when armed) |
| T−30m | final data lock: latest news and odds, xP, optimise; message if the plan changed |
| T−15m | commit if armed (guardrails apply); otherwise a reminder with the moves to make by hand |
| missed | if the worker was down through the commit window, an urgent alert |

**Alerts.** Plans, receipts and problems are sent to every configured channel; test with
`python -m app notify-test`.

* **WhatsApp** (one-way) through the free CallMeBot relay. Add +34 644 66 32 62 to your phone's
  contacts and send it *I allow callmebot to send me messages*. It replies with an API key. Then set
  `FPL_WHATSAPP_PHONE` (with country code) and `FPL_WHATSAPP_APIKEY`. WhatsApp can't take commands
  back, so the kill switch is on the dashboard.
* **Telegram** (two-way): only your `FPL_TELEGRAM_CHAT_ID` is obeyed. Commands are `/status`, `/plan`,
  `/killswitch`, `/enable`, `/dryrun`, `/login` and `/help`.
* **Webhook:** `FPL_ALERT_WEBHOOK_URL` receives `{"text", "urgent"}`.

| Method | Endpoint | Auth |
| --- | --- | --- |
| `GET` | `/api/v1/status/{team_id}` | — |
| `POST` | `/api/v1/optimize/dry-run` | — |
| `POST` | `/api/v1/execute/force` | `Authorization: Bearer $FPL_API_TOKEN` |
| `PATCH` | `/api/v1/config/{team_id}` (kill switch, hit cap, chip profile, `auth_tokens`) | `Authorization: Bearer $FPL_API_TOKEN` |
| `GET` | `/api/v1/audit/{team_id}` (executions with projected vs actual points) | — |

Mutating endpoints are disabled entirely when `FPL_API_TOKEN` is unset. Tokens are never returned
by any endpoint. Read endpoints expose your plan, so do not put the API on the public internet
without a reverse proxy that adds authentication.

## Season simulation (Phase 7)

`python -m app simulate --season 2025-26` replays a past season as if the system had managed a team
from GW1. Each week it trains only on data from before that deadline, predicts ahead, runs the
optimiser and chip logic on the simulated team's state, and scores the decision with the real
points (auto-subs, vice-captain, chips, hits, selling prices, free-transfer rules).
`--baseline` keeps the GW1 squad all season; `--no-chip-forecast` times chips with thresholds only.

2025/26 results:

| | Points | Per week | Chips used |
| --- | --- | --- | --- |
| Full system with chip forecast | **2,023** | 53.2 | 6 of 8 (TC GW17 Haaland 16 pts) |
| Full system, threshold-only chips | 1,938 | 51.0 | 3 of 8 (TC wasted in GW38) |
| Same GW1 squad all season | 1,390 | 36.6 | none |

The simulation is pessimistic. No historical injury news or player odds exist, so it can't see that
a flagged player will miss a match. Realised points were 6% below prediction, almost all of it in the
first half of the season. Open weakness: the Wildcard is only played when it is about to expire,
and the Free Hit was never used.

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

* Automating your account may conflict with FPL's terms of use, and FPL can change its login or API
  without notice. Every failure mode aborts with an alert rather than guessing, but you remain
  responsible for your team.
* Model limitations: no historical injury data exists for training, so a player just back from injury can look
  less likely to start than he is until he plays again. Understat data is stored but not yet used by the model
  (FPL's own per-match xG is used instead).
* Anytime-goalscorer probabilities are raw `1 / median odds` and still include bookmaker margin; there's
  no complete market to de-vig. Phase 3 should calibrate them.
* Understat's site layout has changed before. The adapter tries the JSON endpoint
  (`/getLeagueData/EPL/{season}`) first and falls back to the `playersData` blob embedded in the league page.

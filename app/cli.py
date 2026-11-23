"""Command line entry point.

    python -m app migrate                 # apply SQL migrations
    python -m app ingest core             # bootstrap-static + fixtures
    python -m app ingest history|entry|xg|odds|all
    python -m app ingest archive          # past seasons used as model training data
    python -m app predict                 # xP for the next 5 gameweeks -> player_predictions
    python -m app backtest --season 2025-26
    python -m app optimize [--refresh]    # MILP plan for your team (dry run, nothing is submitted)
    python -m app auth keygen|login|status|refresh|set   # FPL sign-in + token vault (Phase 5)
    python -m app config --auto-execute on|off [--max-hits N] [--aggression BALANCED]
    python -m app execute [--ignore-kill-switch]    # commit the latest plan now (guardrails apply)
    python -m app poll                    # worker: ingestion, xP, deadline actions, Telegram bot
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
        from app.executor.payloads import pair_transfers

        try:
            pairs = pair_transfers(g.transfers_out, g.transfers_in, {p: n[p]["position"] for p in n})
        except ValueError:
            pairs = list(zip(g.transfers_out, g.transfers_in))
        moves = ", ".join(f"{n[o]['name']} -> {n[i]['name']}" for o, i in pairs) or "no transfers"
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
    labels = {"3xc": "Triple Captain", "bboost": "Bench Boost"}
    for f in result.chip_forecasts:
        weeks = "  ".join(f"GW{w.gameweek} {w.value:.1f}{'*' if w.double else ''}" for w in f.weeks)
        lines.append(f"  {labels[f.chip]} forecast (window to GW{f.window_end}): {weeks}")
        lines.append(f"    -> {'use now' if f.recommendation == 'use_now' else 'hold' if f.recommendation == 'hold' else 'threshold decides'}: {f.reason}")
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


async def _simulate(settings: Settings, season: str, last_gw: int, baseline: bool, no_forecast: bool = False) -> int:
    import dataclasses
    import json as _json
    from pathlib import Path

    import asyncpg

    from app.ml.data import load_model_data
    from app.sim.season import SimConfig, simulate

    conn = await asyncpg.connect(settings.database_url)
    try:
        data = await load_model_data(conn)
        data.web_names = {r["code"]: r["web_name"] for r in await conn.fetch(
            "SELECT DISTINCT ON (code) code, web_name FROM archive_players WHERE season = $1", season)}
    finally:
        await conn.close()
    cfg = SimConfig(season=season, last_gw=last_gw, transfers=not baseline, chip_forecast=not no_forecast,
                    label="baseline" if baseline else ("system-no-forecast" if no_forecast else "system"))

    def progress(rec, res) -> None:
        moves = ", ".join(f"{o}→{i}" for o, i in rec.transfers) or "-"
        chip = f" [{rec.chip}]" if rec.chip else ""
        hit = f" -{rec.hits * 4}" if rec.hits else ""
        print(f"GW{rec.gameweek:>2}{chip}: {rec.points:>3} pts{hit} (pred {rec.predicted_xp:5.1f}) total {res.total:>4} | "
              f"C {rec.captain} {rec.captain_points} | {moves} | {rec.seconds}s", flush=True)

    result = await asyncio.to_thread(simulate, data, cfg, progress=progress)
    out = Path("sim_results")
    out.mkdir(exist_ok=True)
    path = out / f"{season}-{cfg.label}.json"
    path.write_text(_json.dumps({"summary": result.summary(),
                                 "gameweeks": [dataclasses.asdict(g) for g in result.gameweeks]}, indent=1, default=str))
    print(_json.dumps(result.summary()), f"\nsaved {path}")
    return 0


async def _poll(settings: Settings) -> int:
    from app.executor.service import Executor
    from app.notify import TelegramNotifier, build_notifier
    from app.notify.telegram_bot import TelegramBot
    from app.scheduler import DeadlineScheduler

    pool = await create_pool(settings.database_url)
    redis = Redis.from_url(settings.redis_url)
    stop = asyncio.Event()
    try:
        async with AsyncExitStack() as stack:
            fpl, xg, odds = await build_sources(settings, stack)
            archive = await build_archive(settings, stack)
            notifier = build_notifier(settings)
            executor = Executor(settings, pool, redis, notifier) if settings.secret_key else None
            scheduler = DeadlineScheduler(settings, pool, notifier, executor=executor) if settings.team_id else None
            worker = IngestionWorker(settings, pool, redis, fpl, xg_adapter=xg, odds_adapter=odds, archive_adapter=archive,
                                     scheduler=scheduler, executor=executor)
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                try:
                    loop.add_signal_handler(sig, lambda: (worker.stop(), stop.set()))
                except NotImplementedError:  # pragma: no cover - Windows
                    pass
            tasks = [asyncio.create_task(worker.run_forever())]
            if settings.telegram_bot_token and settings.telegram_chat_id and settings.team_id:
                bot = TelegramBot(settings, pool, redis, TelegramNotifier(settings.telegram_bot_token, settings.telegram_chat_id))
                tasks.append(asyncio.create_task(bot.run_forever(stop)))
            await asyncio.gather(*tasks)
    finally:
        await redis.aclose()
        await pool.close()
    return 0


async def _notify_test(settings: Settings) -> int:
    from app.notify import LogNotifier, build_notifier

    notifier = build_notifier(settings)
    channels = [type(n).__name__.removesuffix("Notifier") for n in notifier.notifiers if not isinstance(n, LogNotifier)]
    if not channels:
        print("No alert channel configured (set FPL_WHATSAPP_PHONE and FPL_WHATSAPP_APIKEY in .env).")
        return 1
    failures = 0
    for n in notifier.notifiers:
        if isinstance(n, LogNotifier):
            continue
        try:
            await n.send("✅ FPL Manager test message: alerts are working.")
            print(f"{type(n).__name__.removesuffix('Notifier')}: sent")
        except Exception as exc:
            failures += 1
            print(f"{type(n).__name__.removesuffix('Notifier')}: FAILED ({exc})")
    return 1 if failures else 0


async def _auth(settings: Settings, action: str) -> int:
    import getpass

    from app.executor.vault import AuthError, TokenVault, parse_token_input

    if action == "keygen":
        print(TokenVault.generate_key())
        print("Put this in .env as FPL_SECRET_KEY=... and keep it secret.", file=sys.stderr)
        return 0
    if not settings.team_id:
        log.error("set FPL_TEAM_ID first")
        return 2
    pool = await create_pool(settings.database_url)
    try:
        vault = TokenVault(settings, pool)
        if action == "set":
            print("Paste the value of the localStorage key starting with 'oidc.user:https://account.premierleague.com'",
                  "from fantasy.premierleague.com (DevTools > Application > Local Storage), then press Enter.", file=sys.stderr)
            raw = sys.stdin.readline() if not sys.stdin.isatty() else getpass.getpass("tokens (hidden): ")
            tokens = parse_token_input(raw)
            await vault.store(settings.team_id, tokens)
            print(f"stored; access token valid until {tokens.expires_at}; refresh token {'present' if tokens.refresh_token else 'MISSING'}")
            action = "status"
        if action == "login":
            from app.executor.login import complete_login, start_login

            redis = Redis.from_url(settings.redis_url)
            try:
                url = await start_login(settings, redis, settings.team_id)
                print("1. Open this link and sign in to FPL as usual:\n\n" + url + "\n", file=sys.stderr)
                print("2. You will land on fantasy.premierleague.com. Copy the full address from the address bar", file=sys.stderr)
                print("   (it contains ?code=...). If the address bar no longer shows it, open your browser history", file=sys.stderr)
                print("   (Ctrl+H) and copy the newest fantasy.premierleague.com/?code=... entry. Paste it here within a few minutes:", file=sys.stderr)
                redirect = sys.stdin.readline()
                tokens = await complete_login(settings, redis, vault, redirect)
            finally:
                await redis.aclose()
            print(f"signed in; session valid until {tokens.session_expires_at:%d %b %Y}; access token renews automatically")
            action = "status"
        if action == "refresh":
            tokens = await vault.load(settings.team_id)
            if tokens is None:
                raise AuthError("no tokens stored")
            tokens = await vault.refresh(settings.team_id, tokens)
            print(f"refreshed; valid until {tokens.expires_at}")
            return 0
        from app.executor.service import Executor
        from app.notify import LogNotifier

        redis = Redis.from_url(settings.redis_url)
        try:
            ok = await Executor(settings, pool, redis, LogNotifier()).keepalive(settings.team_id)
        finally:
            await redis.aclose()
        h = await vault.health(settings.team_id)
        ends = h.get("session_expires_at")
        print(f"login {'OK' if ok else 'FAILING'}; tokens stored: {h.get('has_tokens')}; access renews automatically; "
              f"session ends {ends:%d %b %Y} (sign in again before then)" if ok and ends else
              f"login {'OK' if ok else 'FAILING'}; tokens stored: {h.get('has_tokens')}; expires {h.get('auth_expires_at')}"
              + (f"; error: {h.get('auth_last_error')}" if not ok else ""))
        return 0 if ok else 1
    except AuthError as exc:
        log.error("%s", exc)
        return 1
    finally:
        await pool.close()


async def _config(settings: Settings, auto: str | None, max_hits: int | None, aggression: str | None) -> int:
    if not settings.team_id:
        log.error("set FPL_TEAM_ID first")
        return 2
    pool = await create_pool(settings.database_url)
    try:
        await pool.execute("INSERT INTO system_config (team_id, auto_execute_enabled) VALUES ($1, FALSE) ON CONFLICT DO NOTHING",
                           settings.team_id)
        if auto is not None:
            if auto == "on" and not await pool.fetchval(
                    "SELECT encrypted_auth_cookie IS NOT NULL AND auth_last_check_ok IS TRUE FROM system_config WHERE team_id=$1",
                    settings.team_id):
                log.error("FPL login must pass a live check first (`python -m app auth set`, then `auth status`)")
                return 1
            await pool.execute("UPDATE system_config SET auto_execute_enabled=$2, updated_at=NOW() WHERE team_id=$1",
                               settings.team_id, auto == "on")
        if max_hits is not None:
            await pool.execute("UPDATE system_config SET max_hits_per_gw=$2 WHERE team_id=$1", settings.team_id, max_hits)
        if aggression is not None:
            await pool.execute("UPDATE system_config SET chip_aggression_profile=$2 WHERE team_id=$1", settings.team_id, aggression)
        row = await pool.fetchrow("SELECT auto_execute_enabled, max_hits_per_gw, chip_aggression_profile FROM system_config WHERE team_id=$1",
                                  settings.team_id)
        print(f"auto-execute {'ON' if row[0] else 'OFF'}; max hits per GW {row[1]}; chip profile {row[2]}")
    finally:
        await pool.close()
    return 0


async def _preflight(settings: Settings) -> int:
    from app.executor.service import Executor
    from app.notify import LogNotifier

    pool = await create_pool(settings.database_url)
    redis = Redis.from_url(settings.redis_url)
    try:
        r = await Executor(settings, pool, redis, LogNotifier()).preflight(settings.team_id)
        names = {x["player_id"]: x["web_name"] for x in await pool.fetch("SELECT player_id, web_name FROM players")}
    finally:
        await redis.aclose()
        await pool.close()
    if "guardrails" not in r:
        print(f"preflight: {r['reason']}")
        return 1
    print(f"Preflight for GW{r['gameweek']} (nothing is submitted):")
    timing = {"kill_switch", "deadline", "plan_fresh"}
    for c in r["guardrails"]:
        mark = "ok" if c["ok"] else ("note" if c["name"] in timing else "FAIL")
        print(f"  [{mark}] {c['name']}: {c['detail']}")
    tr = r.get("transfers_request")
    if tr:
        print("  transfers request:", ", ".join(f"{names[t['element_out']]} (sell £{t['selling_price'] / 10:.1f}) -> "
                                               f"{names[t['element_in']]} (buy £{t['purchase_price'] / 10:.1f})" for t in tr["transfers"]),
              f"| chip {tr['chip']}")
    pk = r["picks_request"]
    xi = [names[p["element"]] + (" (C)" if p["is_captain"] else " (V)" if p["is_vice_captain"] else "") for p in pk["picks"][:11]]
    print("  lineup request: XI", ", ".join(xi), "| bench", ", ".join(names[p["element"]] for p in pk["picks"][11:]), f"| chip {pk['chip']}")
    print("RESULT:", "ready: a commit would be submitted with exactly these requests" if r["ok"] else "a commit would be BLOCKED")
    return 0 if r["ok"] else 1


async def _execute(settings: Settings, ignore_kill_switch: bool) -> int:
    from app.executor.service import Executor
    from app.notify import build_notifier

    if not settings.team_id:
        log.error("set FPL_TEAM_ID first")
        return 2
    pool = await create_pool(settings.database_url)
    redis = Redis.from_url(settings.redis_url)
    try:
        result = await Executor(settings, pool, redis, build_notifier(settings)).commit(
            settings.team_id, trigger="manual", respect_kill_switch=not ignore_kill_switch)
        print(f"{result.status}: {result.reason}")
        for c in result.report:
            print(f"  [{'ok' if c['ok'] else 'FAIL'}] {c['name']}: {c['detail']}")
        return 0 if result.status == "SUCCESS" else 1
    finally:
        await redis.aclose()
        await pool.close()


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
    sub.add_parser("notify-test", help="send a test message to every configured alert channel")
    au = sub.add_parser("auth", help="manage the FPL token vault")
    au.add_argument("action", choices=("keygen", "login", "set", "status", "refresh"))
    cf = sub.add_parser("config", help="kill switch and execution settings")
    cf.add_argument("--auto-execute", choices=("on", "off"))
    cf.add_argument("--max-hits", type=int)
    cf.add_argument("--aggression", choices=("CONSERVATIVE", "BALANCED", "AGGRESSIVE"))
    ex = sub.add_parser("execute", help="commit the latest plan to FPL now (guardrails apply)")
    ex.add_argument("--ignore-kill-switch", action="store_true", help="manual override of auto_execute_enabled only")
    ex.add_argument("--check", action="store_true", help="preflight: run every check and build the requests, submit nothing")
    sm = sub.add_parser("simulate", help="replay a past season with the full decision system (Phase 7)")
    sm.add_argument("--season", default="2025-26")
    sm.add_argument("--to-gw", type=int, default=38)
    sm.add_argument("--baseline", action="store_true", help="keep the GW1 squad all season (no transfers or chips)")
    sm.add_argument("--no-chip-forecast", action="store_true", help="time chips with thresholds only (the old behaviour)")
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
    if args.command == "notify-test":
        return asyncio.run(_notify_test(settings))
    if args.command == "auth":
        return asyncio.run(_auth(settings, args.action))
    if args.command == "config":
        return asyncio.run(_config(settings, args.auto_execute, args.max_hits, args.aggression))
    if args.command == "execute":
        if args.check:
            return asyncio.run(_preflight(settings))
        return asyncio.run(_execute(settings, args.ignore_kill_switch))
    if args.command == "simulate":
        return asyncio.run(_simulate(settings, args.season, args.to_gw, args.baseline, args.no_chip_forecast))
    if args.command == "optimize":
        return asyncio.run(_optimize(settings, args.refresh, args.free_transfers))
    if args.command == "predict":
        return asyncio.run(_predict(settings))
    if args.command == "backtest":
        return asyncio.run(_backtest(settings, args.season, args.from_gw, args.to_gw, args.save))
    return asyncio.run(_poll(settings))


if __name__ == "__main__":
    sys.exit(main())

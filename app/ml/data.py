"""Load model inputs from PostgreSQL into tidy pandas frames keyed by stable FPL ``code``s.

FPL re-numbers player and team ids each season; ``code`` is stable, so the live season
(``player_gw_history``) and past seasons (``archive_*``) are unified on codes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import datetime

import asyncpg
import numpy as np
import pandas as pd

from app.ml.scoring import ScoringRules

STAT_COLS = [
    "minutes", "starts", "total_points", "goals_scored", "assists", "clean_sheets", "goals_conceded",
    "own_goals", "penalties_saved", "penalties_missed", "yellow_cards", "red_cards", "saves", "bonus", "bps",
    "defensive_contribution", "expected_goals", "expected_assists", "expected_goals_conceded", "value",
]

_CURRENT_MATCHES_SQL = """
SELECT h.player_id AS element, h.fixture_id, h.gameweek_id AS gameweek, h.was_home, h.kickoff_time AS kickoff,
       {stats}, NULL::numeric AS fpl_xp,
       p.code AS player_code, p.position,
       CASE WHEN h.was_home THEN th.code ELSE ta.code END AS team_code,
       CASE WHEN h.was_home THEN ta.code ELSE th.code END AS opp_code,
       CASE WHEN h.was_home THEN f.team_h_score ELSE f.team_a_score END AS team_goals,
       CASE WHEN h.was_home THEN f.team_a_score ELSE f.team_h_score END AS opp_goals
FROM player_gw_history h
JOIN players p USING (player_id)
JOIN fixtures f ON f.fixture_id = h.fixture_id
JOIN teams th ON th.team_id = f.team_h
JOIN teams ta ON ta.team_id = f.team_a
"""

_ARCHIVE_MATCHES_SQL = """
SELECT m.season, m.element, m.fixture_id, m.gameweek_id AS gameweek, m.was_home, m.kickoff_time AS kickoff,
       {stats}, m.fpl_xp,
       ap.code AS player_code, ap.position,
       CASE WHEN m.was_home THEN f.team_h_code ELSE f.team_a_code END AS team_code,
       CASE WHEN m.was_home THEN f.team_a_code ELSE f.team_h_code END AS opp_code,
       CASE WHEN m.was_home THEN f.team_h_score ELSE f.team_a_score END AS team_goals,
       CASE WHEN m.was_home THEN f.team_a_score ELSE f.team_h_score END AS opp_goals
FROM archive_player_matches m
JOIN archive_players ap USING (season, element)
JOIN archive_fixtures f USING (season, fixture_id)
WHERE m.season <> $1
"""


def season_label(start_year: int) -> str:
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def season_start(label: str) -> int:
    return int(label[:4])


@dataclass
class ModelData:
    """Everything the model learns from. ``matches`` = one row per player per team fixture."""

    matches: pd.DataFrame
    fixtures: pd.DataFrame          # all seasons, incl. unplayed current-season fixtures
    current_season: str
    scoring: ScoringRules

    @property
    def current_idx(self) -> int:
        return season_start(self.current_season)

    def before(self, cutoff: pd.Timestamp, *, season: str | None = None) -> "ModelData":
        """Data as it was known just before ``cutoff`` (results after it are hidden)."""
        m = self.matches[self.matches["kickoff"] < cutoff]
        f = self.fixtures.copy()
        future = ~(f["kickoff"] < cutoff)
        f.loc[future, ["finished"]] = False
        f.loc[future, ["home_goals", "away_goals", "home_xg", "away_xg"]] = np.nan
        return replace(self, matches=m, fixtures=f, current_season=season or self.current_season)


def _frame(records: list[asyncpg.Record]) -> pd.DataFrame:
    df = pd.DataFrame([dict(r) for r in records])
    for c in df.columns:
        if df[c].dtype == object and len(df) and df[c].map(lambda v: v.__class__.__name__ == "Decimal").any():
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    return df


def _finalise(matches: pd.DataFrame, fixtures: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    matches = matches.copy()
    matches["kickoff"] = pd.to_datetime(matches["kickoff"], utc=True)
    matches["season_idx"] = matches["season"].map(season_start)
    matches["fixture_uid"] = matches["season"] + ":" + matches["fixture_id"].astype(str)
    for c in STAT_COLS + ["fpl_xp", "team_goals", "opp_goals"]:
        matches[c] = pd.to_numeric(matches[c], errors="coerce").astype(float)
    # Team xG per fixture = sum over that team's players.
    txg = matches.groupby(["fixture_uid", "team_code"])["expected_goals"].sum(min_count=1).rename("team_xg")
    matches = matches.join(txg, on=["fixture_uid", "team_code"])
    opp = txg.rename("opp_xg").reset_index().rename(columns={"team_code": "opp_code"})
    matches = matches.merge(opp, on=["fixture_uid", "opp_code"], how="left")
    matches = matches.sort_values(["player_code", "kickoff", "fixture_id"], kind="mergesort").reset_index(drop=True)

    fixtures = fixtures.copy()
    fixtures["kickoff"] = pd.to_datetime(fixtures["kickoff"], utc=True)
    fixtures["season_idx"] = fixtures["season"].map(season_start)
    fixtures["fixture_uid"] = fixtures["season"] + ":" + fixtures["fixture_id"].astype(str)
    for c in ("home_goals", "away_goals"):
        fixtures[c] = pd.to_numeric(fixtures[c], errors="coerce").astype(float)
    h = txg.rename("home_xg").reset_index().rename(columns={"team_code": "home_code"})
    a = txg.rename("away_xg").reset_index().rename(columns={"team_code": "away_code"})
    fixtures = fixtures.merge(h, on=["fixture_uid", "home_code"], how="left").merge(a, on=["fixture_uid", "away_code"], how="left")
    return matches, fixtures


async def load_scoring(conn: asyncpg.Connection) -> ScoringRules:
    cfg = await conn.fetchval(
        """SELECT payload -> 'game_config' -> 'scoring' FROM raw_snapshots
           WHERE source = 'bootstrap-static' ORDER BY last_seen_at DESC LIMIT 1"""
    )
    if isinstance(cfg, str):  # connection without the pool's JSON codec
        cfg = json.loads(cfg)
    return ScoringRules.from_game_config(cfg)


async def load_model_data(conn: asyncpg.Connection) -> ModelData:
    first_deadline: datetime | None = await conn.fetchval("SELECT MIN(deadline_time) FROM gameweeks")
    if first_deadline is None:
        raise RuntimeError("no gameweeks ingested; run core ingestion first")
    current = season_label(first_deadline.year if first_deadline.month >= 7 else first_deadline.year - 1)

    stats_h = ", ".join(f"h.{c}" for c in STAT_COLS)
    stats_m = ", ".join(f"m.{c}" for c in STAT_COLS)
    cur = _frame(await conn.fetch(_CURRENT_MATCHES_SQL.format(stats=stats_h)))
    if len(cur):
        cur.insert(0, "season", current)
    arc = _frame(await conn.fetch(_ARCHIVE_MATCHES_SQL.format(stats=stats_m), current))
    matches = pd.concat([d for d in (arc, cur) if len(d)], ignore_index=True)

    cur_fx = _frame(
        await conn.fetch(
            """SELECT $1::text AS season, f.fixture_id, f.gameweek_id AS gameweek, th.code AS home_code, ta.code AS away_code,
                      f.kickoff_time AS kickoff, f.finished, f.team_h_score AS home_goals, f.team_a_score AS away_goals
               FROM fixtures f JOIN teams th ON th.team_id = f.team_h JOIN teams ta ON ta.team_id = f.team_a""",
            current,
        )
    )
    arc_fx = _frame(
        await conn.fetch(
            """SELECT season, fixture_id, gameweek_id AS gameweek, team_h_code AS home_code, team_a_code AS away_code,
                      kickoff_time AS kickoff, finished, team_h_score AS home_goals, team_a_score AS away_goals
               FROM archive_fixtures WHERE season <> $1""",
            current,
        )
    )
    fixtures = pd.concat([d for d in (arc_fx, cur_fx) if len(d)], ignore_index=True)
    matches, fixtures = _finalise(matches, fixtures)
    return ModelData(matches=matches, fixtures=fixtures, current_season=current, scoring=await load_scoring(conn))


async def load_current_players(conn: asyncpg.Connection) -> pd.DataFrame:
    df = _frame(
        await conn.fetch(
            """SELECT p.player_id, p.code, p.web_name, p.team_id, t.code AS team_code, p.position,
                      p.now_cost AS price, p.status, p.chance_of_playing, p.news, p.news_added
               FROM players p JOIN teams t ON t.team_id = p.team_id"""
        )
    )
    df["price"] = df["price"].astype(float)
    return df


async def load_target_fixtures(conn: asyncpg.Connection, gameweeks: list[int], season: str) -> pd.DataFrame:
    df = _frame(
        await conn.fetch(
            """SELECT f.fixture_id, f.gameweek_id AS gameweek, th.code AS home_code, ta.code AS away_code,
                      f.kickoff_time AS kickoff
               FROM fixtures f JOIN teams th ON th.team_id = f.team_h JOIN teams ta ON ta.team_id = f.team_a
               WHERE f.gameweek_id = ANY($1::int[])
               ORDER BY f.kickoff_time, f.fixture_id""",
            gameweeks,
        )
    )
    if len(df):
        df["kickoff"] = pd.to_datetime(df["kickoff"], utc=True)
        df["fixture_uid"] = season + ":" + df["fixture_id"].astype(str)
    return df


async def load_latest_odds(conn: asyncpg.Connection, as_of: datetime) -> pd.DataFrame:
    return _frame(
        await conn.fetch(
            """SELECT DISTINCT ON (fixture_id) fixture_id, lambda_home AS odds_lh, lambda_away AS odds_la, captured_at
               FROM fixture_odds WHERE captured_at <= $1 ORDER BY fixture_id, captured_at DESC""",
            as_of,
        )
    )

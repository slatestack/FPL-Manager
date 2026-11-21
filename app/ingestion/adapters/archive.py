"""Past-season per-match data from the public FPL archive (github.com/vaastav/Fantasy-Premier-League).

Used only as model training / backtest data. Rows are normalised to the same shape as the
live ``player_gw_history`` table plus stable cross-season ``code`` identifiers for players
and teams, since FPL re-numbers player and team ids every season.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

from app.ingestion.http import ResilientClient

log = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
_POSITIONS = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}

_INT_COLS = [
    "minutes", "total_points", "goals_scored", "assists", "clean_sheets", "goals_conceded", "own_goals",
    "penalties_saved", "penalties_missed", "yellow_cards", "red_cards", "saves", "bonus", "bps",
]


@dataclass
class ArchiveSeason:
    season: str
    players: list[dict[str, Any]]
    fixtures: list[dict[str, Any]]
    matches: list[dict[str, Any]]


def _none(v: Any) -> Any:
    return None if pd.isna(v) else v


def parse_season(season: str, merged_gw: str, players_raw: str, teams: str, fixtures: str) -> ArchiveSeason:
    t = pd.read_csv(io.StringIO(teams))
    team_code = dict(zip(t["id"].astype(int), t["code"].astype(int)))

    p = pd.read_csv(io.StringIO(players_raw))
    p = p[p["element_type"].isin(_POSITIONS)]
    players = [
        {
            "season": season,
            "element": int(r.id),
            "code": int(r.code),
            "web_name": str(r.web_name)[:64],
            "position": _POSITIONS[int(r.element_type)],
            "team_code": int(r.team_code),
        }
        for r in p.itertuples()
    ]
    elements = {pl["element"] for pl in players}

    f = pd.read_csv(io.StringIO(fixtures))
    fx = []
    for r in f.itertuples():
        fx.append(
            {
                "season": season,
                "fixture_id": int(r.id),
                "gameweek_id": int(r.event) if not pd.isna(r.event) else None,
                "team_h_code": team_code[int(r.team_h)],
                "team_a_code": team_code[int(r.team_a)],
                "kickoff_time": pd.Timestamp(r.kickoff_time).to_pydatetime() if not pd.isna(r.kickoff_time) else None,
                "finished": bool(r.finished),
                "team_h_score": int(r.team_h_score) if not pd.isna(r.team_h_score) else None,
                "team_a_score": int(r.team_a_score) if not pd.isna(r.team_a_score) else None,
            }
        )

    m = pd.read_csv(io.StringIO(merged_gw))
    m = m[m["element"].isin(elements)].drop_duplicates(["element", "fixture"], keep="last")
    has_dc = "defensive_contribution" in m.columns
    has_xp = "xP" in m.columns
    matches = []
    for r in m.to_dict("records"):
        row = {
            "season": season,
            "element": int(r["element"]),
            "fixture_id": int(r["fixture"]),
            "gameweek_id": int(r["GW"] if "GW" in r else r["round"]),
            "was_home": bool(r["was_home"]),
            "kickoff_time": pd.Timestamp(r["kickoff_time"]).to_pydatetime(),
            "starts": int(r["starts"]) if "starts" in r and not pd.isna(r["starts"]) else None,
            "defensive_contribution": int(r["defensive_contribution"]) if has_dc and not pd.isna(r["defensive_contribution"]) else None,
            "expected_goals": _none(r.get("expected_goals")),
            "expected_assists": _none(r.get("expected_assists")),
            "expected_goals_conceded": _none(r.get("expected_goals_conceded")),
            "value": int(r["value"]) if not pd.isna(r.get("value")) else None,
            "fpl_xp": _none(r.get("xP")) if has_xp else None,
        }
        for c in _INT_COLS:
            row[c] = int(r[c])
        matches.append(row)
    # Stable column order for the bulk upsert.
    order = ["season", "element", "fixture_id", "gameweek_id", "was_home", "kickoff_time", *_INT_COLS,
             "starts", "defensive_contribution", "expected_goals", "expected_assists",
             "expected_goals_conceded", "value", "fpl_xp"]
    matches = [{k: row[k] for k in order} for row in matches]
    return ArchiveSeason(season=season, players=players, fixtures=fx, matches=matches)


class FPLArchiveAdapter:
    source = "fpl-archive"

    def __init__(self, http: ResilientClient, *, base_url: str = DEFAULT_BASE_URL):
        self._http = http
        self._base = base_url.rstrip("/")

    async def fetch(self, season: str) -> ArchiveSeason:
        base = f"{self._base}/{season}"
        merged = await self._http.get_text(f"{base}/gws/merged_gw.csv")
        players = await self._http.get_text(f"{base}/players_raw.csv")
        teams = await self._http.get_text(f"{base}/teams.csv")
        fixtures = await self._http.get_text(f"{base}/fixtures.csv")
        return parse_season(season, merged, players, teams, fixtures)

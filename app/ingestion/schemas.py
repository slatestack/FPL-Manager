"""Pydantic models validating the subset of FPL payload fields the system relies on.

Unknown fields are ignored so additive upstream changes don't break ingestion, but a
missing or mistyped required field fails loudly before anything is written.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, field_validator


class _Model(BaseModel):
    model_config = ConfigDict(extra="ignore")


def _blank_to_none(v: object) -> object:
    return None if v in ("", None) else v


class ElementType(_Model):
    id: int
    singular_name_short: str


class Team(_Model):
    id: int
    code: int
    name: str
    short_name: str
    strength: int | None = None
    strength_overall_home: int | None = None
    strength_overall_away: int | None = None
    strength_attack_home: int | None = None
    strength_attack_away: int | None = None
    strength_defence_home: int | None = None
    strength_defence_away: int | None = None


class Event(_Model):
    id: int
    name: str
    deadline_time: datetime
    is_previous: bool = False
    is_current: bool = False
    is_next: bool = False
    finished: bool = False
    data_checked: bool = False
    average_entry_score: int | None = None
    highest_score: int | None = None


class Chip(_Model):
    id: int
    name: str
    chip_type: str | None = None
    start_event: int | None = None
    stop_event: int | None = None
    number: int | None = None


class Element(_Model):
    id: int
    code: int | None = None
    web_name: str
    first_name: str | None = None
    second_name: str | None = None
    team: int
    element_type: int
    now_cost: int
    status: str
    chance_of_playing_next_round: int | None = None
    chance_of_playing_this_round: int | None = None
    news: str | None = None
    news_added: datetime | None = None
    selected_by_percent: Decimal | None = None
    form: Decimal | None = None
    points_per_game: Decimal | None = None
    ep_next: Decimal | None = None
    total_points: int | None = None
    minutes: int | None = None
    starts: int | None = None
    expected_goals: Decimal | None = None
    expected_assists: Decimal | None = None
    expected_goals_conceded: Decimal | None = None
    penalties_order: int | None = None
    corners_and_indirect_freekicks_order: int | None = None
    direct_freekicks_order: int | None = None
    cost_change_event: int | None = None
    transfers_in_event: int | None = None
    transfers_out_event: int | None = None

    _blank = field_validator(
        "selected_by_percent", "form", "points_per_game", "ep_next",
        "expected_goals", "expected_assists", "expected_goals_conceded", "news_added",
        mode="before",
    )(_blank_to_none)


class BootstrapStatic(_Model):
    events: list[Event]
    teams: list[Team]
    elements: list[Element]
    element_types: list[ElementType]
    chips: list[Chip] = []


class Fixture(_Model):
    id: int
    code: int | None = None
    event: int | None = None
    team_h: int
    team_a: int
    team_h_difficulty: int | None = None
    team_a_difficulty: int | None = None
    kickoff_time: datetime | None = None
    started: bool | None = None
    finished: bool = False
    finished_provisional: bool = False
    team_h_score: int | None = None
    team_a_score: int | None = None


class HistoryRow(_Model):
    element: int
    fixture: int
    round: int
    opponent_team: int
    was_home: bool
    kickoff_time: datetime | None = None
    minutes: int
    starts: int | None = None
    total_points: int
    goals_scored: int
    assists: int
    clean_sheets: int
    goals_conceded: int
    own_goals: int = 0
    penalties_saved: int = 0
    penalties_missed: int = 0
    yellow_cards: int = 0
    red_cards: int = 0
    saves: int = 0
    bonus: int = 0
    bps: int = 0
    defensive_contribution: int | None = None
    expected_goals: Decimal | None = None
    expected_assists: Decimal | None = None
    expected_goal_involvements: Decimal | None = None
    expected_goals_conceded: Decimal | None = None
    value: int | None = None
    selected: int | None = None

    _blank = field_validator(
        "expected_goals", "expected_assists", "expected_goal_involvements", "expected_goals_conceded",
        mode="before",
    )(_blank_to_none)


class ElementSummary(_Model):
    history: list[HistoryRow] = []


class EntryProfile(_Model):
    id: int
    name: str | None = None
    player_first_name: str | None = None
    player_last_name: str | None = None
    current_event: int | None = None
    summary_overall_points: int | None = None
    summary_overall_rank: int | None = None
    summary_event_points: int | None = None
    last_deadline_bank: int | None = None
    last_deadline_value: int | None = None
    last_deadline_total_transfers: int | None = None

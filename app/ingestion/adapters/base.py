"""Adapter contracts. Adapters return source-native names; the pipeline maps them to FPL ids."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol


@dataclass(frozen=True)
class XGRecord:
    external_id: str
    name: str
    team_names: tuple[str, ...]   # a player who moved clubs mid-season lists several
    games: int
    minutes: int
    goals: int
    npg: int
    assists: int
    shots: int
    key_passes: int
    xg: float
    npxg: float
    xa: float
    xg_chain: float | None = None
    xg_buildup: float | None = None

    @property
    def npxg_per90(self) -> float | None:
        return self.npxg * 90.0 / self.minutes if self.minutes else None

    @property
    def xa_per90(self) -> float | None:
        return self.xa * 90.0 / self.minutes if self.minutes else None


@dataclass(frozen=True)
class FixtureOddsRecord:
    home_team: str
    away_team: str
    commence_time: datetime
    bookmaker_count: int
    p_home: float
    p_draw: float
    p_away: float
    p_over_2_5: float | None
    event_id: str | None = None


@dataclass(frozen=True)
class PlayerGoalOddsRecord:
    player_name: str
    home_team: str
    away_team: str
    commence_time: datetime
    bookmaker_count: int
    median_decimal_odds: float

    @property
    def p_anytime_goal(self) -> float:
        # Raw implied probability; still contains bookmaker margin (no complete market to de-vig).
        return 1.0 / self.median_decimal_odds


@dataclass
class OddsSnapshot:
    source: str
    captured_at: datetime
    fixtures: list[FixtureOddsRecord] = field(default_factory=list)
    goalscorers: list[PlayerGoalOddsRecord] = field(default_factory=list)


class XGAdapter(Protocol):
    source: str

    async def fetch(self, season: int) -> list[XGRecord]: ...


class OddsAdapter(Protocol):
    source: str

    async def fetch(self) -> OddsSnapshot: ...

"""FPL scoring rules.

Point values come from ``bootstrap-static -> game_config.scoring`` (so a rule change upstream
flows through automatically). The thresholds are not published in that block; they were
verified by reproducing ``total_points`` exactly for every 2026/27 match (see tests).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

POSITIONS = ("GKP", "DEF", "MID", "FWD")


def _per_pos(value: Any, default: Mapping[str, int]) -> dict[str, int]:
    if isinstance(value, Mapping):
        return {p: int(value.get(p, default[p])) for p in POSITIONS}
    if value is None:
        return dict(default)
    return {p: int(value) for p in POSITIONS}


@dataclass(frozen=True)
class ScoringRules:
    long_play: int = 2
    short_play: int = 1
    goals_scored: dict[str, int] = field(default_factory=lambda: {"GKP": 10, "DEF": 6, "MID": 5, "FWD": 4})
    assists: int = 3
    clean_sheets: dict[str, int] = field(default_factory=lambda: {"GKP": 4, "DEF": 4, "MID": 1, "FWD": 0})
    goals_conceded: dict[str, int] = field(default_factory=lambda: {"GKP": -1, "DEF": -1, "MID": 0, "FWD": 0})
    saves: int = 1
    penalties_saved: int = 5
    penalties_missed: int = -2
    yellow_cards: int = -1
    red_cards: int = -3
    own_goals: int = -2
    defensive_contribution: dict[str, int] = field(default_factory=lambda: {"GKP": 0, "DEF": 2, "MID": 2, "FWD": 2})
    # Thresholds (not in game_config).
    long_play_minutes: int = 60
    goals_conceded_per: int = 2
    saves_per: int = 3
    dc_threshold: dict[str, int] = field(default_factory=lambda: {"GKP": 10**6, "DEF": 10, "MID": 12, "FWD": 12})

    @classmethod
    def from_game_config(cls, scoring: Mapping[str, Any] | None) -> "ScoringRules":
        if not scoring:
            return cls()
        d = cls()
        return cls(
            long_play=int(scoring.get("long_play", d.long_play)),
            short_play=int(scoring.get("short_play", d.short_play)),
            goals_scored=_per_pos(scoring.get("goals_scored"), d.goals_scored),
            assists=int(scoring.get("assists", d.assists)),
            clean_sheets=_per_pos(scoring.get("clean_sheets"), d.clean_sheets),
            goals_conceded=_per_pos(scoring.get("goals_conceded"), d.goals_conceded),
            saves=int(scoring.get("saves", d.saves)),
            penalties_saved=int(scoring.get("penalties_saved", d.penalties_saved)),
            penalties_missed=int(scoring.get("penalties_missed", d.penalties_missed)),
            yellow_cards=int(scoring.get("yellow_cards", d.yellow_cards)),
            red_cards=int(scoring.get("red_cards", d.red_cards)),
            own_goals=int(scoring.get("own_goals", d.own_goals)),
            defensive_contribution=_per_pos(scoring.get("defensive_contribution"), {p: 0 for p in POSITIONS}),
        )

    def as_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)

    def points(self, pos: str, s: Mapping[str, Any]) -> int:
        """Realised FPL points for one match (excluding nothing: bonus included)."""
        minutes = int(s["minutes"])
        if minutes <= 0:
            return 0
        pts = self.long_play if minutes >= self.long_play_minutes else self.short_play
        pts += int(s["goals_scored"]) * self.goals_scored[pos] + int(s["assists"]) * self.assists
        if int(s["clean_sheets"]) and minutes >= self.long_play_minutes:
            pts += self.clean_sheets[pos]
        pts += (int(s["goals_conceded"]) // self.goals_conceded_per) * self.goals_conceded[pos]
        if pos == "GKP":
            pts += (int(s["saves"]) // self.saves_per) * self.saves
        dc = s.get("defensive_contribution")
        if dc is not None and int(dc) >= self.dc_threshold[pos]:
            pts += self.defensive_contribution[pos]
        pts += int(s["penalties_saved"]) * self.penalties_saved + int(s["penalties_missed"]) * self.penalties_missed
        pts += int(s["yellow_cards"]) * self.yellow_cards + int(s["red_cards"]) * self.red_cards
        pts += int(s["own_goals"]) * self.own_goals + int(s["bonus"])
        return pts

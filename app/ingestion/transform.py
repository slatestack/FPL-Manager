"""Pure mapping functions: validated FPL payloads -> database rows."""

from __future__ import annotations

import logging
from typing import Any

from app.ingestion import schemas

log = logging.getLogger(__name__)

POSITIONS = frozenset({"GKP", "DEF", "MID", "FWD"})


def position_map(element_types: list[schemas.ElementType]) -> dict[int, str]:
    """Map ``element_type`` id -> position code, keeping only the four playing positions.

    Mapping by ``singular_name_short`` rather than hard-coding 1..4 keeps us correct if
    FPL adds non-player element types again (e.g. the 2024/25 'AM' assistant manager).
    """
    return {et.id: et.singular_name_short for et in element_types if et.singular_name_short in POSITIONS}


def team_rows(bootstrap: schemas.BootstrapStatic) -> list[dict[str, Any]]:
    return [
        {
            "team_id": t.id,
            "code": t.code,
            "name": t.name,
            "short_name": t.short_name,
            "strength": t.strength,
            "strength_overall_home": t.strength_overall_home,
            "strength_overall_away": t.strength_overall_away,
            "strength_attack_home": t.strength_attack_home,
            "strength_attack_away": t.strength_attack_away,
            "strength_defence_home": t.strength_defence_home,
            "strength_defence_away": t.strength_defence_away,
        }
        for t in bootstrap.teams
    ]


def gameweek_rows(bootstrap: schemas.BootstrapStatic) -> list[dict[str, Any]]:
    return [
        {
            "gameweek_id": e.id,
            "name": e.name,
            "deadline_time": e.deadline_time,
            "is_previous": e.is_previous,
            "is_current": e.is_current,
            "is_next": e.is_next,
            "finished": e.finished,
            "data_checked": e.data_checked,
            "average_entry_score": e.average_entry_score,
            "highest_score": e.highest_score,
        }
        for e in bootstrap.events
    ]


def chip_rows(bootstrap: schemas.BootstrapStatic) -> list[dict[str, Any]]:
    return [
        {
            "chip_id": c.id,
            "name": c.name,
            "chip_type": c.chip_type,
            "start_event": c.start_event,
            "stop_event": c.stop_event,
            "number": c.number,
        }
        for c in bootstrap.chips
    ]


def player_rows(bootstrap: schemas.BootstrapStatic) -> list[dict[str, Any]]:
    positions = position_map(bootstrap.element_types)
    rows = []
    skipped = 0
    for el in bootstrap.elements:
        pos = positions.get(el.element_type)
        if pos is None:
            skipped += 1
            continue
        rows.append(
            {
                "player_id": el.id,
                "web_name": el.web_name[:64],
                "team_id": el.team,
                "position": pos,
                "now_cost": el.now_cost,
                "status": el.status,
                "chance_of_playing": el.chance_of_playing_next_round,
                "code": el.code,
                "first_name": el.first_name,
                "second_name": el.second_name,
                "news": el.news or None,
                "news_added": el.news_added,
                "chance_of_playing_this_round": el.chance_of_playing_this_round,
                "selected_by_percent": el.selected_by_percent,
                "form": el.form,
                "points_per_game": el.points_per_game,
                "ep_next": el.ep_next,
                "total_points": el.total_points,
                "minutes": el.minutes,
                "starts": el.starts,
                "expected_goals": el.expected_goals,
                "expected_assists": el.expected_assists,
                "expected_goals_conceded": el.expected_goals_conceded,
                "penalties_order": el.penalties_order,
                "corners_and_indirect_freekicks_order": el.corners_and_indirect_freekicks_order,
                "direct_freekicks_order": el.direct_freekicks_order,
                "cost_change_event": el.cost_change_event,
                "transfers_in_event": el.transfers_in_event,
                "transfers_out_event": el.transfers_out_event,
            }
        )
    if skipped:
        log.info("skipped %d non-player elements", skipped)
    return rows


def fixture_rows(fixtures: list[schemas.Fixture]) -> list[dict[str, Any]]:
    return [
        {
            "fixture_id": f.id,
            "code": f.code,
            "gameweek_id": f.event,
            "team_h": f.team_h,
            "team_a": f.team_a,
            "team_h_difficulty": f.team_h_difficulty,
            "team_a_difficulty": f.team_a_difficulty,
            "kickoff_time": f.kickoff_time,
            "started": f.started,
            "finished": f.finished,
            "finished_provisional": f.finished_provisional,
            "team_h_score": f.team_h_score,
            "team_a_score": f.team_a_score,
        }
        for f in fixtures
    ]


def history_rows(summary: schemas.ElementSummary) -> list[dict[str, Any]]:
    return [
        {
            "player_id": h.element,
            "fixture_id": h.fixture,
            "gameweek_id": h.round,
            "opponent_team": h.opponent_team,
            "was_home": h.was_home,
            "kickoff_time": h.kickoff_time,
            "minutes": h.minutes,
            "starts": h.starts,
            "total_points": h.total_points,
            "goals_scored": h.goals_scored,
            "assists": h.assists,
            "clean_sheets": h.clean_sheets,
            "goals_conceded": h.goals_conceded,
            "own_goals": h.own_goals,
            "penalties_saved": h.penalties_saved,
            "penalties_missed": h.penalties_missed,
            "yellow_cards": h.yellow_cards,
            "red_cards": h.red_cards,
            "saves": h.saves,
            "bonus": h.bonus,
            "bps": h.bps,
            "defensive_contribution": h.defensive_contribution,
            "expected_goals": h.expected_goals,
            "expected_assists": h.expected_assists,
            "expected_goal_involvements": h.expected_goal_involvements,
            "expected_goals_conceded": h.expected_goals_conceded,
            "value": h.value,
            "selected": h.selected,
        }
        for h in summary.history
    ]


def entry_row(entry: schemas.EntryProfile) -> dict[str, Any]:
    return {
        "entry_id": entry.id,
        "entry_name": entry.name,
        "player_first_name": entry.player_first_name,
        "player_last_name": entry.player_last_name,
        "current_event": entry.current_event,
        "summary_overall_points": entry.summary_overall_points,
        "summary_overall_rank": entry.summary_overall_rank,
        "summary_event_points": entry.summary_event_points,
        "last_deadline_bank": entry.last_deadline_bank,
        "last_deadline_value": entry.last_deadline_value,
        "last_deadline_total_transfers": entry.last_deadline_total_transfers,
    }


def season_start_year(bootstrap: schemas.BootstrapStatic) -> int:
    """Season start year (e.g. 2026 for 2026/27), taken from the GW1 deadline."""
    first = min(bootstrap.events, key=lambda e: e.id)
    d = first.deadline_time
    return d.year if d.month >= 7 else d.year - 1

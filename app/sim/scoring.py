"""Gameweek scoring for a simulated manager, following FPL's rules:

* starters who play no minutes are replaced in bench order (the bench goalkeeper only for the
  goalkeeper), as long as the formation stays valid (1 GK, >= 3 DEF, >= 2 MID, >= 1 FWD);
* the captain's points double (triple with Triple Captain); if the captain plays no minutes the
  vice-captain gets the armband;
* Bench Boost counts all 15 players and makes no substitutions;
* hits cost 4 points each.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

MIN_FORMATION = {"GKP": 1, "DEF": 3, "MID": 2, "FWD": 1}


@dataclass
class GameweekScore:
    points: int
    lineup_points: int
    hit_cost: int
    autosubs: list[tuple[int, int]]
    captain_used: int | None


def score_gameweek(
    starting_xi: Sequence[int], bench: Sequence[int], captain: int, vice_captain: int,
    points: Mapping[int, int], minutes: Mapping[int, int], position: Mapping[int, str],
    *, chip: str | None = None, hits: int = 0,
) -> GameweekScore:
    played = lambda p: minutes.get(p, 0) > 0  # noqa: E731
    xi = list(starting_xi)
    subs: list[tuple[int, int]] = []
    if chip == "bboost":
        counted = xi + list(bench)
    else:
        used: set[int] = set()
        for out in [p for p in xi if not played(p)]:
            for cand in bench:
                if cand in used or not played(cand):
                    continue
                if (position[out] == "GKP") != (position[cand] == "GKP"):
                    continue
                trial = [cand if p == out else p for p in xi]
                counts = Counter(position[p] for p in trial)
                if all(counts[k] >= v for k, v in MIN_FORMATION.items()):
                    xi = trial
                    used.add(cand)
                    subs.append((out, cand))
                    break
        counted = xi
    total = sum(points.get(p, 0) for p in counted)
    armband = captain if played(captain) else (vice_captain if played(vice_captain) else None)
    if armband is not None:
        total += points.get(armband, 0) * (2 if chip == "3xc" else 1)
    return GameweekScore(points=total - 4 * hits, lineup_points=total, hit_cost=4 * hits, autosubs=subs, captain_used=armband)

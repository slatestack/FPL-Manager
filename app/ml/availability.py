"""P(available) per player per upcoming gameweek from FPL status, chance-of-playing and news.

* ``chance_of_playing_next_round`` is FPL's own estimate for the next gameweek.
* A return date in the news ("Expected back 01 Jan", "Suspended until 29 Nov") makes the
  player available from the first gameweek whose deadline is on/after that date.
* Otherwise doubts recover geometrically over the horizon (injuries more slowly).
* 'u' (left the club) and 'n' (on loan / not in squad) are unavailable throughout.
"""

from __future__ import annotations

import re
from datetime import date, datetime

import pandas as pd

_DATE = re.compile(r"(?:expected back|until|return(?:s|ing)? (?:on|by)?)\s+(\d{1,2})\s+([A-Za-z]{3})", re.I)
_MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
RECOVERY = {"d": 0.5, "i": 0.8, "s": 0.5}


def return_date(news: str | None, reference: datetime) -> date | None:
    if not news:
        return None
    m = _DATE.search(news)
    if not m:
        return None
    month = _MONTHS.get(m.group(2).lower())
    if month is None:
        return None
    day = int(m.group(1))
    year = reference.year
    try:
        d = date(year, month, day)
    except ValueError:
        return None
    if (reference.date() - d).days > 60:  # "back 05 Jan" written in December
        d = date(year + 1, month, day)
    return d


def availability(players: pd.DataFrame, deadlines: dict[int, datetime], reference: datetime) -> pd.DataFrame:
    """Long frame (player_id, gameweek, p_avail) for the gameweeks in ``deadlines`` (in order)."""
    gws = sorted(deadlines)
    rows = []
    for p in players.itertuples():
        status = p.status
        cop = None if pd.isna(p.chance_of_playing) else float(p.chance_of_playing) / 100.0
        if status in ("u", "n"):
            p1 = 0.0
        elif cop is not None:
            p1 = cop
        else:
            p1 = {"a": 1.0, "d": 0.5}.get(status, 0.0)
        back = return_date(p.news, reference) if status in ("i", "s", "d") else None
        first_back = True
        for t, gw in enumerate(gws):
            if status in ("u", "n"):
                pa = 0.0
            elif back is not None:
                if deadlines[gw].date() >= back:
                    pa = 0.9 if first_back else 1.0
                    first_back = False
                else:
                    pa = p1 if t == 0 else min(p1, 0.25)
            elif p1 < 1.0 and status in RECOVERY:
                pa = 1.0 - (1.0 - p1) * RECOVERY[status] ** t
            else:
                pa = p1
            rows.append((p.player_id, gw, pa))
    return pd.DataFrame(rows, columns=["player_id", "gameweek", "p_avail"])

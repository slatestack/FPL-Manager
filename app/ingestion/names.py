"""Resolve external team / player names (Understat, bookmakers) to FPL ids."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from difflib import SequenceMatcher


def normalize(name: str) -> str:
    """Lower-case, strip accents and punctuation, collapse whitespace."""
    text = unicodedata.normalize("NFKD", name)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("&", " and ").replace("'", "").replace("’", "").lower()
    text = text.replace("ø", "o").replace("ß", "ss").replace("æ", "ae").replace("ł", "l").replace("đ", "d")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# Long / bookmaker / Understat spellings -> FPL `name` (normalised on lookup).
TEAM_ALIASES: dict[str, str] = {
    "manchester united": "Man Utd",
    "man united": "Man Utd",
    "manchester utd": "Man Utd",
    "manchester city": "Man City",
    "tottenham hotspur": "Spurs",
    "tottenham": "Spurs",
    "wolverhampton wanderers": "Wolves",
    "wolverhampton": "Wolves",
    "nottingham forest": "Nott'm Forest",
    "nottm forest": "Nott'm Forest",
    "newcastle united": "Newcastle",
    "brighton and hove albion": "Brighton",
    "brighton hove albion": "Brighton",
    "west ham united": "West Ham",
    "leicester city": "Leicester",
    "ipswich town": "Ipswich",
    "leeds united": "Leeds",
    "sheffield united": "Sheffield Utd",
    "luton town": "Luton",
    "afc bournemouth": "Bournemouth",
    "west bromwich albion": "West Brom",
    "norwich city": "Norwich",
    "cardiff city": "Cardiff",
    "swansea city": "Swansea",
    "stoke city": "Stoke",
    "hull city": "Hull",
    "coventry city": "Coventry",
    "queens park rangers": "QPR",
    "huddersfield town": "Huddersfield",
}

_SUFFIXES = {"fc", "afc", "united", "city", "town", "hotspur", "wanderers", "albion", "and", "hove"}


def _strip_suffixes(norm: str) -> str:
    return " ".join(t for t in norm.split() if t not in _SUFFIXES)


class TeamResolver:
    """Maps an external club name to an FPL ``team_id``."""

    def __init__(self, teams: Iterable[Mapping[str, object]], *, fuzzy_threshold: float = 0.8):
        self._by_key: dict[str, int] = {}
        self._names: list[tuple[str, int]] = []
        self.fuzzy_threshold = fuzzy_threshold
        for t in teams:
            tid = int(t["team_id"])  # type: ignore[arg-type]
            for raw in (t["name"], t["short_name"]):
                key = normalize(str(raw))
                self._by_key.setdefault(key, tid)
                stripped = _strip_suffixes(key)
                if stripped:
                    self._by_key.setdefault(stripped, tid)
                self._names.append((key, tid))

    def resolve(self, external: str) -> int | None:
        key = normalize(external)
        alias = TEAM_ALIASES.get(key)
        if alias is not None and normalize(alias) in self._by_key:
            return self._by_key[normalize(alias)]
        if key in self._by_key:
            return self._by_key[key]
        stripped = _strip_suffixes(key)
        if stripped in self._by_key:
            return self._by_key[stripped]
        best_score, best_id = 0.0, None
        for name, tid in self._names:
            score = SequenceMatcher(None, key, name).ratio()
            if score > best_score:
                best_score, best_id = score, tid
        return best_id if best_score >= self.fuzzy_threshold else None


@dataclass(frozen=True)
class PlayerMatch:
    player_id: int
    confidence: float
    matched_by: str  # 'exact' | 'fuzzy' | 'manual'


@dataclass(frozen=True)
class _Candidate:
    player_id: int
    team_id: int
    full: str
    web: str
    tokens: frozenset[str]


class PlayerResolver:
    """Maps an external player name (+ optional club) to an FPL ``player_id``."""

    def __init__(
        self,
        players: Iterable[Mapping[str, object]],
        *,
        fuzzy_threshold: float = 0.78,
        teamless_threshold: float = 0.92,
    ):
        self.fuzzy_threshold = fuzzy_threshold
        self.teamless_threshold = teamless_threshold
        self._by_team: dict[int, list[_Candidate]] = {}
        self._all: list[_Candidate] = []
        for p in players:
            full = normalize(f"{p.get('first_name') or ''} {p.get('second_name') or ''}")
            web = normalize(str(p["web_name"]))
            cand = _Candidate(
                player_id=int(p["player_id"]),  # type: ignore[arg-type]
                team_id=int(p["team_id"]),  # type: ignore[arg-type]
                full=full,
                web=web,
                tokens=frozenset(full.split()) | frozenset(web.split()),
            )
            self._by_team.setdefault(cand.team_id, []).append(cand)
            self._all.append(cand)

    @staticmethod
    def _score(name: str, cand: _Candidate) -> tuple[float, bool]:
        """Return (score, exact)."""
        if name == cand.full or name == cand.web:
            return 1.0, True
        tokens = set(name.split())
        # "Gabriel Martinelli" vs FPL "Gabriel Martinelli Silva" / web "Martinelli"
        if tokens and (tokens <= cand.tokens or set(cand.web.split()) <= tokens and len(cand.web) > 3):
            overlap = len(tokens & cand.tokens) / max(len(tokens), 1)
            return 0.85 + 0.1 * overlap, False
        return max(SequenceMatcher(None, name, cand.full).ratio(), SequenceMatcher(None, name, cand.web).ratio()), False

    def resolve(self, external_name: str, team_ids: Iterable[int] = ()) -> PlayerMatch | None:
        name = normalize(external_name)
        if not name:
            return None
        team_ids = [t for t in team_ids if t is not None]
        pool = [c for t in team_ids for c in self._by_team.get(t, [])] if team_ids else self._all
        threshold = self.fuzzy_threshold if team_ids else self.teamless_threshold
        scored = sorted(((self._score(name, c), c) for c in pool), key=lambda x: x[0][0], reverse=True)
        if not scored:
            return None
        (best, exact), cand = scored[0]
        if best < threshold:
            return None
        # Refuse ambiguous matches (two candidates within a hair of each other).
        if len(scored) > 1 and scored[1][0][0] >= best - 0.02 and scored[1][1].player_id != cand.player_id:
            return None
        return PlayerMatch(cand.player_id, round(best, 3), "exact" if exact else "fuzzy")

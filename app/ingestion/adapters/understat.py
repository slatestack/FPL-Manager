"""Understat season-level xG/xA adapter for the Premier League.

Understat serves league data as JSON from ``/getLeagueData/EPL/{season}`` (XHR) and,
historically, embedded as ``var playersData = JSON.parse('...')`` in the league page.
The adapter tries the JSON endpoint first and falls back to the HTML page so either
site layout works.
"""

from __future__ import annotations

import codecs
import json
import logging
import re
from typing import Any

from app.ingestion.adapters.base import XGRecord
from app.ingestion.http import ResilientClient, UpstreamError

log = logging.getLogger(__name__)

_EMBEDDED_PLAYERS = re.compile(r"playersData\s*=\s*JSON\.parse\('(?P<data>.*?)'\)", re.S)


def decode_embedded_json(escaped: str) -> Any:
    """Decode Understat's hex-escaped (``\\x22``) JSON string literal."""
    text = codecs.decode(escaped, "unicode_escape")
    try:
        # The escapes encode UTF-8 bytes, so re-interpret them as such.
        text = text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    return json.loads(text)


def parse_league_html(html: str) -> list[dict[str, Any]]:
    match = _EMBEDDED_PLAYERS.search(html)
    if not match:
        if "come back soon" in html.lower():
            # Understat answers every URL with HTTP 200 and this page while in maintenance.
            raise UpstreamError("understat is down for maintenance ('We will come back soon' page)")
        raise UpstreamError("understat league page has no embedded playersData")
    return decode_embedded_json(match.group("data"))


def _f(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    return float(value) if value not in (None, "") else default


def _i(row: dict[str, Any], key: str) -> int:
    value = row.get(key)
    return int(float(value)) if value not in (None, "") else 0


def to_records(rows: list[dict[str, Any]]) -> list[XGRecord]:
    out = []
    for r in rows:
        out.append(
            XGRecord(
                external_id=str(r["id"]),
                name=str(r["player_name"]),
                team_names=tuple(t.strip() for t in str(r.get("team_title", "")).split(",") if t.strip()),
                games=_i(r, "games"),
                minutes=_i(r, "time"),
                goals=_i(r, "goals"),
                npg=_i(r, "npg"),
                assists=_i(r, "assists"),
                shots=_i(r, "shots"),
                key_passes=_i(r, "key_passes"),
                xg=_f(r, "xG"),
                npxg=_f(r, "npxG"),
                xa=_f(r, "xA"),
                xg_chain=_f(r, "xGChain") if "xGChain" in r else None,
                xg_buildup=_f(r, "xGBuildup") if "xGBuildup" in r else None,
            )
        )
    return out


class UnderstatAdapter:
    source = "understat"

    def __init__(self, http: ResilientClient, *, base_url: str = "https://understat.com", league: str = "EPL"):
        self._http = http
        self._base = base_url.rstrip("/")
        self._league = league

    async def fetch(self, season: int) -> list[XGRecord]:
        try:
            data = await self._http.get_json(
                f"{self._base}/getLeagueData/{self._league}/{season}",
                headers={"X-Requested-With": "XMLHttpRequest", "Referer": f"{self._base}/league/{self._league}/{season}"},
            )
            rows = data["players"] if isinstance(data, dict) else data
        except (UpstreamError, KeyError, TypeError) as exc:
            log.info("understat JSON endpoint unavailable (%s); falling back to league page", exc)
            rows = parse_league_html(await self._http.get_text(f"{self._base}/league/{self._league}/{season}"))
        return to_records(rows)

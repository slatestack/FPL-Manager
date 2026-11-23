"""The Odds API (v4) adapter: consensus 1X2 + Over/Under 2.5, optional anytime goalscorer.

Docs: https://the-odds-api.com/liveapi/guides/v4/  Each bulk call costs quota per market
and region; per-event player-prop calls cost extra, so they are opt-in.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from app.ingestion import odds_math
from app.ingestion.adapters.base import FixtureOddsRecord, OddsSnapshot, PlayerGoalOddsRecord
from app.ingestion.http import ResilientClient, UpstreamError

log = logging.getLogger(__name__)

GOALSCORER_MARKET = "player_goal_scorer_anytime"


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def parse_match_odds(event: dict[str, Any]) -> FixtureOddsRecord | None:
    home, away = event["home_team"], event["away_team"]
    h2h: list[list[float]] = []
    over: list[float] = []
    for book in event.get("bookmakers", []):
        for market in book.get("markets", []):
            outcomes = {o["name"]: o for o in market.get("outcomes", [])}
            try:
                if market["key"] == "h2h" and {home, away, "Draw"} <= outcomes.keys():
                    h2h.append(
                        odds_math.implied_probabilities(
                            [outcomes[home]["price"], outcomes["Draw"]["price"], outcomes[away]["price"]]
                        )
                    )
                elif market["key"] == "totals" and {"Over", "Under"} <= outcomes.keys():
                    if float(outcomes["Over"].get("point", 0)) == 2.5 and float(outcomes["Under"].get("point", 0)) == 2.5:
                        over.append(
                            odds_math.implied_probabilities([outcomes["Over"]["price"], outcomes["Under"]["price"]])[0]
                        )
            except ValueError:
                log.warning("ignoring malformed %s market from %s", market.get("key"), book.get("key"))
    if not h2h:
        return None
    p_home, p_draw, p_away = odds_math.consensus(h2h)
    return FixtureOddsRecord(
        home_team=home,
        away_team=away,
        commence_time=_parse_time(event["commence_time"]),
        bookmaker_count=len(h2h),
        p_home=p_home,
        p_draw=p_draw,
        p_away=p_away,
        p_over_2_5=sum(over) / len(over) if over else None,
        event_id=event.get("id"),
    )


def parse_goalscorer_odds(event: dict[str, Any]) -> list[PlayerGoalOddsRecord]:
    prices: dict[str, list[float]] = defaultdict(list)
    for book in event.get("bookmakers", []):
        for market in book.get("markets", []):
            if market.get("key") != GOALSCORER_MARKET:
                continue
            for o in market.get("outcomes", []):
                # Player props put the player in `description` ("Yes"/"No" in `name`).
                if o.get("name") == "No":
                    continue
                player = o.get("description") or o.get("name")
                if player and float(o.get("price", 0)) > 1.0:
                    prices[player].append(float(o["price"]))
    commence = _parse_time(event["commence_time"])
    return [
        PlayerGoalOddsRecord(
            player_name=name,
            home_team=event["home_team"],
            away_team=event["away_team"],
            commence_time=commence,
            bookmaker_count=len(p),
            median_decimal_odds=odds_math.median_price(p),
        )
        for name, p in sorted(prices.items())
    ]


class OddsApiAdapter:
    """Team odds in one bulk call (2 credits); goalscorer odds per match (1 credit each), only for
    matches kicking off before ``goalscorers_until`` (the coming gameweek) and only while the
    account keeps at least ``reserve_credits`` in hand."""

    source = "the-odds-api"

    def __init__(
        self,
        http: ResilientClient,
        *,
        api_key: str,
        base_url: str = "https://api.the-odds-api.com/v4",
        sport_key: str = "soccer_epl",
        regions: str = "uk",
        include_goalscorers: bool = False,
        reserve_credits: int = 40,
    ):
        self._http = http
        self._key = api_key
        self._base = base_url.rstrip("/")
        self._sport = sport_key
        self._regions = regions
        self._goalscorers = include_goalscorers
        self.reserve_credits = reserve_credits
        self.credits_remaining: int | None = None

    async def _get(self, url: str, params: dict[str, Any]) -> Any:
        resp = await self._http.request("GET", url, params={"apiKey": self._key, **params})
        remaining = resp.headers.get("x-requests-remaining")
        if remaining is not None:
            try:
                self.credits_remaining = int(float(remaining))
            except ValueError:
                pass
        return resp.json()

    async def fetch(self, *, goalscorers_until: datetime | None = None) -> OddsSnapshot:
        captured = datetime.now(timezone.utc)
        events = await self._get(f"{self._base}/sports/{self._sport}/odds",
                                 {"regions": self._regions, "markets": "h2h,totals", "oddsFormat": "decimal"})
        snapshot = OddsSnapshot(source=self.source, captured_at=captured)
        for event in events:
            rec = parse_match_odds(event)
            if rec is not None:
                snapshot.fixtures.append(rec)
        if self._goalscorers and goalscorers_until is not None:
            soon = [e for e in events if _parse_time(e["commence_time"]) <= goalscorers_until]
            for event in soon:
                if self.credits_remaining is not None and self.credits_remaining <= self.reserve_credits:
                    log.warning("odds credits low (%s left); skipping goalscorer odds", self.credits_remaining)
                    break
                try:
                    detail = await self._get(f"{self._base}/sports/{self._sport}/events/{event['id']}/odds",
                                             {"regions": self._regions, "markets": GOALSCORER_MARKET, "oddsFormat": "decimal"})
                except UpstreamError as exc:
                    log.warning("goalscorer odds unavailable for %s: %s", event.get("id"), exc)
                    continue
                snapshot.goalscorers.extend(parse_goalscorer_odds(detail))
        snapshot.credits_remaining = self.credits_remaining
        return snapshot

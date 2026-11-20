"""Client for the official, unauthenticated FPL read endpoints (Section 4.1)."""

from __future__ import annotations

from typing import Any

import httpx

from app.config import Settings
from app.ingestion.http import ResilientClient


class FPLClient:
    def __init__(self, settings: Settings, *, transport: httpx.AsyncBaseTransport | None = None, **client_kwargs: Any):
        self._http = ResilientClient(
            base_url=settings.api_base_url.rstrip("/") + "/",
            timeout=settings.http_timeout_seconds,
            max_retries=settings.http_max_retries,
            backoff_base=settings.http_backoff_base_seconds,
            backoff_max=settings.http_backoff_max_seconds,
            headers={"User-Agent": settings.user_agent, "Accept": "application/json"},
            transport=transport,
            **client_kwargs,
        )

    async def __aenter__(self) -> "FPLClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._http.aclose()

    async def bootstrap_static(self) -> dict[str, Any]:
        return await self._http.get_json("bootstrap-static/")

    async def fixtures(self) -> list[dict[str, Any]]:
        return await self._http.get_json("fixtures/")

    async def element_summary(self, player_id: int) -> dict[str, Any]:
        return await self._http.get_json(f"element-summary/{int(player_id)}/")

    async def entry(self, team_id: int) -> dict[str, Any]:
        return await self._http.get_json(f"entry/{int(team_id)}/")

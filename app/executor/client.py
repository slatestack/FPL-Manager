"""Authenticated FPL client: ``my-team`` reads and the two write endpoints.

Writes are never retried automatically. If a POST fails ambiguously, the executor re-reads
``my-team`` to find out whether it was applied (idempotency by state, not by blind retry).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from http.cookies import SimpleCookie
from typing import Any

import httpx

from app.config import Settings
from app.executor.vault import AuthError, AuthTokens
from app.ingestion.http import ResilientClient, UpstreamError


@dataclass
class MyTeamPick:
    element: int
    position: int
    selling_price: int
    purchase_price: int
    is_captain: bool
    is_vice_captain: bool


@dataclass
class MyTeam:
    picks: list[MyTeamPick]
    chips: list[dict[str, Any]]
    bank: int
    free_transfers: int | None         # None while transfers are unlimited (Wildcard / Free Hit active)
    transfers_made: int
    transfer_status: str | None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def squad(self) -> set[int]:
        return {p.element for p in self.picks}

    @property
    def starting_xi(self) -> set[int]:
        return {p.element for p in self.picks if p.position <= 11}

    @property
    def bench_order(self) -> list[int]:
        return [p.element for p in sorted(self.picks, key=lambda p: p.position) if p.position > 11]

    @property
    def captain(self) -> int | None:
        return next((p.element for p in self.picks if p.is_captain), None)

    @property
    def vice_captain(self) -> int | None:
        return next((p.element for p in self.picks if p.is_vice_captain), None)

    def chip_status(self, name: str) -> str | None:
        statuses = [c.get("status_for_entry") for c in self.chips if c.get("name") == name]
        for preferred in ("active", "available"):
            if preferred in statuses:
                return preferred
        return statuses[0] if statuses else None

    @property
    def active_chip(self) -> str | None:
        return next((c["name"] for c in self.chips if c.get("status_for_entry") == "active" or c.get("is_pending")), None)


def parse_my_team(d: dict[str, Any]) -> MyTeam:
    t = d.get("transfers") or {}
    limit, made = t.get("limit"), int(t.get("made") or 0)
    status = t.get("status")
    ft = None if status == "unlimited" or limit is None else max(0, int(limit) - made)
    return MyTeam(
        picks=[
            MyTeamPick(int(p["element"]), int(p["position"]), int(p.get("selling_price", 0)), int(p.get("purchase_price", 0)),
                       bool(p.get("is_captain")), bool(p.get("is_vice_captain")))
            for p in d.get("picks", [])
        ],
        chips=list(d.get("chips", [])),
        bank=int(t.get("bank", 0)),
        free_transfers=ft,
        transfers_made=made,
        transfer_status=status,
        raw=d,
    )


class AuthenticatedFPLClient:
    def __init__(self, settings: Settings, tokens: AuthTokens, *, transport: httpx.AsyncBaseTransport | None = None, **kw: Any):
        headers = {
            "User-Agent": settings.user_agent,
            "Accept": "application/json",
            "X-API-Authorization": f"Bearer {tokens.access_token}",
        }
        if tokens.cookie:
            headers["Cookie"] = tokens.cookie
            csrf = SimpleCookie(tokens.cookie).get("csrftoken")
            if csrf is not None:
                headers["X-CSRFToken"] = csrf.value
        self._read = ResilientClient(
            base_url=settings.api_base_url.rstrip("/") + "/", timeout=settings.http_timeout_seconds,
            max_retries=settings.http_max_retries, headers=headers, transport=transport, **kw,
        )
        self._write = httpx.AsyncClient(
            base_url=settings.api_base_url.rstrip("/") + "/", timeout=settings.http_timeout_seconds,
            headers={**headers, "Content-Type": "application/json"}, transport=transport,
        )

    async def __aenter__(self) -> "AuthenticatedFPLClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self._read.aclose()
        await self._write.aclose()

    async def my_team(self, entry: int) -> MyTeam:
        try:
            return parse_my_team(await self._read.get_json(f"my-team/{int(entry)}/"))
        except UpstreamError as exc:
            if exc.status_code in (401, 403):
                raise AuthError(f"FPL rejected the credentials (HTTP {exc.status_code})") from exc
            raise

    async def _post(self, path: str, payload: dict[str, Any]) -> httpx.Response:
        resp = await self._write.post(path, json=payload)
        if resp.status_code in (401, 403):
            raise AuthError(f"FPL rejected the credentials on {path} (HTTP {resp.status_code}): {resp.text[:200]}")
        return resp

    async def post_transfers(self, payload: dict[str, Any]) -> httpx.Response:
        return await self._post("transfers/", payload)

    async def post_my_team(self, entry: int, payload: dict[str, Any]) -> httpx.Response:
        return await self._post(f"my-team/{int(entry)}/", payload)

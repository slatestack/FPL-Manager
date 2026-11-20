"""Async HTTP client with exponential backoff, jitter and ``Retry-After`` support.

Retries on transport errors, timeouts, HTTP 429 and 5xx (FPL answers 503 while
"the game is being updated" around deadlines). Other 4xx are raised immediately.
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone
from typing import Any

import httpx

log = logging.getLogger(__name__)

RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


class UpstreamError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None, url: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.url = url


def _retry_after_seconds(response: httpx.Response) -> float | None:
    value = response.headers.get("retry-after")
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())


def _redact(url: httpx.URL) -> str:
    # Never surface query strings: some upstreams (The Odds API) take the API key there.
    return str(url.copy_with(query=None))


class ResilientClient:
    def __init__(
        self,
        *,
        base_url: str = "",
        timeout: float = 20.0,
        max_retries: int = 5,
        backoff_base: float = 1.0,
        backoff_max: float = 60.0,
        headers: dict[str, str] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=timeout,
            headers=headers or {},
            follow_redirects=True,
            transport=transport,
        )
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self._sleep = sleep

    async def __aenter__(self) -> "ResilientClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    def _backoff(self, attempt: int) -> float:
        # Full jitter: uniform(0, min(max, base * 2**attempt)), floored at base/2.
        cap = min(self.backoff_max, self.backoff_base * (2**attempt))
        return max(self.backoff_base / 2, random.uniform(0, cap))

    async def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        attempt = 0
        while True:
            try:
                response = await self._client.request(method, url, **kwargs)
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                if attempt >= self.max_retries:
                    raise UpstreamError(f"{method} {url} failed after {attempt + 1} attempts: {exc!r}", url=url) from exc
                delay = self._backoff(attempt)
                log.warning("%s %s transport error %r; retry %d in %.1fs", method, url, exc, attempt + 1, delay)
            else:
                if response.status_code < 400:
                    return response
                if response.status_code not in RETRYABLE_STATUS or attempt >= self.max_retries:
                    raise UpstreamError(
                        f"{method} {url} -> HTTP {response.status_code}",
                        status_code=response.status_code,
                        url=_redact(response.url),
                    )
                retry_after = _retry_after_seconds(response)
                delay = min(self.backoff_max, retry_after) if retry_after is not None else self._backoff(attempt)
                log.warning("%s %s -> HTTP %d; retry %d in %.1fs", method, url, response.status_code, attempt + 1, delay)
            attempt += 1
            await self._sleep(delay)

    async def get_json(self, url: str, **kwargs: Any) -> Any:
        response = await self.request("GET", url, **kwargs)
        try:
            return response.json()
        except ValueError as exc:
            raise UpstreamError(f"GET {url} returned non-JSON body", status_code=response.status_code, url=url) from exc

    async def get_text(self, url: str, **kwargs: Any) -> str:
        return (await self.request("GET", url, **kwargs)).text

"""Build clients/adapters from settings, keeping wiring out of the jobs themselves."""

from __future__ import annotations

from contextlib import AsyncExitStack

from app.config import Settings
from app.ingestion.adapters.archive import FPLArchiveAdapter
from app.ingestion.adapters.odds_api import OddsApiAdapter
from app.ingestion.adapters.understat import UnderstatAdapter
from app.ingestion.fpl_client import FPLClient
from app.ingestion.http import ResilientClient


def _http(settings: Settings) -> ResilientClient:
    return ResilientClient(
        timeout=settings.http_timeout_seconds,
        max_retries=settings.http_max_retries,
        backoff_base=settings.http_backoff_base_seconds,
        backoff_max=settings.http_backoff_max_seconds,
        headers={"User-Agent": settings.user_agent},
    )


async def build_sources(
    settings: Settings, stack: AsyncExitStack
) -> tuple[FPLClient, UnderstatAdapter | None, OddsApiAdapter | None]:
    fpl = await stack.enter_async_context(FPLClient(settings))
    xg = None
    if settings.understat_enabled:
        xg = UnderstatAdapter(await stack.enter_async_context(_http(settings)), base_url=settings.understat_base_url)
    odds = None
    if settings.odds_enabled:
        odds = OddsApiAdapter(
            await stack.enter_async_context(_http(settings)),
            api_key=settings.odds_api_key or "",
            base_url=settings.odds_api_base_url,
            sport_key=settings.odds_sport_key,
            regions=settings.odds_regions,
            include_goalscorers=settings.odds_include_goalscorers,
            reserve_credits=settings.odds_reserve_credits,
        )
    return fpl, xg, odds


async def build_archive(settings: Settings, stack: AsyncExitStack) -> FPLArchiveAdapter:
    http = await stack.enter_async_context(_http(settings))
    return FPLArchiveAdapter(http, base_url=settings.archive_base_url)

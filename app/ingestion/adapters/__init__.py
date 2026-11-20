"""Plug-in enrichment adapters (Section 4.1.3)."""

from app.ingestion.adapters.base import (
    FixtureOddsRecord,
    OddsAdapter,
    OddsSnapshot,
    PlayerGoalOddsRecord,
    XGAdapter,
    XGRecord,
)

__all__ = [
    "FixtureOddsRecord",
    "OddsAdapter",
    "OddsSnapshot",
    "PlayerGoalOddsRecord",
    "XGAdapter",
    "XGRecord",
]

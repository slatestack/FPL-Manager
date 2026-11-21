"""Runtime configuration, read from environment variables prefixed with ``FPL_``."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FPL_", env_file=".env", extra="ignore")

    database_url: str = "postgresql://fpl:fpl@localhost:5432/fpl"
    redis_url: str = "redis://localhost:6379/0"

    # Official FPL API
    api_base_url: str = "https://fantasy.premierleague.com/api"
    team_id: int | None = None

    # HTTP resilience
    http_timeout_seconds: float = 20.0
    http_max_retries: int = 5
    http_backoff_base_seconds: float = 1.0
    http_backoff_max_seconds: float = 60.0
    user_agent: str = "fpl-manager/0.1 (+https://github.com/slatestack/fpl-manager)"

    # element-summary fan-out
    history_concurrency: int = Field(default=6, ge=1, le=32)

    # Enrichment adapters
    understat_enabled: bool = True
    understat_base_url: str = "https://understat.com"
    odds_api_key: str | None = None
    odds_api_base_url: str = "https://api.the-odds-api.com/v4"
    odds_regions: str = "uk"
    odds_sport_key: str = "soccer_epl"
    odds_include_goalscorers: bool = False  # per-event calls; consumes extra API quota
    odds_min_interval_minutes: int = 30     # quota protection

    # Past seasons used as model training / backtest data
    archive_base_url: str = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
    archive_seasons: str = "2024-25,2025-26"

    # Raw upstream payload retention (bootstrap-static is ~1.5 MB and changes on most polls)
    raw_snapshot_retention_days: int = 14

    # Only fetch element-summary for players who have played this season
    history_only_with_minutes: bool = True

    # Ingestion lock (Redis) - must outlive the longest ingestion job
    ingest_lock_timeout_seconds: int = 900

    @property
    def archive_season_list(self) -> list[str]:
        return [s.strip() for s in self.archive_seasons.split(",") if s.strip()]

    @property
    def odds_enabled(self) -> bool:
        return bool(self.odds_api_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()

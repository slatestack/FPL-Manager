"""Runtime configuration, read from environment variables prefixed with ``FPL_``."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # str_strip_whitespace: a .env edited on Windows can leave "\r" at the end of every value.
    model_config = SettingsConfigDict(env_prefix="FPL_", env_file=".env", extra="ignore", str_strip_whitespace=True)

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
    odds_include_goalscorers: bool = True   # coming gameweek only, at the T-24h and T-30m re-plans
    odds_reserve_credits: int = 40          # never spend the last credits of the month on scorer odds

    # Past seasons used as model training / backtest data
    archive_base_url: str = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
    archive_seasons: str = "2024-25,2025-26"

    # Raw upstream payload retention (bootstrap-static is ~1.5 MB and changes on most polls)
    raw_snapshot_retention_days: int = 14

    # Only fetch element-summary for players who have played this season
    history_only_with_minutes: bool = True

    # Phase 5: authenticated execution
    secret_key: str | None = None            # Fernet key encrypting the stored FPL tokens
    oidc_token_url: str = "https://account.premierleague.com/as/token"
    oidc_client_id: str = "bfcbaf69-aade-4c1b-8f00-c1cb8a193030"   # FPL web app's public client
    oidc_authorize_url: str = "https://account.premierleague.com/as/authorize"
    oidc_redirect_uri: str = "https://fantasy.premierleague.com/"
    oidc_scope: str = "openid profile email offline_access p1:update:user consent interest"
    login_reminder_days: int = 14            # warn this long before the 180-day FPL session ends
    commit_minutes_before: int = 15          # T-15m autonomous commit
    final_lock_minutes_before: int = 30      # T-30m final data lock + re-optimise
    dry_run_hours_before: int = 24           # T-24h dry run + summary
    commit_cutoff_minutes: int = 3           # never start a commit closer than this to the deadline
    plan_max_age_minutes: int = 45           # a commit needs a plan at least this fresh
    keepalive_hours: int = 4

    # Phase 6: control plane + notifications
    api_token: str | None = None             # required for mutating control-plane endpoints
    telegram_bot_token: str | None = None
    telegram_chat_id: str | None = None
    alert_webhook_url: str | None = None
    whatsapp_phone: str | None = None        # your WhatsApp number with country code, e.g. +19195551234
    whatsapp_apikey: str | None = None       # from CallMeBot (message their bot once to get it)
    whatsapp_api_url: str = "https://api.callmebot.com/whatsapp.php"

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

"""Token vault (Section 7.1).

FPL signs in through an OIDC authorization-code + PKCE flow at account.premierleague.com with
anti-bot protection, so the system never handles your password. Instead you copy the token
pair your logged-in browser already holds (the ``oidc.user:...`` entry in the site's
localStorage) once. The pair is encrypted with ``FPL_SECRET_KEY`` (Fernet) in
``system_config.encrypted_auth_cookie``. The access token is renewed with the standard
``refresh_token`` grant against the public client the FPL web app uses.
"""

from __future__ import annotations

import base64
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import asyncpg
import httpx
from cryptography.fernet import Fernet, InvalidToken

from app.config import Settings

log = logging.getLogger(__name__)
RENEW_MARGIN = timedelta(minutes=5)


class AuthError(RuntimeError):
    """Credentials missing, expired beyond refresh, or rejected by FPL."""


@dataclass
class AuthTokens:
    access_token: str
    refresh_token: str | None
    expires_at: datetime | None
    cookie: str | None = None          # optional raw Cookie header (legacy session cookies)

    @property
    def session_expires_at(self) -> datetime | None:
        """When re-login becomes unavoidable: the refresh token's hard expiry (FPL sessions last 180 days)."""
        return _jwt_exp(self.refresh_token) if self.refresh_token else self.expires_at

    def expired(self, now: datetime | None = None, margin: timedelta = RENEW_MARGIN) -> bool:
        if self.expires_at is None:
            return False
        return (now or datetime.now(timezone.utc)) + margin >= self.expires_at

    def to_json(self) -> str:
        d = asdict(self)
        d["expires_at"] = self.expires_at.isoformat() if self.expires_at else None
        return json.dumps(d)

    @classmethod
    def from_json(cls, raw: str) -> "AuthTokens":
        d = json.loads(raw)
        exp = d.get("expires_at")
        return cls(d["access_token"], d.get("refresh_token"), datetime.fromisoformat(exp) if exp else None, d.get("cookie"))


def _jwt_exp(token: str) -> datetime | None:
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        exp = json.loads(base64.urlsafe_b64decode(payload)).get("exp")
        return datetime.fromtimestamp(int(exp), tz=timezone.utc) if exp else None
    except (IndexError, ValueError, TypeError):
        return None


def parse_token_input(raw: str) -> AuthTokens:
    """Accept the browser's ``oidc.user:...`` JSON, a JSON object with access/refresh tokens, or a bare access token."""
    raw = raw.strip()
    if not raw:
        raise AuthError("empty token input")
    if raw.startswith("{"):
        d = json.loads(raw)
        access = d.get("access_token")
        if not access:
            raise AuthError("JSON has no access_token")
        exp = None
        if isinstance(d.get("expires_at"), (int, float)):
            exp = datetime.fromtimestamp(int(d["expires_at"]), tz=timezone.utc)
        return AuthTokens(access, d.get("refresh_token"), exp or _jwt_exp(access), d.get("cookie"))
    return AuthTokens(raw, None, _jwt_exp(raw))


class TokenVault:
    def __init__(self, settings: Settings, pool: asyncpg.Pool, *, transport: httpx.AsyncBaseTransport | None = None):
        if not settings.secret_key:
            raise AuthError("FPL_SECRET_KEY is not set (generate one with `python -m app auth keygen`)")
        self._fernet = Fernet(settings.secret_key.encode())
        self.settings = settings
        self.pool = pool
        self._transport = transport

    @staticmethod
    def generate_key() -> str:
        return Fernet.generate_key().decode()

    async def ensure_config_row(self, team_id: int) -> None:
        # Auto-execution starts disabled: a fresh install never touches the team until armed explicitly.
        await self.pool.execute(
            "INSERT INTO system_config (team_id, auto_execute_enabled) VALUES ($1, FALSE) ON CONFLICT (team_id) DO NOTHING",
            team_id,
        )

    async def store(self, team_id: int, tokens: AuthTokens) -> None:
        await self.ensure_config_row(team_id)
        await self.pool.execute(
            """UPDATE system_config SET encrypted_auth_cookie = $2, auth_expires_at = $3, updated_at = NOW()
               WHERE team_id = $1""",
            team_id, self._fernet.encrypt(tokens.to_json().encode()).decode(), tokens.expires_at,
        )

    async def load(self, team_id: int) -> AuthTokens | None:
        blob = await self.pool.fetchval("SELECT encrypted_auth_cookie FROM system_config WHERE team_id = $1", team_id)
        if not blob:
            return None
        try:
            return AuthTokens.from_json(self._fernet.decrypt(blob.encode()).decode())
        except InvalidToken as exc:
            raise AuthError("stored tokens cannot be decrypted with FPL_SECRET_KEY") from exc

    async def refresh(self, team_id: int, tokens: AuthTokens) -> AuthTokens:
        if not tokens.refresh_token:
            raise AuthError("access token expired and no refresh token is stored; re-import tokens")
        async with httpx.AsyncClient(timeout=20, transport=self._transport) as client:
            resp = await client.post(
                self.settings.oidc_token_url,
                data={"grant_type": "refresh_token", "refresh_token": tokens.refresh_token, "client_id": self.settings.oidc_client_id},
                headers={"Accept": "application/json"},
            )
        if resp.status_code != 200:
            raise AuthError(f"token refresh rejected (HTTP {resp.status_code}); re-import tokens")
        d = resp.json()
        access = d["access_token"]
        expires = datetime.now(timezone.utc) + timedelta(seconds=int(d["expires_in"])) if d.get("expires_in") else _jwt_exp(access)
        fresh = AuthTokens(access, d.get("refresh_token") or tokens.refresh_token, expires, tokens.cookie)
        await self.store(team_id, fresh)
        log.info("FPL access token refreshed; valid until %s", fresh.expires_at)
        return fresh

    async def valid_tokens(self, team_id: int) -> AuthTokens:
        tokens = await self.load(team_id)
        if tokens is None:
            raise AuthError("no FPL tokens stored; run `python -m app auth set`")
        if tokens.expired():
            tokens = await self.refresh(team_id, tokens)
        return tokens

    async def record_check(self, team_id: int, ok: bool, error: str | None = None) -> None:
        await self.ensure_config_row(team_id)
        await self.pool.execute(
            "UPDATE system_config SET auth_last_check_at = NOW(), auth_last_check_ok = $2, auth_last_error = $3 WHERE team_id = $1",
            team_id, ok, error,
        )

    async def health(self, team_id: int) -> dict[str, Any]:
        row = await self.pool.fetchrow(
            """SELECT encrypted_auth_cookie IS NOT NULL AS has_tokens, auth_expires_at, auth_last_check_at,
                      auth_last_check_ok, auth_last_error FROM system_config WHERE team_id = $1""",
            team_id,
        )
        out = dict(row) if row else {"has_tokens": False}
        if out.get("has_tokens"):
            try:
                tokens = await self.load(team_id)
                out["session_expires_at"] = tokens.session_expires_at if tokens else None
            except AuthError:
                out["session_expires_at"] = None
        return out

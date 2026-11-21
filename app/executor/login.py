"""Browser-assisted FPL login (OIDC authorization code + PKCE), replacing manual token copying.

FPL sessions end 180 days after sign-in no matter how often they are renewed, so a person
has to sign in about twice a season. This makes that a one-minute job:

1. ``start_login`` returns a normal FPL sign-in link (the same client and return address the
   FPL web app uses) and remembers the PKCE verifier for 15 minutes.
2. You sign in as usual and land back on fantasy.premierleague.com; copy that page address.
3. ``complete_login`` exchanges the one-time code for tokens and stores them encrypted.

Your password only ever goes to FPL's own sign-in page.
"""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
import urllib.parse
from datetime import datetime, timedelta, timezone

import httpx
from redis.asyncio import Redis

from app.config import Settings
from app.executor.vault import AuthError, AuthTokens, TokenVault, _jwt_exp

PENDING_KEY = "fpl:login:{state}"
PENDING_TTL = 900


def _challenge(verifier: str) -> str:
    return base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()


async def start_login(settings: Settings, redis: Redis, team_id: int) -> str:
    state = secrets.token_urlsafe(24)
    verifier = secrets.token_urlsafe(64)
    await redis.set(PENDING_KEY.format(state=state), json.dumps({"verifier": verifier, "team_id": team_id}), ex=PENDING_TTL)
    params = {
        "client_id": settings.oidc_client_id, "redirect_uri": settings.oidc_redirect_uri, "response_type": "code",
        "scope": settings.oidc_scope, "state": state, "code_challenge": _challenge(verifier), "code_challenge_method": "S256",
    }
    return f"{settings.oidc_authorize_url}?{urllib.parse.urlencode(params)}"


def parse_redirect(redirect: str) -> tuple[str, str]:
    """Pull ``code`` and ``state`` out of the address you landed on after signing in."""
    query = urllib.parse.urlparse(redirect.strip()).query or redirect.strip().lstrip("?")
    q = urllib.parse.parse_qs(query)
    if "error" in q:
        raise AuthError(f"sign-in failed: {q.get('error_description', q['error'])[0]}")
    if "code" not in q or "state" not in q:
        raise AuthError("that address has no sign-in code; copy the full address after signing in")
    return q["code"][0], q["state"][0]


async def complete_login(
    settings: Settings, redis: Redis, vault: TokenVault, redirect: str, *, transport: httpx.AsyncBaseTransport | None = None
) -> AuthTokens:
    code, state = parse_redirect(redirect)
    key = PENDING_KEY.format(state=state)
    pending = await redis.get(key)
    if pending is None:
        raise AuthError("sign-in link expired or was not started here; start the login again")
    await redis.delete(key)  # single use
    pending = json.loads(pending)
    async with httpx.AsyncClient(timeout=20, transport=transport) as client:
        resp = await client.post(
            settings.oidc_token_url,
            data={"grant_type": "authorization_code", "code": code, "redirect_uri": settings.oidc_redirect_uri,
                  "client_id": settings.oidc_client_id, "code_verifier": pending["verifier"]},
            headers={"Accept": "application/json"},
        )
    if resp.status_code != 200:
        raise AuthError(f"FPL refused the sign-in code (HTTP {resp.status_code}); the code is single-use and short-lived, try again")
    d = resp.json()
    access = d["access_token"]
    expires = datetime.now(timezone.utc) + timedelta(seconds=int(d["expires_in"])) if d.get("expires_in") else _jwt_exp(access)
    tokens = AuthTokens(access, d.get("refresh_token"), expires)
    await vault.store(int(pending["team_id"]), tokens)
    return tokens

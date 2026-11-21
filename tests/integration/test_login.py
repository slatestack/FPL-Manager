import base64
import hashlib
import json
import os
import urllib.parse
from datetime import datetime, timedelta, timezone

import httpx
import pytest
import respx
from cryptography.fernet import Fernet
from redis.asyncio import Redis

from app.config import Settings
from app.executor.login import complete_login, parse_redirect, start_login
from app.executor.service import Executor
from app.executor.vault import AuthError, AuthTokens, TokenVault

pytestmark = pytest.mark.skipif(not os.environ.get("TEST_REDIS_URL"), reason="TEST_REDIS_URL not set")
TEAM = 4242


def _jwt(claims: dict) -> str:
    enc = lambda d: base64.urlsafe_b64encode(json.dumps(d).encode()).decode().rstrip("=")  # noqa: E731
    return f"{enc({'alg': 'RS256'})}.{enc(claims)}.sig"


@pytest.fixture
async def ctx(pool):
    settings = Settings(database_url="", team_id=TEAM, secret_key=Fernet.generate_key().decode(),
                        api_base_url="https://fpl.test/api", http_max_retries=0)
    redis = Redis.from_url(os.environ["TEST_REDIS_URL"])
    await redis.flushdb()
    yield settings, redis, TokenVault(settings, pool)
    await redis.aclose()


def test_parse_redirect():
    assert parse_redirect("https://fantasy.premierleague.com/?code=abc&state=xyz") == ("abc", "xyz")
    with pytest.raises(AuthError, match="no sign-in code"):
        parse_redirect("https://fantasy.premierleague.com/")
    with pytest.raises(AuthError, match="access_denied|denied"):
        parse_redirect("https://fantasy.premierleague.com/?error=access_denied&error_description=denied")


async def test_full_sign_in_with_pkce(ctx):
    settings, redis, vault = ctx
    url = await start_login(settings, redis, TEAM)
    q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
    assert q["client_id"] == [settings.oidc_client_id] and q["redirect_uri"] == ["https://fantasy.premierleague.com/"]
    assert q["code_challenge_method"] == ["S256"] and "offline_access" in q["scope"][0]
    refresh = _jwt({"exp": int((datetime.now(timezone.utc) + timedelta(days=180)).timestamp())})
    seen = {}

    def token_endpoint(request):
        form = dict(urllib.parse.parse_qsl(request.content.decode()))
        seen.update(form)
        challenge = base64.urlsafe_b64encode(hashlib.sha256(form["code_verifier"].encode()).digest()).rstrip(b"=").decode()
        assert challenge == q["code_challenge"][0], "verifier must match the challenge in the link"
        return httpx.Response(200, json={"access_token": "AT", "refresh_token": refresh, "expires_in": 3600})

    with respx.mock() as router:
        router.post(settings.oidc_token_url).mock(side_effect=token_endpoint)
        tokens = await complete_login(settings, redis, vault, f"https://fantasy.premierleague.com/?code=C1&state={q['state'][0]}")
    assert seen["grant_type"] == "authorization_code" and seen["code"] == "C1"
    assert (await vault.load(TEAM)).access_token == "AT"
    assert 179 <= (tokens.session_expires_at - datetime.now(timezone.utc)).days <= 180
    with pytest.raises(AuthError, match="expired or was not started"):
        await complete_login(settings, redis, vault, f"https://fantasy.premierleague.com/?code=C1&state={q['state'][0]}")


async def test_foreign_state_is_rejected(ctx):
    settings, redis, vault = ctx
    with pytest.raises(AuthError, match="expired or was not started"):
        await complete_login(settings, redis, vault, "https://fantasy.premierleague.com/?code=C&state=not-ours")


class Notes:
    def __init__(self):
        self.messages = []

    async def send(self, text, *, urgent=False, kill_button=False):
        self.messages.append((text, urgent))


async def test_keepalive_renews_and_warns_before_session_end(ctx, pool):
    settings, redis, vault = ctx
    soon = datetime.now(timezone.utc) + timedelta(days=5)
    await vault.store(TEAM, AuthTokens("OLD", _jwt({"exp": int(soon.timestamp())}), datetime.now(timezone.utc) + timedelta(minutes=10)))
    notes = Notes()
    with respx.mock() as router:
        router.post(settings.oidc_token_url).mock(return_value=httpx.Response(
            200, json={"access_token": "NEW", "refresh_token": _jwt({"exp": int(soon.timestamp()), "jti": "2"}), "expires_in": 3600}))
        router.get(f"https://fpl.test/api/my-team/{TEAM}/").mock(return_value=httpx.Response(200, json={"picks": [], "chips": [], "transfers": {}}))
        ok = await Executor(settings, pool, redis, notes).keepalive(TEAM)
    assert ok and (await vault.load(TEAM)).access_token == "NEW", "renewed proactively within 30 minutes of expiry"
    assert "session ends in" in notes.messages[-1][0] and "/login" in notes.messages[-1][0]

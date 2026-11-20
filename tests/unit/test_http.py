import httpx
import pytest
import respx

from app.ingestion.http import ResilientClient, UpstreamError


class FakeSleep:
    def __init__(self):
        self.calls: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


@respx.mock
async def test_retries_on_503_then_succeeds():
    route = respx.get("https://x.test/a").mock(
        side_effect=[httpx.Response(503), httpx.Response(429, headers={"Retry-After": "7"}), httpx.Response(200, json={"ok": 1})]
    )
    sleep = FakeSleep()
    async with ResilientClient(max_retries=3, sleep=sleep) as client:
        assert await client.get_json("https://x.test/a") == {"ok": 1}
    assert route.call_count == 3
    assert len(sleep.calls) == 2 and sleep.calls[1] == 7.0, "Retry-After must be honoured"


@respx.mock
async def test_non_retryable_status_raises_immediately():
    route = respx.get("https://x.test/missing").mock(return_value=httpx.Response(404))
    sleep = FakeSleep()
    async with ResilientClient(max_retries=5, sleep=sleep) as client:
        with pytest.raises(UpstreamError) as err:
            await client.get_json("https://x.test/missing")
    assert err.value.status_code == 404 and route.call_count == 1 and sleep.calls == []


@respx.mock
async def test_gives_up_after_max_retries_on_transport_errors():
    route = respx.get("https://x.test/down").mock(side_effect=httpx.ConnectError("boom"))
    sleep = FakeSleep()
    async with ResilientClient(max_retries=2, sleep=sleep) as client:
        with pytest.raises(UpstreamError):
            await client.get_json("https://x.test/down")
    assert route.call_count == 3 and len(sleep.calls) == 2


@respx.mock
async def test_error_never_contains_query_string_secrets():
    respx.get("https://odds.test/odds").mock(return_value=httpx.Response(401))
    async with ResilientClient(max_retries=0) as client:
        with pytest.raises(UpstreamError) as err:
            await client.get_json("https://odds.test/odds", params={"apiKey": "SECRET"})
    assert "SECRET" not in str(err.value) and "SECRET" not in (err.value.url or "")


def test_backoff_is_bounded():
    client = ResilientClient(backoff_base=1.0, backoff_max=10.0)
    for attempt in range(12):
        assert 0.5 <= client._backoff(attempt) <= 10.0

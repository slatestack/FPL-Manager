"""Notifications: WhatsApp (CallMeBot), Telegram (with inline kill-switch button), generic webhook, or log only."""

from __future__ import annotations

import logging
from typing import Any, Protocol

import httpx

from app.config import Settings

log = logging.getLogger(__name__)

KILL_BUTTON = {"inline_keyboard": [[{"text": "🛑 Kill switch (disable auto-execute)", "callback_data": "killswitch"}]]}


class Notifier(Protocol):
    async def send(self, text: str, *, urgent: bool = False, kill_button: bool = False) -> None: ...


class LogNotifier:
    async def send(self, text: str, *, urgent: bool = False, kill_button: bool = False) -> None:
        (log.warning if urgent else log.info)("notification: %s", text)


class TelegramNotifier:
    def __init__(self, token: str, chat_id: str, *, transport: httpx.AsyncBaseTransport | None = None):
        self._url = f"https://api.telegram.org/bot{token}"
        self.chat_id = chat_id
        self._transport = transport

    async def call(self, method: str, payload: dict[str, Any], timeout: float = 20) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=timeout, transport=self._transport) as c:
            resp = await c.post(f"{self._url}/{method}", json=payload)
        data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
        if resp.status_code != 200 or not data.get("ok"):
            # Never log the URL: it contains the bot token.
            raise RuntimeError(f"telegram {method} failed (HTTP {resp.status_code}): {data.get('description', '')}")
        return data

    async def send(self, text: str, *, urgent: bool = False, kill_button: bool = False) -> None:
        payload: dict[str, Any] = {"chat_id": self.chat_id, "text": ("🚨 " if urgent else "") + text, "disable_web_page_preview": True}
        if kill_button:
            payload["reply_markup"] = KILL_BUTTON
        await self.call("sendMessage", payload)


class WhatsAppNotifier:
    """WhatsApp alerts through CallMeBot's free personal relay (one-way: it can't receive commands).

    The API key travels in the query string, so it is never logged or put in an error message.
    WhatsApp messages can't carry buttons; when an action could be stopped, the message says how.
    """

    MAX_CHARS = 1500

    def __init__(self, phone: str, apikey: str, *, api_url: str = "https://api.callmebot.com/whatsapp.php",
                 transport: httpx.AsyncBaseTransport | None = None):
        self.phone = phone.replace(" ", "")
        self._apikey = apikey
        self.api_url = api_url
        self._transport = transport

    async def send(self, text: str, *, urgent: bool = False, kill_button: bool = False) -> None:
        body = ("🚨 " if urgent else "") + text
        if kill_button:
            body += "\n\nTo stop automatic submission: open http://localhost:8000 on your PC and press Kill switch."
        if len(body) > self.MAX_CHARS:
            body = body[: self.MAX_CHARS - 1] + "…"
        async with httpx.AsyncClient(timeout=30, transport=self._transport) as c:
            resp = await c.get(self.api_url, params={"phone": self.phone, "text": body, "apikey": self._apikey})
        if resp.status_code != 200 or "error" in resp.text.lower()[:300]:
            raise RuntimeError(f"WhatsApp message not accepted by CallMeBot (HTTP {resp.status_code})")


class WebhookNotifier:
    def __init__(self, url: str, *, transport: httpx.AsyncBaseTransport | None = None):
        self.url = url
        self._transport = transport

    async def send(self, text: str, *, urgent: bool = False, kill_button: bool = False) -> None:
        async with httpx.AsyncClient(timeout=15, transport=self._transport) as c:
            await c.post(self.url, json={"text": text, "urgent": urgent})


class MultiNotifier:
    def __init__(self, notifiers: list[Notifier]):
        self.notifiers = notifiers

    async def send(self, text: str, *, urgent: bool = False, kill_button: bool = False) -> None:
        for n in self.notifiers:
            try:
                await n.send(text, urgent=urgent, kill_button=kill_button)
            except Exception as exc:  # one broken channel must not hide the message from the others
                log.error("notifier %s failed: %s", type(n).__name__, exc)


def build_notifier(settings: Settings) -> MultiNotifier:
    notifiers: list[Notifier] = [LogNotifier()]
    if settings.telegram_bot_token and settings.telegram_chat_id:
        notifiers.append(TelegramNotifier(settings.telegram_bot_token, settings.telegram_chat_id))
    if settings.whatsapp_phone and settings.whatsapp_apikey:
        notifiers.append(WhatsAppNotifier(settings.whatsapp_phone, settings.whatsapp_apikey, api_url=settings.whatsapp_api_url))
    if settings.alert_webhook_url:
        notifiers.append(WebhookNotifier(settings.alert_webhook_url))
    return MultiNotifier(notifiers)

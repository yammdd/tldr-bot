"""Lightweight aiohttp HTTP health endpoint (/healthz) for container & cloud deployments."""

from __future__ import annotations

import logging
from typing import Any

from aiohttp import web

logger = logging.getLogger(__name__)


class HealthServer:
    """Exposes GET /healthz with non-sensitive bot readiness metrics."""

    def __init__(self, bot: Any, *, host: str = "0.0.0.0", port: int = 8080) -> None:
        self._bot = bot
        self._host = host
        self._port = port
        self._runner: web.AppRunner | None = None

    async def _handle_healthz(self, request: web.Request) -> web.Response:
        ready = bool(self._bot.is_ready())
        latency = getattr(self._bot, "latency", 0.0)
        latency_ms = round(latency * 1000, 1) if latency == latency else None
        payload = {
            "status": "ok" if ready else "starting",
            "ready": ready,
            "guilds": len(getattr(self._bot, "guilds", ())),
            "shards": getattr(self._bot, "shard_count", 1) or 1,
            "latency_ms": latency_ms,
            "active_jobs": self._bot.concurrency.global_active,
        }
        return web.json_response(payload, status=200 if ready else 503)

    async def start(self) -> None:
        app = web.Application()
        app.router.add_get("/healthz", self._handle_healthz)
        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self._host, self._port)
        await site.start()
        logger.info("Health endpoint listening on http://%s:%d/healthz", self._host, self._port)

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

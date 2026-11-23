"""Fair multi-server concurrency guard preventing one busy server from starving others."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator


class ConcurrencyLimitError(Exception):
    """Raised when per-guild or global concurrent summary capacity is temporarily full."""

    def __init__(self, scope: str) -> None:
        super().__init__(scope)
        self.scope = scope  # "guild_busy" or "global_busy"


class ConcurrencyGuard:
    """Non-blocking per-guild and global in-flight task limiter."""

    def __init__(
        self,
        *,
        max_per_guild: int = 2,
        max_global: int = 20,
    ) -> None:
        self._max_per_guild = max(1, max_per_guild)
        self._max_global = max(1, max_global)
        self._guild_active: dict[int, int] = {}
        self._global_active: int = 0

    @property
    def global_active(self) -> int:
        return self._global_active

    def guild_active(self, guild_id: int) -> int:
        return self._guild_active.get(guild_id, 0)

    @asynccontextmanager
    async def acquire(self, guild_id: int) -> AsyncIterator[None]:
        current_guild = self._guild_active.get(guild_id, 0)
        if current_guild >= self._max_per_guild:
            raise ConcurrencyLimitError("guild_busy")
        if self._global_active >= self._max_global:
            raise ConcurrencyLimitError("global_busy")

        self._guild_active[guild_id] = current_guild + 1
        self._global_active += 1
        try:
            yield
        finally:
            self._global_active = max(0, self._global_active - 1)
            remaining = self._guild_active.get(guild_id, 1) - 1
            if remaining <= 0:
                self._guild_active.pop(guild_id, None)
            else:
                self._guild_active[guild_id] = remaining

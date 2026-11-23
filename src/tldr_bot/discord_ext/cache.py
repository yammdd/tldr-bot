"""In-memory TTL summary cache and per-user cooldown tracker."""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CachedSummaryEntry:
    summary_result: Any
    created_at_monotonic: float


class SummaryCache:
    """Bounded LRU + TTL RAM cache for summary results (avoids duplicate LLM bills)."""

    def __init__(self, *, ttl_seconds: int = 180, max_entries: int = 128) -> None:
        self._ttl = max(0, ttl_seconds)
        self._max_entries = max(1, max_entries)
        self._store: OrderedDict[str, CachedSummaryEntry] = OrderedDict()

    @staticmethod
    def make_key(
        *,
        guild_id: int,
        channel_id: int,
        latest_message_id: int,
        scope_tag: str,
        style_id: str,
        language_code: str,
        focus_query: str | None,
        model_id: str,
    ) -> str:
        norm_focus = (focus_query or "").strip().lower()
        return (
            f"{guild_id}:{channel_id}:{latest_message_id}:{scope_tag}:"
            f"{style_id}:{language_code}:{model_id}:{norm_focus}"
        )

    def get(self, key: str) -> Any | None:
        if self._ttl <= 0:
            return None
        entry = self._store.get(key)
        if entry is None:
            return None
        if time.monotonic() - entry.created_at_monotonic > self._ttl:
            self._store.pop(key, None)
            return None
        self._store.move_to_end(key)
        return entry.summary_result

    def set(self, key: str, summary_result: Any) -> None:
        if self._ttl <= 0:
            return
        self._store[key] = CachedSummaryEntry(
            summary_result=summary_result,
            created_at_monotonic=time.monotonic(),
        )
        self._store.move_to_end(key)
        while len(self._store) > self._max_entries:
            self._store.popitem(last=False)

    def clear_guild(self, guild_id: int) -> None:
        prefix = f"{guild_id}:"
        to_delete = [k for k in self._store if k.startswith(prefix)]
        for k in to_delete:
            self._store.pop(k, None)


class UserCooldownTracker:
    """Simple per-(guild_id, user_id) cooldown guard to prevent API spam."""

    def __init__(self, cooldown_seconds: int = 10) -> None:
        self._cooldown = max(0, cooldown_seconds)
        self._last_used: dict[tuple[int, int], float] = {}

    def check_remaining(self, guild_id: int, user_id: int) -> float:
        if self._cooldown <= 0:
            return 0.0
        key = (guild_id, user_id)
        now = time.monotonic()
        last = self._last_used.get(key)
        if last is None:
            return 0.0
        elapsed = now - last
        if elapsed < self._cooldown:
            return self._cooldown - elapsed
        return 0.0

    def touch(self, guild_id: int, user_id: int) -> None:
        if self._cooldown <= 0:
            return
        now = time.monotonic()
        self._last_used[(guild_id, user_id)] = now
        # Opportunistic pruning if map grows large
        if len(self._last_used) > 2048:
            cutoff = now - self._cooldown * 2
            stale = [k for k, ts in self._last_used.items() if ts < cutoff]
            for k in stale:
                self._last_used.pop(k, None)

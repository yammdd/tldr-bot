"""Discord channel history fetcher with duration parsing, progress updates, and hard caps."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable

_DURATION_PART_RE = re.compile(r"(\d+)\s*([dhm])", re.IGNORECASE)


@dataclass(frozen=True)
class FetchResult:
    messages: list[Any]  # Chronological order: oldest -> newest
    fetched_count: int
    hit_hard_cap: bool
    latest_message_id: int


FetchProgressCallback = Callable[[int], Awaitable[None]]


def parse_since_duration(raw: str, *, max_days: int = 7) -> timedelta | None:
    """Parse durations like '30m', '2h', '12h', '1h30m', '3d', 'today' within [5m, max_days]."""
    cleaned = (raw or "").strip().lower()
    if not cleaned:
        return None

    if cleaned in {"today", "homnay", "hôm nay"}:
        now = datetime.now(timezone.utc)
        delta = now - now.replace(hour=0, minute=0, second=0, microsecond=0)
        return max(delta, timedelta(minutes=30))

    matches = _DURATION_PART_RE.findall(cleaned)
    if not matches:
        return None

    # Ensure there is no stray garbage text besides digits, units, and spaces
    stripped = _DURATION_PART_RE.sub("", cleaned).strip()
    if stripped:
        return None

    total_minutes = 0
    for amount_str, unit in matches:
        amount = int(amount_str)
        if unit == "d":
            total_minutes += amount * 1440
        elif unit == "h":
            total_minutes += amount * 60
        elif unit == "m":
            total_minutes += amount

    if total_minutes < 5 or total_minutes > max_days * 1440:
        return None
    return timedelta(minutes=total_minutes)


async def fetch_channel_messages(
    channel: Any,
    *,
    count: int | None = None,
    since: timedelta | None = None,
    after_message: Any | None = None,
    hard_cap: int = 1500,
    on_progress: FetchProgressCallback | None = None,
) -> FetchResult:
    """Fetch messages in chronological order (oldest -> newest) while respecting hard_cap."""
    collected: list[Any] = []
    hit_hard_cap = False

    if after_message is not None:
        # Context menu mode: summarize starting FROM the selected message forward up to hard_cap
        collected.append(after_message)
        remaining = max(0, hard_cap - 1)
        if remaining > 0:
            async for msg in channel.history(
                limit=remaining + 1, after=after_message, oldest_first=True
            ):
                if len(collected) >= hard_cap:
                    hit_hard_cap = True
                    break
                collected.append(msg)
                if on_progress and len(collected) % 250 == 0:
                    await on_progress(len(collected))
    elif since is not None:
        cutoff = datetime.now(timezone.utc) - since
        async for msg in channel.history(limit=hard_cap + 1, oldest_first=False):
            msg_time = getattr(msg, "created_at", None)
            if msg_time is not None:
                if msg_time.tzinfo is None:
                    msg_time = msg_time.replace(tzinfo=timezone.utc)
                if msg_time < cutoff:
                    break
            if len(collected) >= hard_cap:
                hit_hard_cap = True
                break
            collected.append(msg)
            if on_progress and len(collected) % 250 == 0:
                await on_progress(len(collected))
        collected.reverse()
    else:
        target = min(max(1, count or 100), hard_cap)
        async for msg in channel.history(limit=target, oldest_first=False):
            collected.append(msg)
            if on_progress and len(collected) % 250 == 0:
                await on_progress(len(collected))
        collected.reverse()

    latest_id = int(getattr(collected[-1], "id", 0)) if collected else 0
    return FetchResult(
        messages=collected,
        fetched_count=len(collected),
        hit_hard_cap=hit_hard_cap,
        latest_message_id=latest_id,
    )

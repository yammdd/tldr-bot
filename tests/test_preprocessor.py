"""Tests for conversation preprocessing, reply disentanglement, coalescing, and duration parsing."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from tldr_bot.discord_ext.fetcher import parse_since_duration
from tldr_bot.discord_ext.preprocessor import preprocess_messages


@dataclass
class _User:
    id: int
    display_name: str
    name: str
    bot: bool = False


@dataclass
class _Ref:
    message_id: int
    resolved: Any = None


@dataclass
class _Msg:
    id: int
    author: _User
    content: str
    created_at: datetime
    mentions: list[_User] = field(default_factory=list)
    channel_mentions: list[Any] = field(default_factory=list)
    role_mentions: list[Any] = field(default_factory=list)
    attachments: list[Any] = field(default_factory=list)
    embeds: list[Any] = field(default_factory=list)
    reference: _Ref | None = None
    jump_url: str = ""

    def __post_init__(self) -> None:
        if not self.jump_url:
            self.jump_url = f"https://discord.com/channels/1/2/{self.id}"


def test_preprocessor_speaker_disambiguation_reply_coalescing_and_personal_radar() -> None:
    t0 = datetime(2026, 11, 23, 14, 0, tzinfo=timezone.utc)
    alex1 = _User(id=10, display_name="Alex", name="alex_dev")
    alex2 = _User(id=11, display_name="Alex", name="alex_ops")
    maya = _User(id=20, display_name="Maya", name="maya_pm")
    bot_user = _User(id=99, display_name="MusicBot", name="musicbot", bot=True)

    m1 = _Msg(id=101, author=alex1, content="Let us grab ramen tonight at 7:30 PM", created_at=t0)
    # Same speaker burst within 30s -> should coalesce into m1
    m2 = _Msg(id=102, author=alex1, content="near downtown", created_at=t0 + timedelta(seconds=25))
    # Bot noise -> filtered out
    m_bot = _Msg(id=103, author=bot_user, content="Now playing: Lo-fi beats", created_at=t0 + timedelta(seconds=40))
    # Emoji-only spam -> filtered out
    m_emoji = _Msg(id=104, author=alex2, content="<:pepe:123456> !!!", created_at=t0 + timedelta(seconds=50))
    # Short crucial confirmation ("ok") replying to m1 -> MUST BE KEPT with reply tag
    m3 = _Msg(
        id=105,
        author=maya,
        content="ok",
        created_at=t0 + timedelta(seconds=60),
        reference=_Ref(message_id=101, resolved=m1),
    )
    # Gap > 30 minutes -> session break inserted
    m4 = _Msg(
        id=106,
        author=alex2,
        content="Hey <@20> remember to send the launch slides",
        created_at=t0 + timedelta(minutes=55),
        mentions=[maya],
    )
    # Mega paste -> truncated cleanly
    m5 = _Msg(
        id=107,
        author=alex1,
        content="ERROR_LOG " + ("x" * 1200),
        created_at=t0 + timedelta(minutes=56),
    )

    conv = preprocess_messages(
        [m1, m2, m_bot, m_emoji, m3, m4, m5],
        requester_id=20,
        max_single_message_chars=300,
        session_gap_minutes=30,
    )

    assert conv.raw_message_count == 7
    assert conv.kept_message_count == 5

    transcript = "\n".join(turn.text_line for turn in conv.turns)
    # Duplicate "Alex" display names disambiguated with (@username)
    assert "Alex (@alex_dev)" in transcript
    assert "Alex (@alex_ops)" in transcript
    # Burst messages m1 + m2 coalesced on the same line
    assert "grab ramen tonight at 7:30 PM | near downtown" in transcript
    # Reply graph preserved for short confirmation "ok"
    assert "->(Alex (@alex_dev) #1:" in transcript
    assert ": ok" in transcript
    # Mention <@20> resolved to @Maya
    assert "@Maya remember to send the launch slides" in transcript
    # Session break marker inserted between 14:01 and 14:55
    assert any(turn.is_session_break for turn in conv.turns)
    # Long paste truncated
    assert "chars truncated]" in transcript

    # Personal Mention Radar captured m4 targeting Maya (id=20)
    assert len(conv.requester_mentions) == 1
    assert conv.requester_mentions[0].author_name == "Alex (@alex_ops)"
    assert conv.requester_mentions[0].jump_url.endswith("/106")


def test_parse_since_duration() -> None:
    assert parse_since_duration("30m", max_days=7) == timedelta(minutes=30)
    assert parse_since_duration("2h", max_days=7) == timedelta(hours=2)
    assert parse_since_duration("1h30m", max_days=7) == timedelta(hours=1, minutes=30)
    assert parse_since_duration("7d", max_days=7) == timedelta(days=7)
    assert parse_since_duration("today", max_days=7) is not None

    assert parse_since_duration("2m", max_days=7) is None
    assert parse_since_duration("8d", max_days=7) is None
    assert parse_since_duration("abc", max_days=7) is None
    assert parse_since_duration("2h foo", max_days=7) is None

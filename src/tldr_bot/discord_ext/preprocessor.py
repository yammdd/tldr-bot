"""Conversation preprocessor: speaker resolution, reply graph, coalescing, session breaks & Personal Radar."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Sequence

from tldr_bot.llm.token_budget import TranscriptTurn

_CUSTOM_EMOJI_RE = re.compile(r"<a?:([a-zA-Z0-9_]+):\d+>")
_USER_MENTION_RE = re.compile(r"<@!?(\d+)>")
_CHANNEL_MENTION_RE = re.compile(r"<#(\d+)>")
_ROLE_MENTION_RE = re.compile(r"<@&(\d+)>")
_WHITESPACE_RE = re.compile(r"[ \t]+")
_NEWLINES_RE = re.compile(r"\n{2,}")
_WORD_CHAR_RE = re.compile(r"[0-9A-Za-z\u00c0-\u1ef9\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")

# Short symbolic or shorthand confirmations that carry real agreement/decision meaning in group chats
_MEANINGFUL_SHORT_WORDS: frozenset[str] = frozenset(
    {
        "ok",
        "oke",
        "okie",
        "okay",
        "k",
        "kk",
        "yes",
        "no",
        "yep",
        "nope",
        "nah",
        "pass",
        "skip",
        "done",
        "+1",
        "-1",
    }
)


@dataclass(frozen=True)
class ParticipantStat:
    user_id: int
    display_name: str
    message_count: int


@dataclass(frozen=True)
class MentionHit:
    ref_id: int
    message_id: int
    author_name: str
    timestamp_hhmm: str
    snippet: str
    jump_url: str


@dataclass(frozen=True)
class PreprocessedConversation:
    turns: tuple[TranscriptTurn, ...]
    ref_to_jump_url: dict[int, str]
    valid_ref_ids: set[int]
    participants: tuple[ParticipantStat, ...]
    participants_summary: str
    requester_mentions: tuple[MentionHit, ...]
    raw_message_count: int
    kept_message_count: int
    time_range_label: str


def _get_author_display_and_username(user: Any) -> tuple[int, str, str]:
    uid = int(getattr(user, "id", 0))
    display = str(
        getattr(user, "display_name", None)
        or getattr(user, "global_name", None)
        or getattr(user, "name", None)
        or f"User{uid}"
    ).strip()
    username = str(getattr(user, "name", display)).strip()
    return uid, display, username


def _build_unique_speaker_map(messages: Sequence[Any]) -> dict[int, str]:
    """Map user_id -> unique display name (appends @username only when display_names collide)."""
    id_to_info: dict[int, tuple[str, str]] = {}

    def _register(user: Any) -> None:
        if user is None:
            return
        uid, display, username = _get_author_display_and_username(user)
        if uid and uid not in id_to_info:
            id_to_info[uid] = (display, username)

    for msg in messages:
        _register(getattr(msg, "author", None))
        for mentioned in getattr(msg, "mentions", ()) or ():
            _register(mentioned)
        ref = getattr(msg, "reference", None)
        resolved = getattr(ref, "resolved", None) if ref is not None else None
        if resolved is not None:
            _register(getattr(resolved, "author", None))

    display_counts = Counter(info[0].lower() for info in id_to_info.values())
    resolved_map: dict[int, str] = {}
    for uid, (display, username) in id_to_info.items():
        if display_counts[display.lower()] > 1 and username.lower() != display.lower():
            resolved_map[uid] = f"{display} (@{username})"
        else:
            resolved_map[uid] = display
    return resolved_map


def _format_attachments_and_embeds(msg: Any) -> str:
    tags: list[str] = []
    for att in getattr(msg, "attachments", ()) or ():
        filename = str(getattr(att, "filename", "file"))
        ctype = str(getattr(att, "content_type", "") or "").lower()
        if ctype.startswith("image/") or filename.lower().endswith(
            (".png", ".jpg", ".jpeg", ".gif", ".webp")
        ):
            tags.append(f"[img: {filename}]")
        elif ctype.startswith("video/") or filename.lower().endswith((".mp4", ".mov", ".webm")):
            tags.append(f"[video: {filename}]")
        else:
            tags.append(f"[file: {filename}]")

    for embed in (getattr(msg, "embeds", ()) or ())[:2]:
        title = str(getattr(embed, "title", "") or "").strip()
        desc = str(getattr(embed, "description", "") or "").strip()
        if title:
            short_desc = f" - {desc[:90]}" if desc else ""
            tags.append(f"[embed: {title[:80]}{short_desc}]")
        elif desc:
            tags.append(f"[embed: {desc[:110]}]")

    return " ".join(tags)


def _clean_message_text(
    msg: Any,
    speaker_map: dict[int, str],
    *,
    max_chars: int,
) -> str:
    content = str(getattr(msg, "content", "") or "")

    # Resolve mentions -> human display names
    def _repl_user(m: re.Match[str]) -> str:
        uid = int(m.group(1))
        name = speaker_map.get(uid)
        return f"@{name}" if name else "@user"

    content = _USER_MENTION_RE.sub(_repl_user, content)

    channel_lookup = {
        int(getattr(ch, "id", 0)): str(getattr(ch, "name", "channel"))
        for ch in (getattr(msg, "channel_mentions", ()) or ())
    }
    content = _CHANNEL_MENTION_RE.sub(
        lambda m: f"#{channel_lookup.get(int(m.group(1)), 'channel')}", content
    )

    role_lookup = {
        int(getattr(r, "id", 0)): str(getattr(r, "name", "role"))
        for r in (getattr(msg, "role_mentions", ()) or ())
    }
    content = _ROLE_MENTION_RE.sub(
        lambda m: f"@{role_lookup.get(int(m.group(1)), 'role')}", content
    )

    # Compact custom emojis <a:pepe:123456> -> :pepe:
    content = _CUSTOM_EMOJI_RE.sub(r":\1:", content)
    content = _NEWLINES_RE.sub("\n", content)
    content = _WHITESPACE_RE.sub(" ", content).strip()

    extra = _format_attachments_and_embeds(msg)
    combined = f"{content} {extra}".strip() if extra else content

    if len(combined) > max_chars:
        omitted = len(combined) - max_chars
        combined = f"{combined[:max_chars].rstrip()}... [+{omitted} chars truncated]"
    return combined


def _is_meaningful_content(text: str) -> bool:
    if not text:
        return False
    lowered = text.strip().lower()
    if lowered in _MEANINGFUL_SHORT_WORDS:
        return True
    if "[" in text and "]" in text:
        return True
    # Strip emoji shortcodes :xxx: and check if any real word characters remain
    without_shortcodes = re.sub(r":[a-zA-Z0-9_]+:", "", text).strip()
    if not without_shortcodes:
        return False
    return bool(_WORD_CHAR_RE.search(without_shortcodes))


def _is_system_noise(msg: Any) -> bool:
    msg_type = getattr(msg, "type", None)
    if msg_type is None:
        return False
    name = getattr(msg_type, "name", str(msg_type)).lower()
    # Keep normal messages, replies, and thread starter messages
    return name not in {"default", "reply", "thread_starter_message", "chat_input_command"}


def _msg_timestamp(msg: Any) -> datetime:
    ts = getattr(msg, "created_at", None)
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc)


def _format_gap_label(delta_seconds: float, ts: datetime) -> str:
    mins = int(delta_seconds // 60)
    hours, rem_mins = divmod(mins, 60)
    if hours >= 24:
        days, rem_hours = divmod(hours, 24)
        gap_str = f"{days}d {rem_hours}h"
    elif hours > 0:
        gap_str = f"{hours}h{rem_mins:02d}m" if rem_mins else f"{hours}h"
    else:
        gap_str = f"{mins}m"
    return f"=== [Gap {gap_str} | {ts.strftime('%m-%d %H:%M')}] ==="


def _format_time_range(messages: Sequence[Any]) -> str:
    if not messages:
        return ""
    first_ts = _msg_timestamp(messages[0])
    last_ts = _msg_timestamp(messages[-1])
    if first_ts.date() == last_ts.date():
        return f"{first_ts.strftime('%H:%M')} - {last_ts.strftime('%H:%M')} UTC"
    return f"{first_ts.strftime('%d/%m %H:%M')} - {last_ts.strftime('%d/%m %H:%M')} UTC"


def extract_requester_mentions(
    messages: Sequence[Any],
    *,
    requester_id: int | None,
    speaker_map: dict[int, str],
    msg_id_to_ref: dict[int, int],
    max_hits: int = 8,
) -> tuple[MentionHit, ...]:
    """Deterministically detect messages mentioning or replying to the requesting user."""
    if not requester_id:
        return ()

    hits: list[MentionHit] = []
    for msg in messages:
        author = getattr(msg, "author", None)
        author_id = int(getattr(author, "id", 0)) if author else 0
        if author_id == requester_id or getattr(author, "bot", False):
            continue

        mentioned_ids = {int(getattr(u, "id", 0)) for u in (getattr(msg, "mentions", ()) or ())}
        is_mentioned = requester_id in mentioned_ids

        is_replied_to_requester = False
        ref = getattr(msg, "reference", None)
        resolved = getattr(ref, "resolved", None) if ref is not None else None
        if resolved is not None:
            parent_author = getattr(resolved, "author", None)
            if parent_author is not None and int(getattr(parent_author, "id", 0)) == requester_id:
                is_replied_to_requester = True

        if not (is_mentioned or is_replied_to_requester):
            continue

        mid = int(getattr(msg, "id", 0))
        ref_id = msg_id_to_ref.get(mid, 0)
        author_name = speaker_map.get(author_id, _get_author_display_and_username(author)[1])
        snippet = _clean_message_text(msg, speaker_map, max_chars=120)
        if not snippet:
            continue
        ts = _msg_timestamp(msg)
        jump_url = str(getattr(msg, "jump_url", "") or "")
        hits.append(
            MentionHit(
                ref_id=ref_id,
                message_id=mid,
                author_name=author_name,
                timestamp_hhmm=ts.strftime("%H:%M"),
                snippet=snippet,
                jump_url=jump_url,
            )
        )

    return tuple(hits[-max_hits:])


def preprocess_messages(
    messages: Sequence[Any],
    *,
    requester_id: int | None = None,
    max_single_message_chars: int = 800,
    coalesce_window_seconds: int = 120,
    session_gap_minutes: int = 30,
    skip_bots: bool = True,
) -> PreprocessedConversation:
    """Clean, enrich, coalesce, and segment Discord messages for high-accuracy LLM summarization."""
    if not messages:
        return PreprocessedConversation(
            turns=(),
            ref_to_jump_url={},
            valid_ref_ids=set(),
            participants=(),
            participants_summary="",
            requester_mentions=(),
            raw_message_count=0,
            kept_message_count=0,
            time_range_label="",
        )

    speaker_map = _build_unique_speaker_map(messages)
    time_range_label = _format_time_range(messages)

    # First pass: filter & clean individual messages, tracking participant counts and msg_id lookup
    cleaned_items: list[dict[str, Any]] = []
    participant_counts: Counter[int] = Counter()

    for msg in messages:
        author = getattr(msg, "author", None)
        if author is None:
            continue
        if skip_bots and getattr(author, "bot", False):
            continue
        if _is_system_noise(msg):
            continue

        text = _clean_message_text(msg, speaker_map, max_chars=max_single_message_chars)
        if not _is_meaningful_content(text):
            continue

        author_id = int(getattr(author, "id", 0))
        author_name = speaker_map.get(author_id, _get_author_display_and_username(author)[1])
        participant_counts[author_id] += 1

        ref = getattr(msg, "reference", None)
        reply_to_msg_id = int(getattr(ref, "message_id", 0) or 0) if ref is not None else 0
        resolved_parent = getattr(ref, "resolved", None) if ref is not None else None

        cleaned_items.append(
            {
                "msg_id": int(getattr(msg, "id", 0)),
                "author_id": author_id,
                "author_name": author_name,
                "ts": _msg_timestamp(msg),
                "text": text,
                "jump_url": str(getattr(msg, "jump_url", "") or ""),
                "reply_to_msg_id": reply_to_msg_id,
                "resolved_parent": resolved_parent,
            }
        )

    # Second pass: assign ref_ids, insert session breaks, enrich replies, and coalesce same-speaker bursts
    turns: list[TranscriptTurn] = []
    ref_to_jump_url: dict[int, str] = {}
    msg_id_to_meta: dict[int, tuple[int, str, str]] = {}  # msg_id -> (ref_id, author_name, snippet)
    msg_id_to_ref: dict[int, int] = {}

    next_ref_id = 1
    prev_ts: datetime | None = None

    # Track currently open coalesced speaker turn
    active: dict[str, Any] | None = None

    def _flush_active() -> None:
        nonlocal active
        if active is None:
            return
        reply_tag = active["reply_tag"]
        header = f"[#{active['ref_id']} {active['ts'].strftime('%H:%M')}] {active['author_name']}{reply_tag}:"
        line = f"{header} {active['text']}"
        turns.append(
            TranscriptTurn(
                ref_id=active["ref_id"],
                text_line=line,
                raw_msg_count=active["raw_count"],
                is_session_break=False,
            )
        )
        active = None

    for item in cleaned_items:
        ts: datetime = item["ts"]

        # Detect long silence -> emit a natural topic/session boundary
        if prev_ts is not None:
            gap_sec = (ts - prev_ts).total_seconds()
            if gap_sec >= session_gap_minutes * 60:
                _flush_active()
                turns.append(
                    TranscriptTurn(
                        ref_id=0,
                        text_line=_format_gap_label(gap_sec, ts),
                        raw_msg_count=0,
                        is_session_break=True,
                    )
                )

        # Build reply-graph tag if this message replies to another message
        reply_tag = ""
        parent_id = item["reply_to_msg_id"]
        if parent_id:
            if parent_id in msg_id_to_meta:
                p_ref, p_author, p_snip = msg_id_to_meta[parent_id]
                reply_tag = f' ->({p_author} #{p_ref}: "{p_snip}")'
            elif item["resolved_parent"] is not None:
                p_obj = item["resolved_parent"]
                p_auth_obj = getattr(p_obj, "author", None)
                if p_auth_obj is not None:
                    p_uid = int(getattr(p_auth_obj, "id", 0))
                    p_author = speaker_map.get(
                        p_uid, _get_author_display_and_username(p_auth_obj)[1]
                    )
                    p_snip = _clean_message_text(p_obj, speaker_map, max_chars=45)
                    reply_tag = f' ->({p_author}: "{p_snip}")'

        # Coalesce with active turn ONLY if same author, within window, no new reply target, and fits cap
        can_coalesce = (
            active is not None
            and active["author_id"] == item["author_id"]
            and not reply_tag
            and (ts - active["last_ts"]).total_seconds() <= coalesce_window_seconds
            and len(active["text"]) + len(item["text"]) + 3 <= max_single_message_chars
        )

        if can_coalesce and active is not None:
            sep = " " if active["text"].endswith((".", "!", "?", ";", ",")) else " | "
            active["text"] = f"{active['text']}{sep}{item['text']}"
            active["last_ts"] = ts
            active["raw_count"] += 1
            msg_id_to_ref[item["msg_id"]] = active["ref_id"]
            msg_id_to_meta[item["msg_id"]] = (
                active["ref_id"],
                item["author_name"],
                item["text"][:45],
            )
        else:
            _flush_active()
            ref_id = next_ref_id
            next_ref_id += 1
            ref_to_jump_url[ref_id] = item["jump_url"]
            msg_id_to_ref[item["msg_id"]] = ref_id
            msg_id_to_meta[item["msg_id"]] = (ref_id, item["author_name"], item["text"][:45])
            active = {
                "ref_id": ref_id,
                "author_id": item["author_id"],
                "author_name": item["author_name"],
                "ts": ts,
                "last_ts": ts,
                "reply_tag": reply_tag,
                "text": item["text"],
                "raw_count": 1,
            }

        prev_ts = ts

    _flush_active()

    # Participant statistics sorted by activity
    participants = tuple(
        ParticipantStat(
            user_id=uid,
            display_name=speaker_map.get(uid, f"User{uid}"),
            message_count=cnt,
        )
        for uid, cnt in participant_counts.most_common()
    )
    participants_summary = ", ".join(
        f"{p.display_name} ({p.message_count} msgs)" for p in participants[:25]
    )
    if len(participants) > 25:
        participants_summary += f", +{len(participants) - 25} others"

    requester_mentions = extract_requester_mentions(
        messages,
        requester_id=requester_id,
        speaker_map=speaker_map,
        msg_id_to_ref=msg_id_to_ref,
    )

    valid_ref_ids = {turn.ref_id for turn in turns if turn.ref_id > 0}
    return PreprocessedConversation(
        turns=tuple(turns),
        ref_to_jump_url=ref_to_jump_url,
        valid_ref_ids=valid_ref_ids,
        participants=participants,
        participants_summary=participants_summary,
        requester_mentions=requester_mentions,
        raw_message_count=len(messages),
        kept_message_count=len(cleaned_items),
        time_range_label=time_range_label,
    )

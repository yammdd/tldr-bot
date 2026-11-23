"""Token counting (Vietnamese/CJK-aware) and session-aware context chunking."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal, Sequence


@dataclass(frozen=True)
class TranscriptTurn:
    """A single preprocessed unit (coalesced speaker turn or session separator)."""

    ref_id: int  # 0 for session break markers, >= 1 for speaker turns
    text_line: str
    raw_msg_count: int = 1
    is_session_break: bool = False


@dataclass(frozen=True)
class TranscriptChunk:
    chunk_index: int
    turns: tuple[TranscriptTurn, ...]
    estimated_tokens: int

    def render_text(self) -> str:
        return "\n".join(turn.text_line for turn in self.turns)


@dataclass(frozen=True)
class BudgetPlan:
    mode: Literal["single_pass", "map_reduce"]
    chunks: tuple[TranscriptChunk, ...]
    total_estimated_tokens: int
    kept_turns: int
    total_turns: int
    kept_raw_messages: int
    total_raw_messages: int
    was_token_capped: bool


@lru_cache(maxsize=1)
def _load_tiktoken_encoding():
    """Load tiktoken encoding if cached/available; fall back cleanly if offline."""
    try:
        import tiktoken

        try:
            return tiktoken.get_encoding("o200k_base")
        except Exception:
            return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def _heuristic_token_count(text: str) -> int:
    """Offline fallback token estimator calibrated for Latin, Vietnamese diacritics, and CJK."""
    if not text:
        return 0
    ascii_chars = 0
    non_ascii_chars = 0
    for ch in text:
        if ord(ch) < 128:
            ascii_chars += 1
        else:
            non_ascii_chars += 1
    # English/ASCII ~ 3.6 chars/token; Vietnamese diacritics / CJK ~ 1.4 chars/token
    return max(1, math.ceil(ascii_chars / 3.6 + non_ascii_chars / 1.4))


def estimate_tokens(text: str, *, safety_multiplier: float = 1.15) -> int:
    """Estimate token count conservatively across OpenAI, DeepSeek, Gemini, and Claude tokenizers."""
    if not text:
        return 0
    enc = _load_tiktoken_encoding()
    if enc is not None:
        base = len(enc.encode(text, disallowed_special=()))
        return max(1, math.ceil(base * safety_multiplier))
    return max(1, math.ceil(_heuristic_token_count(text) * safety_multiplier))


def plan_context_budget(
    turns: Sequence[TranscriptTurn],
    *,
    single_pass_limit: int,
    chunk_target_tokens: int,
    max_total_input_tokens: int,
    overlap_turns: int = 5,
    max_chunks: int = 6,
) -> BudgetPlan:
    """Select turns within max_total_input_tokens and decide single-pass vs session-aware Map-Reduce."""
    if not turns:
        return BudgetPlan(
            mode="single_pass",
            chunks=(),
            total_estimated_tokens=0,
            kept_turns=0,
            total_turns=0,
            kept_raw_messages=0,
            total_raw_messages=0,
            was_token_capped=False,
        )

    turn_tokens = [estimate_tokens(t.text_line) + 1 for t in turns]
    total_raw = sum(t.raw_msg_count for t in turns)

    # Step 1: Enforce hard token ceiling by keeping the MOST RECENT turns
    effective_cap = max(single_pass_limit, max_total_input_tokens)
    kept_indices: list[int] = []
    running_tokens = 0
    for idx in range(len(turns) - 1, -1, -1):
        tok = turn_tokens[idx]
        if running_tokens + tok > effective_cap and kept_indices:
            break
        kept_indices.append(idx)
        running_tokens += tok

    kept_indices.reverse()
    # Drop leading session break marker if truncation started on one
    while len(kept_indices) > 1 and turns[kept_indices[0]].is_session_break:
        removed_idx = kept_indices.pop(0)
        running_tokens -= turn_tokens[removed_idx]

    selected_turns = [turns[i] for i in kept_indices]
    selected_tokens = [turn_tokens[i] for i in kept_indices]
    was_capped = len(selected_turns) < len(turns)
    kept_raw = sum(t.raw_msg_count for t in selected_turns)

    # Step 2: Single-pass if within optimal attention threshold
    if running_tokens <= single_pass_limit:
        chunk = TranscriptChunk(
            chunk_index=0,
            turns=tuple(selected_turns),
            estimated_tokens=running_tokens,
        )
        return BudgetPlan(
            mode="single_pass",
            chunks=(chunk,),
            total_estimated_tokens=running_tokens,
            kept_turns=len(selected_turns),
            total_turns=len(turns),
            kept_raw_messages=kept_raw,
            total_raw_messages=total_raw,
            was_token_capped=was_capped,
        )

    # Step 3: Session-aware chunking for Map-Reduce (avoiding Lost-in-the-Middle)
    chunks: list[TranscriptChunk] = []
    n = len(selected_turns)
    cursor = 0
    min_chunk_tokens = max(1_500, int(chunk_target_tokens * 0.55))

    while cursor < n:
        acc_tokens = 0
        end = cursor
        last_break_end: int | None = None

        while end < n:
            next_tok = selected_tokens[end]
            if acc_tokens + next_tok > chunk_target_tokens and end > cursor:
                break
            acc_tokens += next_tok
            if selected_turns[end].is_session_break and acc_tokens >= min_chunk_tokens:
                last_break_end = end + 1
            end += 1

        # Prefer splitting cleanly on a natural time gap (session break)
        split_end = last_break_end if (last_break_end is not None and end < n) else end
        if split_end <= cursor:
            split_end = min(n, cursor + 1)

        slice_turns = selected_turns[cursor:split_end]
        slice_tokens = sum(selected_tokens[cursor:split_end])
        chunks.append(
            TranscriptChunk(
                chunk_index=len(chunks),
                turns=tuple(slice_turns),
                estimated_tokens=slice_tokens,
            )
        )

        if split_end >= n:
            break

        # Carry a short overlap (non-break turns) so cross-boundary replies keep context
        overlap_start = max(cursor + 1, split_end - overlap_turns)
        cursor = overlap_start

    # Safety clamp if chunk count somehow exceeded max_chunks (keep most recent chunks)
    if len(chunks) > max_chunks:
        chunks = chunks[-max_chunks:]
        chunks = [
            TranscriptChunk(chunk_index=i, turns=c.turns, estimated_tokens=c.estimated_tokens)
            for i, c in enumerate(chunks)
        ]
        was_capped = True
        kept_turns_set = sum(len(c.turns) for c in chunks)
        running_tokens = sum(c.estimated_tokens for c in chunks)
    else:
        kept_turns_set = len(selected_turns)

    return BudgetPlan(
        mode="map_reduce",
        chunks=tuple(chunks),
        total_estimated_tokens=running_tokens,
        kept_turns=kept_turns_set,
        total_turns=len(turns),
        kept_raw_messages=kept_raw,
        total_raw_messages=total_raw,
        was_token_capped=was_capped,
    )

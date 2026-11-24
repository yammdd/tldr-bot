"""Orchestrates single-pass and parallel Map-Reduce summarization with faithfulness post-validation."""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal, Sequence

from tldr_bot.llm.client import UnifiedLLMClient
from tldr_bot.llm.presets import compute_context_thresholds
from tldr_bot.llm.prompts import (
    build_map_chunk_messages,
    build_reduce_messages,
    build_single_pass_messages,
)
from tldr_bot.llm.token_budget import TranscriptTurn, plan_context_budget
from tldr_bot.security.vault import GuildLLMConfig
from tldr_bot.settings import Settings

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_MASS_MENTION_RE = re.compile(r"@(everyone|here)\b", re.IGNORECASE)


@dataclass(frozen=True)
class TopicItem:
    title: str
    speakers: tuple[str, ...]
    summary: str
    status: str  # decided | open | resolved | info
    ref_ids: tuple[int, ...]


@dataclass(frozen=True)
class PersonSummaryItem:
    speaker: str
    summary: str
    ref_ids: tuple[int, ...]


@dataclass(frozen=True)
class ActionItem:
    owner: str
    task: str
    ref_id: int | None


@dataclass(frozen=True)
class HighlightItem:
    label: str
    shared_by: str
    ref_id: int | None


@dataclass(frozen=True)
class SummaryResult:
    overview: str
    topics: tuple[TopicItem, ...] = ()
    by_person: tuple[PersonSummaryItem, ...] = ()
    decisions: tuple[str, ...] = ()
    action_items: tuple[ActionItem, ...] = ()
    open_questions: tuple[str, ...] = ()
    highlights: tuple[HighlightItem, ...] = ()
    raw_fallback_markdown: str | None = None

    strategy: Literal["single_pass", "map_reduce"] = "single_pass"
    chunks_used: int = 1
    input_tokens_estimated: int = 0
    api_prompt_tokens: int = 0
    api_completion_tokens: int = 0
    elapsed_seconds: float = 0.0
    was_token_capped: bool = False
    kept_raw_messages: int = 0
    total_raw_messages: int = 0
    model_used: str = ""
    extra_metadata: dict[str, Any] = field(default_factory=dict)


def _neutralize_mentions(text: str) -> str:
    return _MASS_MENTION_RE.sub(lambda m: f"@\u200b{m.group(1)}", text)


def _strip_think_tags(text: str) -> str:
    return _THINK_BLOCK_RE.sub("", text).strip()


def extract_json_object(raw_text: str) -> dict[str, Any] | None:
    """Robustly extract a JSON object from model output (handles <think> blocks and code fences)."""
    cleaned = _strip_think_tags(raw_text)
    if not cleaned:
        return None

    for candidate in (cleaned,):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    fence_match = _CODE_FENCE_RE.search(cleaned)
    if fence_match:
        try:
            parsed = json.loads(fence_match.group(1).strip())
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(cleaned[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    return None


def _clean_ref_ids(raw_ids: Any, valid_ref_ids: set[int], *, limit: int = 3) -> tuple[int, ...]:
    if not isinstance(raw_ids, list):
        return ()
    cleaned: list[int] = []
    seen: set[int] = set()
    for item in raw_ids:
        try:
            val = int(str(item).lstrip("#"))
        except (ValueError, TypeError):
            continue
        if val in valid_ref_ids and val not in seen:
            seen.add(val)
            cleaned.append(val)
            if len(cleaned) >= limit:
                break
    return tuple(cleaned)


def _clean_single_ref(raw_id: Any, valid_ref_ids: set[int]) -> int | None:
    if raw_id is None:
        return None
    try:
        val = int(str(raw_id).lstrip("#"))
    except (ValueError, TypeError):
        return None
    return val if val in valid_ref_ids else None


def _clean_str(value: Any, *, max_len: int = 1200) -> str:
    if not isinstance(value, str):
        return ""
    return _neutralize_mentions(value.strip())[:max_len]


def parse_and_validate_summary(
    raw_content: str,
    *,
    valid_ref_ids: set[int],
    strategy: Literal["single_pass", "map_reduce"],
    chunks_used: int,
    input_tokens_estimated: int,
    api_prompt_tokens: int,
    api_completion_tokens: int,
    elapsed_seconds: float,
    was_token_capped: bool,
    kept_raw_messages: int,
    total_raw_messages: int,
    model_used: str,
) -> SummaryResult:
    """Parse structured JSON output and validate every citation ID against real transcript turns."""
    data = extract_json_object(raw_content)
    if data is None:
        fallback = _neutralize_mentions(_strip_think_tags(raw_content))
        return SummaryResult(
            overview="",
            raw_fallback_markdown=fallback[:3800],
            strategy=strategy,
            chunks_used=chunks_used,
            input_tokens_estimated=input_tokens_estimated,
            api_prompt_tokens=api_prompt_tokens,
            api_completion_tokens=api_completion_tokens,
            elapsed_seconds=elapsed_seconds,
            was_token_capped=was_token_capped,
            kept_raw_messages=kept_raw_messages,
            total_raw_messages=total_raw_messages,
            model_used=model_used,
        )

    overview = _clean_str(data.get("overview"), max_len=400)

    topics: list[TopicItem] = []
    for raw_topic in data.get("topics") or []:
        if not isinstance(raw_topic, dict):
            continue
        title = _clean_str(raw_topic.get("title"), max_len=120)
        summary = _clean_str(raw_topic.get("summary"), max_len=900)
        if not title and not summary:
            continue
        speakers = tuple(
            _clean_str(s, max_len=60)
            for s in (raw_topic.get("speakers") or [])
            if isinstance(s, str) and s.strip()
        )[:10]
        raw_status = str(raw_topic.get("status") or "info").strip().lower()
        status = raw_status if raw_status in {"decided", "open", "resolved", "info"} else "info"
        ref_ids = _clean_ref_ids(raw_topic.get("ref_ids"), valid_ref_ids, limit=3)
        topics.append(
            TopicItem(
                title=title or "Topic",
                speakers=speakers,
                summary=summary,
                status=status,
                ref_ids=ref_ids,
            )
        )

    by_person: list[PersonSummaryItem] = []
    for raw_person in data.get("by_person") or []:
        if not isinstance(raw_person, dict):
            continue
        speaker = _clean_str(raw_person.get("speaker"), max_len=80)
        summary = _clean_str(raw_person.get("summary"), max_len=500)
        if not speaker or not summary:
            continue
        ref_ids = _clean_ref_ids(raw_person.get("ref_ids"), valid_ref_ids, limit=2)
        by_person.append(PersonSummaryItem(speaker=speaker, summary=summary, ref_ids=ref_ids))

    decisions = tuple(
        _clean_str(d, max_len=350)
        for d in (data.get("decisions") or [])
        if isinstance(d, str) and d.strip()
    )[:10]

    action_items: list[ActionItem] = []
    for raw_action in data.get("action_items") or []:
        if not isinstance(raw_action, dict):
            continue
        owner = _clean_str(raw_action.get("owner"), max_len=80)
        task = _clean_str(raw_action.get("task"), max_len=350)
        if not task:
            continue
        ref_id = _clean_single_ref(raw_action.get("ref_id"), valid_ref_ids)
        action_items.append(ActionItem(owner=owner, task=task, ref_id=ref_id))

    open_questions = tuple(
        _clean_str(q, max_len=350)
        for q in (data.get("open_questions") or [])
        if isinstance(q, str) and q.strip()
    )[:8]

    highlights: list[HighlightItem] = []
    for raw_hl in data.get("highlights") or []:
        if not isinstance(raw_hl, dict):
            continue
        label = _clean_str(raw_hl.get("label"), max_len=240)
        shared_by = _clean_str(raw_hl.get("shared_by"), max_len=80)
        if not label:
            continue
        ref_id = _clean_single_ref(raw_hl.get("ref_id"), valid_ref_ids)
        highlights.append(HighlightItem(label=label, shared_by=shared_by, ref_id=ref_id))

    return SummaryResult(
        overview=overview,
        topics=tuple(topics[:10]),
        by_person=tuple(by_person[:12]),
        decisions=decisions,
        action_items=tuple(action_items[:10]),
        open_questions=open_questions,
        highlights=tuple(highlights[:8]),
        raw_fallback_markdown=None,
        strategy=strategy,
        chunks_used=chunks_used,
        input_tokens_estimated=input_tokens_estimated,
        api_prompt_tokens=api_prompt_tokens,
        api_completion_tokens=api_completion_tokens,
        elapsed_seconds=elapsed_seconds,
        was_token_capped=was_token_capped,
        kept_raw_messages=kept_raw_messages,
        total_raw_messages=total_raw_messages,
        model_used=model_used,
    )


ProgressCallback = Callable[[str, int, int], Awaitable[None]]


class ConversationSummarizer:
    """Runs single-pass or session-aware parallel Map-Reduce summarization."""

    def __init__(self, llm_client: UnifiedLLMClient, settings: Settings) -> None:
        self._llm = llm_client
        self._settings = settings

    async def summarize(
        self,
        *,
        creds: GuildLLMConfig,
        channel_name: str,
        time_range: str,
        participants_summary: str,
        turns: Sequence[TranscriptTurn],
        valid_ref_ids: set[int],
        language_code: str,
        style_id: str = "standard",
        focus_query: str | None = None,
        on_stage_update: ProgressCallback | None = None,
    ) -> SummaryResult:
        t0 = time.perf_counter()
        thresholds = compute_context_thresholds(creds.model, self._settings)
        plan = plan_context_budget(
            turns,
            single_pass_limit=thresholds.single_pass_limit,
            chunk_target_tokens=thresholds.chunk_target_tokens,
            max_total_input_tokens=thresholds.max_total_input_tokens,
            overlap_turns=5,
        )

        if not plan.chunks:
            return SummaryResult(
                overview="",
                elapsed_seconds=time.perf_counter() - t0,
                model_used=creds.model,
            )

        total_prompt_tokens = 0
        total_completion_tokens = 0

        if plan.mode == "single_pass":
            if on_stage_update:
                await on_stage_update("single_pass", plan.total_estimated_tokens, 1)

            messages = build_single_pass_messages(
                channel_name=channel_name,
                time_range=time_range,
                participants_summary=participants_summary,
                transcript_text=plan.chunks[0].render_text(),
                language_code=language_code,
                style_id=style_id,
                focus_query=focus_query,
            )
            resp = await self._llm.complete_chat(
                creds,
                messages,
                max_tokens=4096,
                temperature=0.35 if style_id in {"casual", "fun"} else 0.15,
                json_mode=True,
            )
            total_prompt_tokens = resp.prompt_tokens
            total_completion_tokens = resp.completion_tokens
            final_raw = resp.content
        else:
            if on_stage_update:
                await on_stage_update(
                    "map_reduce", plan.total_estimated_tokens, len(plan.chunks)
                )

            sem = asyncio.Semaphore(self._settings.max_parallel_map_calls)

            async def _run_map(idx: int, chunk_text: str):
                async with sem:
                    map_msgs = build_map_chunk_messages(
                        chunk_index=idx,
                        total_chunks=len(plan.chunks),
                        channel_name=channel_name,
                        participants_summary=participants_summary,
                        chunk_text=chunk_text,
                        language_code=language_code,
                        focus_query=focus_query,
                    )
                    return await self._llm.complete_chat(
                        creds,
                        map_msgs,
                        max_tokens=3000,
                        temperature=0.1,
                        json_mode=True,
                    )

            map_responses = await asyncio.gather(
                *(_run_map(c.chunk_index, c.render_text()) for c in plan.chunks)
            )
            extracted_segments: list[Any] = []
            for idx, r in enumerate(map_responses):
                total_prompt_tokens += r.prompt_tokens
                total_completion_tokens += r.completion_tokens
                parsed_seg = extract_json_object(r.content)
                if parsed_seg is not None:
                    extracted_segments.append(parsed_seg)
                else:
                    extracted_segments.append(
                        {"segment_index": idx + 1, "raw_notes": _strip_think_tags(r.content)}
                    )

            reduce_msgs = build_reduce_messages(
                channel_name=channel_name,
                time_range=time_range,
                participants_summary=participants_summary,
                segment_summaries_json=json.dumps(extracted_segments, ensure_ascii=False),
                language_code=language_code,
                style_id=style_id,
                focus_query=focus_query,
            )
            reduce_resp = await self._llm.complete_chat(
                creds,
                reduce_msgs,
                max_tokens=4096,
                temperature=0.35 if style_id in {"casual", "fun"} else 0.15,
                json_mode=True,
            )
            total_prompt_tokens += reduce_resp.prompt_tokens
            total_completion_tokens += reduce_resp.completion_tokens
            final_raw = reduce_resp.content

        elapsed = time.perf_counter() - t0
        return parse_and_validate_summary(
            final_raw,
            valid_ref_ids=valid_ref_ids,
            strategy=plan.mode,
            chunks_used=len(plan.chunks),
            input_tokens_estimated=plan.total_estimated_tokens,
            api_prompt_tokens=total_prompt_tokens,
            api_completion_tokens=total_completion_tokens,
            elapsed_seconds=elapsed,
            was_token_capped=plan.was_token_capped,
            kept_raw_messages=plan.kept_raw_messages,
            total_raw_messages=plan.total_raw_messages,
            model_used=creds.model,
        )

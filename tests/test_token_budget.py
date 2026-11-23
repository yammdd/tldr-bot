"""Tests for token estimation and session-aware context chunking."""

from __future__ import annotations

from tldr_bot.llm.presets import compute_context_thresholds
from tldr_bot.llm.token_budget import TranscriptTurn, estimate_tokens, plan_context_budget
from tldr_bot.settings import Settings


def test_token_estimation_and_single_pass_plan() -> None:
    turns = [
        TranscriptTurn(ref_id=1, text_line="[#1 14:00] Alex: Should we ship the release tonight?", raw_msg_count=1),
        TranscriptTurn(ref_id=2, text_line="[#2 14:01] Maya: Yes, staging checks all passed.", raw_msg_count=1),
    ]
    assert estimate_tokens(turns[0].text_line) >= 10

    plan = plan_context_budget(
        turns,
        single_pass_limit=10_000,
        chunk_target_tokens=6_000,
        max_total_input_tokens=50_000,
    )
    assert plan.mode == "single_pass"
    assert len(plan.chunks) == 1
    assert plan.was_token_capped is False
    assert plan.kept_raw_messages == 2


def test_map_reduce_splits_on_session_breaks_with_overlap_and_caps_total_tokens() -> None:
    turns: list[TranscriptTurn] = []
    ref = 1
    for session in range(6):
        if session > 0:
            turns.append(
                TranscriptTurn(
                    ref_id=0,
                    text_line=f"=== [Gap 2h | 11-23 {10 + session}:00] ===",
                    raw_msg_count=0,
                    is_session_break=True,
                )
            )
        for j in range(15):
            turns.append(
                TranscriptTurn(
                    ref_id=ref,
                    text_line=(
                        f"[#{ref} 14:{j:02d}] Alex: Detailed engineering discussion for deployment milestone "
                        f"item {session}-{j} covering rollout checklist, metrics, and rollback criteria."
                    ),
                    raw_msg_count=1,
                )
            )
            ref += 1

    plan = plan_context_budget(
        turns,
        single_pass_limit=900,
        chunk_target_tokens=700,
        max_total_input_tokens=20_000,
        overlap_turns=3,
        max_chunks=10,
    )
    assert plan.mode == "map_reduce"
    assert len(plan.chunks) >= 2
    first_refs = {t.ref_id for t in plan.chunks[0].turns if t.ref_id > 0}
    second_refs = {t.ref_id for t in plan.chunks[1].turns if t.ref_id > 0}
    assert len(first_refs & second_refs) >= 1

    capped_plan = plan_context_budget(
        turns,
        single_pass_limit=500,
        chunk_target_tokens=400,
        max_total_input_tokens=800,
    )
    assert capped_plan.was_token_capped is True
    last_turn = capped_plan.chunks[-1].turns[-1]
    assert last_turn.ref_id == ref - 1


def test_context_thresholds_adapt_to_small_and_large_models() -> None:
    settings = Settings(discord_token="x")
    small = compute_context_thresholds("gpt-3.5-turbo", settings)
    large = compute_context_thresholds("gemini-3.8-flash", settings)
    assert small.single_pass_limit < large.single_pass_limit
    assert small.single_pass_limit <= 8_000
    assert large.single_pass_limit == settings.single_pass_token_threshold

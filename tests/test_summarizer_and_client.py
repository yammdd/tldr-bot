"""Tests for UnifiedLLMClient smoke testing, error mapping, Summarizer, Embed rendering & Bot command tree."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import openai
import pytest

from tldr_bot.discord_ext.preprocessor import (
    MentionHit,
    ParticipantStat,
    PreprocessedConversation,
)
from tldr_bot.discord_ext.ui_summary import build_summary_embeds
from tldr_bot.llm.client import (
    LLMError,
    LLMResponse,
    UnifiedLLMClient,
    filter_and_rank_chat_models,
    map_openai_exception,
)
from tldr_bot.llm.summarizer import ConversationSummarizer, parse_and_validate_summary
from tldr_bot.llm.token_budget import TranscriptTurn
from tldr_bot.main import TldrBot
from tldr_bot.security.vault import GuildLLMConfig
from tldr_bot.settings import Settings


def _fake_completion(text: str, prompt_tokens: int = 50, completion_tokens: int = 20):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
        ),
    )


@pytest.mark.asyncio
async def test_live_smoke_test_success_and_reasoning_model_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    client = UnifiedLLMClient(allow_local_endpoints=False)

    async def _fake_validate(raw_url: str, *, allow_local: bool = False, resolve_dns: bool = True) -> str:
        from tldr_bot.security.ssrf import validate_endpoint_url
        return validate_endpoint_url(raw_url, allow_local=allow_local, resolve_dns=False)

    monkeypatch.setattr("tldr_bot.llm.client.validate_endpoint_url_async", _fake_validate)

    calls: list[dict] = []

    class _FakeSDKClient:
        def __init__(self) -> None:
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self._create)
            )
            self.closed = False

        async def _create(self, **kwargs):
            calls.append(kwargs)
            if "max_tokens" in kwargs:
                req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
                resp = httpx.Response(
                    400,
                    request=req,
                    json={"error": {"message": "Unsupported parameter: 'max_tokens'. Use 'max_completion_tokens'."}},
                )
                raise openai.BadRequestError("Unsupported parameter: max_tokens", response=resp, body=resp.json())
            return _fake_completion("OK")

        async def close(self) -> None:
            self.closed = True

    fake_sdk = _FakeSDKClient()
    monkeypatch.setattr(client, "_build_sdk_client", lambda **kw: fake_sdk)

    res = await client.verify_credentials(
        base_url="https://api.openai.com/v1/chat/completions",
        api_key="sk-test-secret-1234567890abcdef",
        model="o4-mini",
    )
    assert res.normalized_base_url == "https://api.openai.com/v1"
    assert res.reply_preview == "OK"
    assert fake_sdk.closed is True
    assert len(calls) == 2
    assert "max_completion_tokens" in calls[1]


@pytest.mark.asyncio
async def test_live_smoke_test_maps_401_and_redacts_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    client = UnifiedLLMClient(allow_local_endpoints=False)

    async def _fake_validate(raw_url: str, *, allow_local: bool = False, resolve_dns: bool = True) -> str:
        from tldr_bot.security.ssrf import validate_endpoint_url
        return validate_endpoint_url(raw_url, allow_local=allow_local, resolve_dns=False)

    monkeypatch.setattr("tldr_bot.llm.client.validate_endpoint_url_async", _fake_validate)

    secret = "sk-my-very-secret-key-9988776655"

    class _FailSDKClient:
        def __init__(self) -> None:
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self._create)
            )

        async def _create(self, **kwargs):
            req = httpx.Request("POST", "https://api.deepseek.com/chat/completions")
            resp = httpx.Response(401, request=req, json={"error": {"message": f"Invalid key {secret}"}})
            raise openai.AuthenticationError(f"Invalid key {secret}", response=resp, body=resp.json())

        async def close(self) -> None:
            pass

    monkeypatch.setattr(client, "_build_sdk_client", lambda **kw: _FailSDKClient())

    with pytest.raises(LLMError) as exc_info:
        await client.verify_credentials(
            base_url="https://api.deepseek.com",
            api_key=secret,
            model="deepseek-chat",
        )
    assert exc_info.value.code == "err_auth"
    assert secret not in exc_info.value.detail


def test_map_openai_exception_handles_google_list_error_body() -> None:
    req = httpx.Request("POST", "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions")
    list_body = [{"error": {"code": 404, "message": "Model deprecated. Use gemini-3.8-flash."}}]
    resp = httpx.Response(404, request=req, json=list_body)
    exc = openai.NotFoundError("Error code: 404", response=resp, body=list_body)
    mapped = map_openai_exception(exc, api_key="AQ.FakeTestKey1234567890123456", model="gemini-2.5-flash")
    assert mapped.code == "err_not_found"
    assert "gemini-3.8-flash" in mapped.detail


def test_filter_and_rank_chat_models() -> None:
    raw = [
        "models/text-embedding-004",
        "models/imagen-4.0-generate-001",
        "models/veo-3.0-generate",
        "models/gemini-3.8-flash",
        "models/gemini-flash-lite-latest",
        "models/gemini-3.1-pro-preview",
        "models/gemini-3.5-flash-lite",
        "models/gemini-3.8-flash",
    ]
    ranked = filter_and_rank_chat_models(raw, current_model="gemini-3.8-flash")
    assert "text-embedding-004" not in ranked
    assert "imagen-4.0-generate-001" not in ranked
    assert "veo-3.0-generate" not in ranked
    assert ranked[0] == "gemini-3.8-flash"
    assert ranked[1] == "gemini-flash-lite-latest"
    assert len(ranked) == 4


def test_parse_and_validate_summary_strips_think_hallucinated_refs_and_mass_mentions() -> None:
    payload = {
        "overview": "Team finalized dinner plans and @everyone noted the Redis hotfix.",
        "topics": [
            {
                "title": "Dinner plans",
                "speakers": ["Alex", "Maya"],
                "summary": "**Alex** suggested ramen at 7:30 PM; **Maya** confirmed.",
                "status": "decided",
                "ref_ids": [1, 9999, 2],
            }
        ],
        "action_items": [{"owner": "Alex", "task": "Reserve table @here", "ref_id": 2}],
        "open_questions": ["Whether **Sam** can join tonight."],
    }
    raw = f"<think>internal reasoning</think>\n```json\n{json.dumps(payload)}\n```"

    res = parse_and_validate_summary(
        raw,
        valid_ref_ids={1, 2},
        strategy="single_pass",
        chunks_used=1,
        input_tokens_estimated=500,
        api_prompt_tokens=480,
        api_completion_tokens=120,
        elapsed_seconds=1.2,
        was_token_capped=False,
        kept_raw_messages=2,
        total_raw_messages=2,
        model_used="deepseek-chat",
    )
    assert "@everyone" not in res.overview
    assert res.topics[0].ref_ids == (1, 2)
    assert "@here" not in res.action_items[0].task


@pytest.mark.asyncio
async def test_summarizer_runs_parallel_map_reduce_when_over_threshold() -> None:
    settings = Settings(
        discord_token="x",
        single_pass_token_threshold=150,
        map_chunk_target_tokens=120,
        max_total_input_tokens=5000,
        max_parallel_map_calls=2,
    )

    mock_llm = AsyncMock()
    call_count = 0

    async def _fake_complete(creds, messages, **kwargs):
        nonlocal call_count
        call_count += 1
        sys_prompt = messages[0]["content"]
        if "Stage 1 (Map)" in sys_prompt:
            seg_json = json.dumps(
                {
                    "segment_index": call_count,
                    "topics": [
                        {
                            "title": "Release roadmap",
                            "speakers": ["Alex", "Maya"],
                            "details": "**Alex** proposed the release window; **Maya** agreed.",
                            "status": "decided",
                            "ref_ids": [1],
                        }
                    ],
                    "decisions": ["**Alex** and **Maya** confirmed the release timeline."],
                }
            )
            return LLMResponse(content=seg_json, prompt_tokens=100, completion_tokens=50)
        final_json = json.dumps(
            {
                "overview": "**Alex** and **Maya** finalized the release plan.",
                "topics": [
                    {
                        "title": "Release plan",
                        "speakers": ["Alex", "Maya"],
                        "summary": "**Alex** proposed shipping next week and **Maya** confirmed.",
                        "status": "decided",
                        "ref_ids": [1, 10],
                    }
                ],
                "decisions": ["Ship release build next week."],
                "action_items": [{"owner": "Alex", "task": "Prepare release artifact", "ref_id": 10}],
            }
        )
        return LLMResponse(content=final_json, prompt_tokens=150, completion_tokens=80)

    mock_llm.complete_chat.side_effect = _fake_complete
    summarizer = ConversationSummarizer(mock_llm, settings)

    turns = [
        TranscriptTurn(
            ref_id=i,
            text_line=f"[#{i} 14:{i:02d}] Alex: Detailed milestone discussion item {i} regarding release readiness and checks.",
            raw_msg_count=1,
        )
        for i in range(1, 15)
    ]
    creds = GuildLLMConfig(
        provider_id="deepseek",
        provider_name="DeepSeek",
        base_url="https://api.deepseek.com",
        model="deepseek-chat",
        api_key="sk-fake",
        updated_at_iso="2026-11-23T00:00:00+00:00",
    )

    result = await summarizer.summarize(
        creds=creds,
        channel_name="general",
        time_range="14:00 - 14:15 UTC",
        participants_summary="Alex (10 msgs), Maya (4 msgs)",
        turns=turns,
        valid_ref_ids=set(range(1, 15)),
        language_code="en",
        style_id="standard",
    )

    assert result.strategy == "map_reduce"
    assert result.chunks_used >= 2
    assert call_count == result.chunks_used + 1
    assert result.topics[0].ref_ids == (1, 10)


def test_embed_builder_hides_personal_mentions_on_public_share() -> None:
    conv = PreprocessedConversation(
        turns=(TranscriptTurn(ref_id=1, text_line="[#1 14:00] Alex: Hi", raw_msg_count=1),),
        ref_to_jump_url={1: "https://discord.com/channels/1/2/101"},
        valid_ref_ids={1},
        participants=(ParticipantStat(user_id=10, display_name="Alex", message_count=5),),
        participants_summary="Alex (5 msgs)",
        requester_mentions=(
            MentionHit(
                ref_id=1,
                message_id=101,
                author_name="Maya",
                timestamp_hhmm="14:05",
                snippet="please share the slide deck",
                jump_url="https://discord.com/channels/1/2/101",
            ),
        ),
        raw_message_count=5,
        kept_message_count=5,
        time_range_label="14:00 - 14:10 UTC",
    )
    summary = parse_and_validate_summary(
        json.dumps(
            {
                "overview": "Team coordinated on the slide deck.",
                "topics": [
                    {
                        "title": "Slide deck prep",
                        "speakers": ["Alex", "Maya"],
                        "summary": "**Maya** asked for the slide deck.",
                        "status": "open",
                        "ref_ids": [1],
                    }
                ],
            }
        ),
        valid_ref_ids={1},
        strategy="single_pass",
        chunks_used=1,
        input_tokens_estimated=120,
        api_prompt_tokens=100,
        api_completion_tokens=40,
        elapsed_seconds=1.1,
        was_token_capped=False,
        kept_raw_messages=5,
        total_raw_messages=5,
        model_used="deepseek-chat",
    )

    private_embeds = build_summary_embeds(
        result=summary,
        conv=conv,
        channel_name="general",
        lang_code="en",
        style_id="standard",
        public_share=False,
    )
    public_embeds = build_summary_embeds(
        result=summary,
        conv=conv,
        channel_name="general",
        lang_code="en",
        style_id="standard",
        public_share=True,
        requester_display="Alex",
    )

    private_field_names = [f.name for f in private_embeds[0].fields]
    public_field_names = [f.name for f in public_embeds[0].fields]
    assert any("Mentions of you" in n for n in private_field_names)
    assert not any("Mentions of you" in n for n in public_field_names)
    assert private_embeds[0].color.to_rgb() == (200, 109, 68)
    assert public_embeds[0].color.to_rgb() == (43, 48, 56)


@pytest.mark.asyncio
async def test_bot_registers_all_slash_commands_and_context_menu(tmp_path) -> None:
    settings = Settings(
        discord_token="dummy",
        key_storage_mode="memory",
        db_file_path=tmp_path / "tldr_bot.db",
    )
    bot = TldrBot(settings)
    from tldr_bot.discord_ext.cogs.admin_cog import AdminCog
    from tldr_bot.discord_ext.cogs.tldr_cog import TldrCog

    await bot.add_cog(AdminCog(bot))
    await bot.add_cog(TldrCog(bot))

    cmd_names = {cmd.name for cmd in bot.tree.get_commands()}
    assert {
        "config",
        "model",
        "language",
        "status",
        "reset_config",
        "help",
        "tldr",
        "TL;DR from here",
    }.issubset(cmd_names)

    from tldr_bot.health import HealthServer

    hs = HealthServer(bot, port=0)
    resp_starting = await hs._handle_healthz(None)
    assert resp_starting.status == 503
    starting_payload = json.loads(resp_starting.text)
    assert starting_payload["ready"] is False
    assert starting_payload["status"] == "starting"

    bot.is_ready = lambda: True  # type: ignore[method-assign]
    resp_ready = await hs._handle_healthz(None)
    assert resp_ready.status == 200
    ready_payload = json.loads(resp_ready.text)
    assert ready_payload["ready"] is True
    assert ready_payload["status"] == "ok"

    await bot.close()

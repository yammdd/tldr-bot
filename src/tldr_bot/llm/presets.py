"""Built-in OpenAI-compatible LLM provider presets and model context window profiles."""

from __future__ import annotations

from dataclasses import dataclass

from tldr_bot.settings import Settings


@dataclass(frozen=True)
class ProviderPreset:
    id: str
    name: str
    description: str
    default_base_url: str
    default_model: str
    is_custom: bool = False


PROVIDER_PRESETS: tuple[ProviderPreset, ...] = (
    ProviderPreset(
        id="gemini",
        name="Google Gemini",
        description="1M context window, fast for 500 to 1,500 message threads",
        default_base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        default_model="gemini-3.8-flash",
    ),
    ProviderPreset(
        id="deepseek",
        name="DeepSeek",
        description="Low cost, strong Vietnamese and multilingual synthesis",
        default_base_url="https://api.deepseek.com",
        default_model="deepseek-chat",
    ),
    ProviderPreset(
        id="openai",
        name="OpenAI",
        description="Reliable structured JSON output (gpt-4o-mini / gpt-5-mini)",
        default_base_url="https://api.openai.com/v1",
        default_model="gpt-4o-mini",
    ),
    ProviderPreset(
        id="claude",
        name="Anthropic Claude",
        description="High faithfulness via official OpenAI-compatible endpoint",
        default_base_url="https://api.anthropic.com/v1/",
        default_model="claude-haiku-4-5",
    ),
    ProviderPreset(
        id="openrouter",
        name="OpenRouter",
        description="Single API key routing to Gemini, Claude, GPT, DeepSeek, Llama",
        default_base_url="https://openrouter.ai/api/v1",
        default_model="google/gemini-2.5-flash",
    ),
    ProviderPreset(
        id="groq",
        name="Groq",
        description="Low-latency inference with Llama 3.3",
        default_base_url="https://api.groq.com/openai/v1",
        default_model="llama-3.3-70b-versatile",
    ),
    ProviderPreset(
        id="custom",
        name="Custom Endpoint",
        description="Any OpenAI-compatible Base URL (LiteLLM, vLLM, gateway)",
        default_base_url="https://",
        default_model="gpt-4o-mini",
        is_custom=True,
    ),
)

PRESET_BY_ID: dict[str, ProviderPreset] = {p.id: p for p in PROVIDER_PRESETS}


def get_preset(provider_id: str) -> ProviderPreset:
    return PRESET_BY_ID.get(provider_id, PRESET_BY_ID["custom"])


_MODEL_CONTEXT_HINTS: tuple[tuple[str, int], ...] = (
    ("gemini", 1_000_000),
    ("claude", 200_000),
    ("gpt-5", 200_000),
    ("gpt-4.1", 1_000_000),
    ("gpt-4o", 128_000),
    ("o1", 128_000),
    ("o3", 200_000),
    ("o4", 200_000),
    ("deepseek", 64_000),
    ("llama-3.3", 128_000),
    ("llama-3.1", 128_000),
    ("qwen", 64_000),
    ("mistral-large", 128_000),
    ("gpt-4-turbo", 128_000),
    ("gpt-3.5", 16_000),
)


def estimate_model_context_window(model_name: str) -> int:
    """Return a conservative context window estimate for the specified model ID."""
    lowered = (model_name or "").strip().lower()
    for needle, ctx in _MODEL_CONTEXT_HINTS:
        if needle in lowered:
            return ctx
    return 32_000


@dataclass(frozen=True)
class ContextThresholds:
    model_context_window: int
    single_pass_limit: int
    chunk_target_tokens: int
    max_total_input_tokens: int


def compute_context_thresholds(model_name: str, settings: Settings) -> ContextThresholds:
    """Clamp single-pass and map-chunk token limits to both model capacity and attention sweet spot."""
    ctx_window = estimate_model_context_window(model_name)
    safe_per_call_cap = max(4_000, int(ctx_window * 0.65) - 4_000)

    single_pass_limit = min(settings.single_pass_token_threshold, safe_per_call_cap)
    chunk_target = min(
        settings.map_chunk_target_tokens, max(3_500, int(safe_per_call_cap * 0.75))
    )
    chunk_target = min(chunk_target, single_pass_limit)

    return ContextThresholds(
        model_context_window=ctx_window,
        single_pass_limit=single_pass_limit,
        chunk_target_tokens=chunk_target,
        max_total_input_tokens=settings.max_total_input_tokens,
    )

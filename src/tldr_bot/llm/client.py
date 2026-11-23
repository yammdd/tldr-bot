"""Unified Async OpenAI-compatible LLM client with live smoke testing and safe error mapping."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx
import openai
from openai import AsyncOpenAI

from tldr_bot.security.redactor import redact_secrets
from tldr_bot.security.ssrf import EndpointSecurityError, validate_endpoint_url_async
from tldr_bot.security.vault import GuildLLMConfig


class LLMError(Exception):
    """Structured, user-safe LLM error whose message never leaks secrets."""

    def __init__(
        self,
        code: str,
        detail: str = "",
        *,
        status_code: int | None = None,
        model: str = "",
    ) -> None:
        super().__init__(detail or code)
        self.code = code
        self.detail = detail
        self.status_code = status_code
        self.model = model


@dataclass(frozen=True)
class SmokeTestResult:
    normalized_base_url: str
    latency_ms: int
    reply_preview: str


@dataclass(frozen=True)
class LLMResponse:
    content: str
    prompt_tokens: int
    completion_tokens: int


def _extract_error_detail(exc: Exception, api_key: str) -> str:
    raw = ""
    body = getattr(exc, "body", None)
    # Google's OpenAI-compatible layer wraps errors in a single-element JSON list
    if isinstance(body, list) and body and isinstance(body[0], dict):
        body = body[0]
    if isinstance(body, dict):
        err_obj = body.get("error")
        if isinstance(err_obj, dict) and err_obj.get("message"):
            raw = str(err_obj["message"])
        elif body.get("message"):
            raw = str(body["message"])
    if not raw:
        raw = str(exc)
    redacted = redact_secrets(raw, (api_key,))
    return redacted[:280].strip()


def map_openai_exception(exc: Exception, *, api_key: str, model: str) -> LLMError:
    """Translate OpenAI SDK / HTTPX exceptions into localized, redacted LLMError codes."""
    detail = _extract_error_detail(exc, api_key)
    status = getattr(exc, "status_code", None)

    if isinstance(exc, EndpointSecurityError):
        return LLMError(exc.code, detail=str(exc), model=model)
    if isinstance(exc, openai.AuthenticationError) or status == 401:
        return LLMError("err_auth", detail=detail, status_code=401, model=model)
    if isinstance(exc, openai.PermissionDeniedError) or status == 403:
        return LLMError("err_forbidden", detail=detail, status_code=403, model=model)
    if isinstance(exc, openai.RateLimitError) or status in {402, 429}:
        return LLMError("err_quota", detail=detail, status_code=status or 429, model=model)
    if isinstance(exc, openai.NotFoundError) or status == 404:
        return LLMError("err_not_found", detail=detail, status_code=404, model=model)
    if isinstance(exc, openai.BadRequestError) or status == 400:
        return LLMError("err_bad_request", detail=detail, status_code=400, model=model)
    if isinstance(exc, openai.APITimeoutError) or isinstance(exc, httpx.TimeoutException):
        return LLMError("err_timeout", detail=detail, model=model)
    if isinstance(exc, openai.APIConnectionError) or isinstance(exc, httpx.ConnectError):
        return LLMError("err_connection", detail=detail, model=model)
    if isinstance(exc, openai.APIStatusError):
        if status and status >= 500:
            return LLMError("err_server", detail=detail, status_code=status, model=model)
        if status == 402:
            return LLMError("err_quota", detail=detail, status_code=402, model=model)
        return LLMError("err_unknown", detail=detail, status_code=status, model=model)

    return LLMError("err_unknown", detail=detail, status_code=status, model=model)


def _extract_choice_text(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if not choices:
        return ""
    msg = getattr(choices[0], "message", None)
    if msg is None:
        return ""
    content = getattr(msg, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"]))
            elif hasattr(item, "text") and item.text:
                parts.append(str(item.text))
        return "".join(parts)
    return ""


_NON_CHAT_MODEL_KEYWORDS: tuple[str, ...] = (
    "embedding",
    "embed-",
    "whisper",
    "tts",
    "dall-e",
    "imagen",
    "veo",
    "moderation",
    "transcribe",
    "audio",
    "speech",
    "realtime",
    "robotics",
    "computer-use",
    "image",
    "aqa",
    "rerank",
    "babbage",
    "davinci",
    "sora",
    "lyria",
    "nano-banana",
)

_CHAT_MODEL_PRIORITY: tuple[tuple[str, int], ...] = (
    ("flash-lite-latest", 0),
    ("flash-latest", 1),
    ("flash-lite", 2),
    ("flash", 3),
    ("mini", 4),
    ("deepseek-chat", 4),
    ("deepseek-flash", 4),
    ("haiku", 5),
    ("sonnet", 6),
    ("gpt-5", 6),
    ("gpt-4o", 7),
    ("versatile", 7),
    ("pro", 8),
)


def filter_and_rank_chat_models(
    raw_model_ids: list[str],
    *,
    current_model: str = "",
) -> list[str]:
    """Normalize provider model IDs, drop non-chat models, and rank fast chat models first."""
    seen: set[str] = set()
    cleaned: list[str] = []
    for raw in raw_model_ids:
        mid = str(raw or "").strip()
        if mid.startswith("models/"):
            mid = mid[len("models/") :]
        if not mid or mid in seen:
            continue
        low = mid.lower()
        if any(bad in low for bad in _NON_CHAT_MODEL_KEYWORDS):
            continue
        seen.add(mid)
        cleaned.append(mid)

    def _sort_key(mid: str) -> tuple[int, int, str]:
        if current_model and mid == current_model:
            return (-1, 0, mid.lower())
        low = mid.lower()
        rank = 50
        for token, score in _CHAT_MODEL_PRIORITY:
            if token in low:
                rank = min(rank, score)
        # Prefer stable names slightly over preview/experimental snapshots within same tier
        is_preview = 1 if ("preview" in low or "exp" in low or "alpha" in low) else 0
        return (rank, is_preview, low)

    cleaned.sort(key=_sort_key)
    return cleaned


class UnifiedLLMClient:
    """Async OpenAI-compatible LLM client with live key testing and parameter fallbacks."""

    def __init__(
        self,
        *,
        allow_local_endpoints: bool = False,
        request_timeout_seconds: int = 60,
        smoke_test_timeout_seconds: int = 15,
    ) -> None:
        self._allow_local = allow_local_endpoints
        self._request_timeout = request_timeout_seconds
        self._smoke_timeout = smoke_test_timeout_seconds

    def _build_sdk_client(
        self,
        *,
        api_key: str,
        base_url: str,
        timeout: float,
        max_retries: int = 1,
    ) -> AsyncOpenAI:
        return AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

    async def verify_credentials(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
    ) -> SmokeTestResult:
        """Run SSRF validation + a tiny real completion call to confirm key/endpoint/model/quota work."""
        try:
            normalized_url = await validate_endpoint_url_async(
                base_url, allow_local=self._allow_local, resolve_dns=True
            )
        except EndpointSecurityError as exc:
            raise LLMError(exc.code, detail=str(exc), model=model) from exc

        clean_key = (api_key or "").strip()
        clean_model = (model or "").strip()
        if not clean_key:
            raise LLMError("err_auth", detail="Empty API key.", model=clean_model)
        if not clean_model:
            raise LLMError("err_not_found", detail="Model ID is empty.", model=clean_model)

        client = self._build_sdk_client(
            api_key=clean_key,
            base_url=normalized_url,
            timeout=float(self._smoke_timeout),
            max_retries=0,
        )
        t0 = time.perf_counter()
        try:
            messages = [{"role": "user", "content": "Ping. Reply with OK."}]
            try:
                response = await client.chat.completions.create(
                    model=clean_model,
                    messages=messages,
                    max_tokens=8,
                    temperature=0,
                )
            except openai.BadRequestError as exc:
                err_text = str(exc).lower()
                # Reasoning models (e.g., o-series / newer GPT/Gemini) often require max_completion_tokens
                # and/or disallow temperature=0.
                if "max_tokens" in err_text or "max_completion_tokens" in err_text or "temperature" in err_text:
                    response = await client.chat.completions.create(
                        model=clean_model,
                        messages=messages,
                        max_completion_tokens=24,
                    )
                else:
                    raise

            text = _extract_choice_text(response).strip()
            # Some reasoning models consume small token budgets inside reasoning and return empty content,
            # while still returning a valid completion ID & choices array.
            if not getattr(response, "choices", None):
                raise LLMError(
                    "err_bad_request",
                    detail="Endpoint returned an empty or non-OpenAI-compatible response.",
                    model=clean_model,
                )

            latency_ms = max(1, int((time.perf_counter() - t0) * 1000))
            return SmokeTestResult(
                normalized_base_url=normalized_url,
                latency_ms=latency_ms,
                reply_preview=text[:30],
            )
        except LLMError:
            raise
        except Exception as exc:
            raise map_openai_exception(exc, api_key=clean_key, model=clean_model) from exc
        finally:
            await client.close()

    async def list_chat_models(self, creds: GuildLLMConfig) -> list[str]:
        """Query GET /models on the server's endpoint and return filtered, ranked chat model IDs."""
        try:
            normalized_url = await validate_endpoint_url_async(
                creds.base_url, allow_local=self._allow_local, resolve_dns=True
            )
        except EndpointSecurityError as exc:
            raise LLMError(exc.code, detail=str(exc), model=creds.model) from exc

        client = self._build_sdk_client(
            api_key=creds.api_key,
            base_url=normalized_url,
            timeout=float(min(self._smoke_timeout, 12)),
            max_retries=0,
        )
        try:
            page = await client.models.list()
            raw_ids: list[str] = []
            for item in getattr(page, "data", ()) or ():
                mid = getattr(item, "id", None)
                if mid:
                    raw_ids.append(str(mid))
            return filter_and_rank_chat_models(raw_ids, current_model=creds.model)
        except Exception as exc:
            raise map_openai_exception(exc, api_key=creds.api_key, model=creds.model) from exc
        finally:
            await client.close()

    async def complete_chat(
        self,
        creds: GuildLLMConfig,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 2400,
        temperature: float = 0.2,
        json_mode: bool = True,
    ) -> LLMResponse:
        """Execute a chat completion request with automatic compatibility fallbacks across providers."""
        # Re-validate endpoint against DNS rebinding / SSRF before sending chat transcripts
        try:
            normalized_url = await validate_endpoint_url_async(
                creds.base_url, allow_local=self._allow_local, resolve_dns=True
            )
        except EndpointSecurityError as exc:
            raise LLMError(exc.code, detail=str(exc), model=creds.model) from exc

        client = self._build_sdk_client(
            api_key=creds.api_key,
            base_url=normalized_url,
            timeout=float(self._request_timeout),
            max_retries=2,
        )
        try:
            kwargs: dict[str, Any] = {
                "model": creds.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            response = await self._call_with_param_fallback(client, kwargs)
            content = _extract_choice_text(response)
            if not content or not content.strip():
                raise LLMError(
                    "err_unknown",
                    detail="Model returned an empty summary response.",
                    model=creds.model,
                )

            usage = getattr(response, "usage", None)
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0

            return LLMResponse(
                content=content.strip(),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        except LLMError:
            raise
        except Exception as exc:
            raise map_openai_exception(exc, api_key=creds.api_key, model=creds.model) from exc
        finally:
            await client.close()

    async def _call_with_param_fallback(
        self, client: AsyncOpenAI, kwargs: dict[str, Any]
    ) -> Any:
        """Gracefully degrade optional params (response_format, temperature, max_tokens) on HTTP 400."""
        attempt_kwargs = dict(kwargs)
        for _ in range(3):
            try:
                return await client.chat.completions.create(**attempt_kwargs)
            except openai.BadRequestError as exc:
                err_text = str(exc).lower()
                changed = False
                if "response_format" in attempt_kwargs and (
                    "response_format" in err_text or "json_object" in err_text or "json" in err_text
                ):
                    attempt_kwargs.pop("response_format", None)
                    changed = True
                if "max_tokens" in attempt_kwargs and (
                    "max_tokens" in err_text or "max_completion_tokens" in err_text
                ):
                    val = attempt_kwargs.pop("max_tokens")
                    attempt_kwargs["max_completion_tokens"] = val
                    changed = True
                if "temperature" in attempt_kwargs and "temperature" in err_text:
                    attempt_kwargs.pop("temperature", None)
                    changed = True
                if not changed:
                    raise
        return await client.chat.completions.create(**attempt_kwargs)

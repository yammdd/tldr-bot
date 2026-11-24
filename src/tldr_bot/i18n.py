"""Bot UI strings (English), summary output languages, and SQLite guild preferences store."""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tldr_bot.storage.database import open_sqlite_wal


@dataclass(frozen=True)
class LanguageSpec:
    code: str
    label: str
    llm_instruction: str


LANGUAGE_SPECS: dict[str, LanguageSpec] = {
    "en": LanguageSpec(
        code="en",
        label="English",
        llm_instruction="English (clear, natural, concise)",
    ),
    "vi": LanguageSpec(
        code="vi",
        label="Vietnamese",
        llm_instruction="Vietnamese (natural, everyday conversational tone, clear and concise)",
    ),
    "ja": LanguageSpec(
        code="ja",
        label="Japanese",
        llm_instruction="Japanese (natural, concise, accurate)",
    ),
    "ko": LanguageSpec(
        code="ko",
        label="Korean",
        llm_instruction="Korean (natural, concise, accurate)",
    ),
    "zh": LanguageSpec(
        code="zh",
        label="Chinese (Simplified)",
        llm_instruction="Simplified Chinese (natural, concise, accurate)",
    ),
    "auto": LanguageSpec(
        code="auto",
        label="Auto-detect from chat",
        llm_instruction="the SAME dominant language used by the speakers in the conversation",
    ),
}


UI_STRINGS: dict[str, str] = {
    "guild_only": "This command only works inside a Discord server",
    "perm_role_too_low": (
        "Permission denied. Only the Server Owner or members with a role higher than "
        "`{bot_role}` can change the API Key, Endpoint, or bot configuration."
    ),
    "perm_no_channel_history": (
        "The bot lacks View Channel or Read Message History permission in this channel"
    ),
    "not_configured": (
        "This server has not configured an API Key and Endpoint yet. "
        "Someone with a role higher than the bot can run `/config` once to set it up."
    ),
    "cooldown_wait": "Wait **{seconds:.1f}s** before running another summary",
    "guild_busy": (
        "This server already reached its concurrent summary limit. Try again in a few seconds."
    ),
    "global_busy": (
        "The bot is currently under heavy load. Try again in a few seconds."
    ),
    "invalid_count": (
        "Message count must be between **{min_count} and {max_count}** for speed and accuracy"
    ),
    "invalid_since": (
        "Invalid time window. Examples: `30m`, `2h`, `12h`, `24h`, `3d`, `7d`, `today` (max **{max_days} days**)"
    ),
    "config_panel_title": "Configure API Key & Endpoint",
    "config_panel_desc": (
        "Select a provider below or choose **Custom Endpoint** to open the private modal.\n"
        "- Credentials are submitted privately and never touch chat history\n"
        "- A tiny live smoke test verifies Key, Endpoint, and Model before saving\n"
        "- Storage mode: `{storage_mode}` | Write-Only (keys can never be viewed)"
    ),
    "config_select_placeholder": "Select an API provider",
    "config_modal_title": "Configure {provider_name}",
    "config_field_apikey": "API Key (Write-Only)",
    "config_field_endpoint": "Endpoint Base URL",
    "config_field_model": "Model ID (e.g. {default_model})",
    "config_testing": "Verifying connection to **{provider_name}** (`{model}`)...",
    "config_success": (
        "Verification passed (`{latency_ms}ms`)\n"
        "- Provider: **{provider_name}**\n"
        "- Endpoint: `{endpoint_host}`\n"
        "- Model: `{model}`\n"
        "- Storage: `{storage_mode}` (Write-Only)\n\n"
        "Ready to use `/tldr` or right-click a message > Apps > **TL;DR from here**."
    ),
    "config_failed_keep_old": (
        "API Key verification failed. Existing configuration (if any) remains unchanged.\n"
        "- Reason: {error}\n"
        "- Check your API Key, Endpoint URL, Model ID, or balance and retry `/config`."
    ),
    "config_reset_done": "Cleared this server's API Key and LLM configuration",
    "config_reset_empty": "This server has no saved API configuration",
    "model_panel_title": "Select Model",
    "model_panel_desc": (
        "Provider: **{provider_name}** | `{endpoint_host}`\n"
        "Current model: `{current_model}` | Found **{count}** chat models\n\n"
        "Pick a model below or click **Enter manually**. A live smoke test runs before switching."
    ),
    "model_panel_no_list": (
        "Provider: **{provider_name}** | `{endpoint_host}`\n"
        "Current model: `{current_model}`\n\n"
        "This endpoint did not return a `/models` list ({reason}). Click **Enter manually** to set a model ID."
    ),
    "model_select_placeholder": "Choose a model ({start}-{end} of {total})",
    "model_option_active": "Active",
    "model_btn_manual": "Enter manually",
    "model_modal_title": "Switch Model ID",
    "model_modal_field": "New Model ID",
    "model_testing": "Testing model `{model}` on **{provider_name}**...",
    "model_switched": (
        "Switched model (`{latency_ms}ms`)\n"
        "- Provider: **{provider_name}** (`{endpoint_host}`)\n"
        "- Active model: `{model}`"
    ),
    "model_failed_keep_old": (
        "Could not switch to `{new_model}`. Kept previous model `{old_model}`.\n"
        "- Reason: {error}"
    ),
    "lang_updated": "Server default summary language set to **{language_label}**",
    "lang_current": "Current summary language: **{language_label}**",
    "status_title": "TL;DR Status",
    "status_configured_yes": "Configured | Write-Only",
    "status_configured_no": "Not configured (run `/config`)",
    "status_field_state": "Status",
    "status_field_provider": "Provider",
    "status_field_model": "Model",
    "status_field_lang": "Summary language",
    "status_field_limits": "Context limits",
    "status_limits_val": (
        "`{max_msgs}` msgs/run | `{max_days}` days | `{max_tokens:,}` input tokens | `{storage_mode}`"
    ),
    "status_footer": "Write-Only | API keys are never displayed",
    "help_title": "TL;DR Guide",
    "help_desc": (
        "Distills 5 to 1,500 unread channel messages into speaker-attributed topics, "
        "decisions, open questions, and highlights."
    ),
    "help_setup_header": "Admin setup (one-time)",
    "help_setup_body": (
        "1. In **Server Settings > Roles**, place the Bot role below your Admin/Mod roles\n"
        "2. Run `/config`, choose a provider (Gemini, DeepSeek, OpenAI, Claude, Custom...), and paste your API Key privately\n"
        "3. Optional: run `/model` to switch models without re-entering your key, or `/language` to choose the summary output language"
    ),
    "help_usage_header": "Everyday usage",
    "help_usage_body": (
        "- `/tldr count:500`: summarize the last 500 messages\n"
        "- `/tldr since:6h`: summarize by timeframe (`30m`, `2h`, `24h`, `3d`, `today`)\n"
        "- Right-click any message > **Apps > TL;DR from here**: catch up from that exact message\n"
        "- Summaries are visible to everyone by default; set `private:True` to keep one private"
    ),
    "help_footer": "Write-Only | Chat messages are never stored on disk",
    "unexpected_command_error": "An unexpected error occurred while running that command. Please try again.",
    "progress_fetching": "Fetching `#{channel}` | {fetched} messages",
    "progress_analyzing": (
        "`{fetched} -> {lines}` turns (~{tokens:,} tok) | Summarizing with `{model}`..."
    ),
    "progress_map_reduce": (
        "Long thread ({fetched} msgs, ~{tokens:,} tok) | Running {chunks} parallel segments with `{model}`..."
    ),
    "no_messages_found": "Not enough meaningful messages found in the selected range",
    "summary_title": "TL;DR | #{channel}",
    "summary_participants": "Participants ({count}): {list}",
    "summary_topics_header": "Key topics",
    "summary_decisions_header": "Decisions & action items",
    "summary_open_header": "Open questions",
    "summary_highlights_header": "Links & files",
    "summary_mentions_header": "Mentions of you",
    "summary_jump_link": "jump",
    "summary_jump_full": "jump",
    "summary_capped_note": (
        "Note: capped to the most recent {used:,}/{fetched:,} messages to maintain accuracy."
    ),
    "summary_cache_badge": " | cache",
    "summary_footer": (
        "Only visible to you | {fetched} -> {lines} | {tokens:,} tok | {elapsed:.1f}s | {model}{cache_badge}"
    ),
    "summary_public_footer": (
        "Requested by {requester} | {fetched} -> {lines} | {tokens:,} tok | {elapsed:.1f}s | {model}{cache_badge}"
    ),
    "topic_status_decided": "Decided",
    "topic_status_open": "Open",
    "topic_status_resolved": "Resolved",
    "topic_status_info": "",
    "btn_refresh": "Regenerate",
    "btn_prev": "Prev",
    "btn_next": "Next",
    "btn_working": "Updating summary...",
    "err_ssrf_scheme": "Endpoint URL must start with `https://`",
    "err_ssrf_private": "Endpoint blocked because it resolves to a private or localhost address",
    "err_ssrf_invalid": "Invalid Endpoint URL format",
    "err_ssrf_dns": "Could not resolve the Endpoint hostname",
    "err_auth": "Invalid or revoked API Key (HTTP 401). Update it via `/config`.",
    "err_forbidden": "API Key lacks access to this model or region (HTTP 403)",
    "err_quota": "API Key out of credits or rate limited (HTTP 402/429). Retry later or update via `/config`.",
    "err_not_found": "Model `{model}` not found or Endpoint path is wrong (HTTP 404): {detail}",
    "err_bad_request": "Endpoint rejected the request (HTTP 400): {detail}",
    "err_server": "Provider returned a temporary server error (HTTP 5xx). Try again shortly.",
    "err_timeout": "Endpoint timed out. Retry or reduce the message count.",
    "err_connection": "Could not reach the Endpoint URL",
    "err_unknown": "API error: {detail}",
    "summary_failed": "Could not generate summary.\n{error}",
}


def get_language_spec(code: str | None, fallback: str = "en") -> LanguageSpec:
    normalized = (code or "").strip().lower()
    if normalized in LANGUAGE_SPECS:
        return LANGUAGE_SPECS[normalized]
    return LANGUAGE_SPECS.get(fallback, LANGUAGE_SPECS["en"])


def t(key: str, **kwargs: Any) -> str:
    """Format an English UI string safely (preserving empty-string entries without key fallback)."""
    template = UI_STRINGS[key] if key in UI_STRINGS else key
    if kwargs and template:
        try:
            return template.format(**kwargs)
        except (KeyError, ValueError):
            return template
    return template


class GuildPreferencesStore:
    """Thread-safe SQLite WAL store for non-secret per-guild preferences (summary language)."""

    def __init__(
        self,
        db_file: Path,
        default_language: str = "en",
        *,
        persist: bool = True,
    ) -> None:
        self._default_language = get_language_spec(default_language).code
        self._persist = persist
        self._lock = threading.RLock()
        self._cache: dict[int, str] = {}
        self._conn: sqlite3.Connection | None = (
            open_sqlite_wal(db_file) if persist else None
        )

    def get_language(self, guild_id: int | None) -> str:
        if guild_id is None:
            return self._default_language
        with self._lock:
            cached = self._cache.get(guild_id)
            if cached is not None:
                return cached
            if self._conn is not None:
                row = self._conn.execute(
                    "SELECT language FROM guild_preferences WHERE guild_id = ?",
                    (guild_id,),
                ).fetchone()
                if row is not None:
                    code = get_language_spec(row["language"], fallback=self._default_language).code
                    self._cache[guild_id] = code
                    return code
            return self._default_language

    def set_language(self, guild_id: int, language_code: str) -> LanguageSpec:
        spec = get_language_spec(language_code, fallback=self._default_language)
        now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._lock:
            self._cache[guild_id] = spec.code
            if self._conn is not None:
                self._conn.execute(
                    """
                    INSERT INTO guild_preferences (guild_id, language, updated_at_iso)
                    VALUES (?, ?, ?)
                    ON CONFLICT(guild_id) DO UPDATE SET
                        language = excluded.language,
                        updated_at_iso = excluded.updated_at_iso
                    """,
                    (guild_id, spec.code, now_iso),
                )
                self._conn.commit()
        return spec

    def delete_guild(self, guild_id: int) -> None:
        with self._lock:
            self._cache.pop(guild_id, None)
            if self._conn is not None:
                self._conn.execute(
                    "DELETE FROM guild_preferences WHERE guild_id = ?", (guild_id,)
                )
                self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._cache.clear()
            if self._conn is not None:
                self._conn.close()
                self._conn = None

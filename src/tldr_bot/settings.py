"""Application settings loaded from environment variables (.env)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

SUPPORTED_LANGUAGES: tuple[str, ...] = ("en", "vi", "ja", "ko", "zh", "auto")


def _parse_bool(raw: str | None, default: bool = False) -> bool:
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _parse_int(raw: str | None, default: int, *, min_val: int, max_val: int) -> int:
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw.strip())
    except ValueError:
        return default
    return max(min_val, min(max_val, value))


@dataclass(frozen=True)
class Settings:
    discord_token: str
    key_storage_mode: Literal["encrypted", "memory"] = "encrypted"
    vault_master_key: str = ""
    db_file_path: Path = Path("data/tldr_bot.db")
    default_language: str = "en"

    # Message & context budget limits
    max_messages_hard_cap: int = 1500
    default_message_count: int = 100
    min_message_count: int = 5
    max_days_since: int = 7
    max_single_message_chars: int = 800
    single_pass_token_threshold: int = 28000
    map_chunk_target_tokens: int = 18000
    max_total_input_tokens: int = 100000
    max_parallel_map_calls: int = 3

    # Multi-server concurrency, caching & reliability
    cache_ttl_seconds: int = 180
    cache_max_entries: int = 512
    user_cooldown_seconds: int = 10
    max_concurrent_per_guild: int = 2
    max_concurrent_global: int = 20
    purge_on_guild_remove: bool = False
    llm_request_timeout_seconds: int = 60
    smoke_test_timeout_seconds: int = 15

    # Deployment, healthcheck & security
    health_port: int = 0
    allow_local_endpoints: bool = False
    dev_guild_id: int | None = None
    log_level: str = "INFO"

    @classmethod
    def from_env(cls, env_file: str | Path | None = None) -> "Settings":
        if env_file is not None:
            load_dotenv(env_file, override=False)
        else:
            load_dotenv(override=False)

        raw_mode = (os.getenv("KEY_STORAGE_MODE") or "encrypted").strip().lower()
        storage_mode: Literal["encrypted", "memory"] = (
            "memory" if raw_mode == "memory" else "encrypted"
        )

        raw_lang = (os.getenv("DEFAULT_LANGUAGE") or "en").strip().lower()
        default_lang = raw_lang if raw_lang in SUPPORTED_LANGUAGES else "en"

        raw_dev_guild = (os.getenv("DEV_GUILD_ID") or "").strip()
        dev_guild_id = int(raw_dev_guild) if raw_dev_guild.isdigit() else None

        max_cap = _parse_int(
            os.getenv("MAX_MESSAGES_HARD_CAP"), 1500, min_val=50, max_val=2000
        )
        default_count = _parse_int(
            os.getenv("DEFAULT_MESSAGE_COUNT"), 100, min_val=5, max_val=max_cap
        )
        min_count = _parse_int(
            os.getenv("MIN_MESSAGE_COUNT"), 5, min_val=1, max_val=min(50, default_count)
        )

        return cls(
            discord_token=(os.getenv("DISCORD_TOKEN") or "").strip(),
            key_storage_mode=storage_mode,
            vault_master_key=(os.getenv("VAULT_MASTER_KEY") or "").strip(),
            db_file_path=Path(os.getenv("DB_FILE_PATH") or "data/tldr_bot.db"),
            default_language=default_lang,
            max_messages_hard_cap=max_cap,
            default_message_count=default_count,
            min_message_count=min_count,
            max_days_since=_parse_int(
                os.getenv("MAX_DAYS_SINCE"), 7, min_val=1, max_val=14
            ),
            max_single_message_chars=_parse_int(
                os.getenv("MAX_SINGLE_MESSAGE_CHARS"), 800, min_val=200, max_val=4000
            ),
            single_pass_token_threshold=_parse_int(
                os.getenv("SINGLE_PASS_TOKEN_THRESHOLD"),
                28000,
                min_val=4000,
                max_val=120000,
            ),
            map_chunk_target_tokens=_parse_int(
                os.getenv("MAP_CHUNK_TARGET_TOKENS"),
                18000,
                min_val=4000,
                max_val=60000,
            ),
            max_total_input_tokens=_parse_int(
                os.getenv("MAX_TOTAL_INPUT_TOKENS"),
                100000,
                min_val=10000,
                max_val=300000,
            ),
            max_parallel_map_calls=_parse_int(
                os.getenv("MAX_PARALLEL_MAP_CALLS"), 3, min_val=1, max_val=6
            ),
            cache_ttl_seconds=_parse_int(
                os.getenv("CACHE_TTL_SECONDS"), 180, min_val=0, max_val=3600
            ),
            cache_max_entries=_parse_int(
                os.getenv("CACHE_MAX_ENTRIES"), 512, min_val=32, max_val=4096
            ),
            user_cooldown_seconds=_parse_int(
                os.getenv("USER_COOLDOWN_SECONDS"), 10, min_val=0, max_val=120
            ),
            max_concurrent_per_guild=_parse_int(
                os.getenv("MAX_CONCURRENT_PER_GUILD"), 2, min_val=1, max_val=10
            ),
            max_concurrent_global=_parse_int(
                os.getenv("MAX_CONCURRENT_GLOBAL"), 20, min_val=1, max_val=100
            ),
            purge_on_guild_remove=_parse_bool(
                os.getenv("PURGE_ON_GUILD_REMOVE"), default=False
            ),
            llm_request_timeout_seconds=_parse_int(
                os.getenv("LLM_REQUEST_TIMEOUT_SECONDS"), 60, min_val=15, max_val=180
            ),
            smoke_test_timeout_seconds=_parse_int(
                os.getenv("SMOKE_TEST_TIMEOUT_SECONDS"), 15, min_val=5, max_val=45
            ),
            health_port=_parse_int(
                os.getenv("HEALTH_PORT") or os.getenv("PORT"),
                0,
                min_val=0,
                max_val=65535,
            ),
            allow_local_endpoints=_parse_bool(
                os.getenv("ALLOW_LOCAL_ENDPOINTS"), default=False
            ),
            dev_guild_id=dev_guild_id,
            log_level=(os.getenv("LOG_LEVEL") or "INFO").strip().upper(),
        )

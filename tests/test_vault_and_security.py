"""Tests for SQLite WAL KeyVault (AES-256-GCM + AAD & RAM-only), SSRF, roles, redaction, i18n & concurrency."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pytest

from tldr_bot.discord_ext.concurrency import ConcurrencyGuard, ConcurrencyLimitError
from tldr_bot.i18n import UI_STRINGS, GuildPreferencesStore, t
from tldr_bot.security.permissions import check_role_above_bot
from tldr_bot.security.redactor import redact_secrets
from tldr_bot.security.ssrf import EndpointSecurityError, validate_endpoint_url
from tldr_bot.security.vault import KeyVault


@dataclass
class _Role:
    name: str
    position: int


@dataclass
class _Member:
    id: int
    top_role: _Role


@dataclass
class _Guild:
    owner_id: int
    me: _Member


def test_encrypted_sqlite_vault_roundtrip_aad_tamper_detection_and_write_only(tmp_path: Path) -> None:
    db_file = tmp_path / "tldr_bot.db"
    secret_key = "sk-live-super-secret-value-999999999"

    vault = KeyVault(mode="encrypted", db_file=db_file, master_key="strong-master-passphrase")
    status = vault.set_config(
        1001,
        provider_id="deepseek",
        provider_name="DeepSeek",
        base_url="https://api.deepseek.com",
        model="deepseek-chat",
        api_key=secret_key,
    )

    assert status.configured is True
    assert status.provider_name == "DeepSeek"
    assert status.endpoint_host == "api.deepseek.com"
    assert secret_key not in repr(status)
    assert not hasattr(status, "api_key")
    assert vault.configured_guild_count() == 1

    # Raw SQLite file NEVER contains plaintext API key
    raw_bytes = db_file.read_bytes()
    assert secret_key.encode("utf-8") not in raw_bytes

    # Runtime repr/str never leaks API key
    creds = vault.get_runtime_credentials(1001)
    assert creds is not None
    assert creds.api_key == secret_key
    assert secret_key not in repr(creds)
    assert secret_key not in str(creds)
    vault.close()

    # Survives restart with same master key
    reloaded = KeyVault(mode="encrypted", db_file=db_file, master_key="strong-master-passphrase")
    reloaded_creds = reloaded.get_runtime_credentials(1001)
    assert reloaded_creds is not None
    assert reloaded_creds.api_key == secret_key
    assert reloaded_creds.model == "deepseek-chat"

    # Switching model preserves encrypted API key without re-entering it
    updated_status = reloaded.update_model(1001, "deepseek-v4-pro")
    assert updated_status.model == "deepseek-v4-pro"
    creds_after_switch = reloaded.get_runtime_credentials(1001)
    assert creds_after_switch is not None
    assert creds_after_switch.model == "deepseek-v4-pro"
    assert creds_after_switch.api_key == secret_key

    # Cross-guild ciphertext swap attack MUST fail AAD authentication!
    conn = sqlite3.connect(str(db_file))
    conn.execute("UPDATE guild_credentials SET guild_id = 2002 WHERE guild_id = 1001")
    conn.commit()
    conn.close()
    with pytest.raises(ValueError, match="Corrupted or tampered"):
        reloaded.get_runtime_credentials(2002)

    # Restore guild_id and verify deletion & wrong master key rejection
    conn = sqlite3.connect(str(db_file))
    conn.execute("UPDATE guild_credentials SET guild_id = 1001 WHERE guild_id = 2002")
    conn.commit()
    conn.close()

    assert reloaded.delete_config(1001) is True
    assert reloaded.delete_config(1001) is False
    reloaded.close()

    with pytest.raises(ValueError):
        KeyVault(mode="encrypted", db_file=db_file, master_key="wrong-passphrase")


def test_memory_vault_has_zero_disk_footprint(tmp_path: Path) -> None:
    db_file = tmp_path / "should_not_exist.db"
    vault = KeyVault(mode="memory", db_file=db_file)
    vault.set_config(
        2002,
        provider_id="gemini",
        provider_name="Google Gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        model="gemini-2.5-flash",
        api_key="AIzaSyTestSecretKey123456789012345",
    )
    assert not db_file.exists()
    assert vault.get_public_status(2002).configured is True

    vault.clear_memory()
    assert vault.get_public_status(2002).configured is False


def test_ssrf_validation_and_normalization() -> None:
    normalized = validate_endpoint_url(
        "https://api.deepseek.com/v1/chat/completions",
        allow_local=False,
        resolve_dns=False,
    )
    assert normalized == "https://api.deepseek.com/v1"

    with pytest.raises(EndpointSecurityError) as exc_info:
        validate_endpoint_url("http://api.openai.com/v1", allow_local=False, resolve_dns=False)
    assert exc_info.value.code == "err_ssrf_scheme"

    for bad_url in (
        "https://localhost:8080/v1",
        "https://127.0.0.1/v1",
        "https://169.254.169.254/latest/meta-data",
        "https://10.0.0.5/v1",
        "https://192.168.1.10/v1",
        "https://[::1]/v1",
    ):
        with pytest.raises(EndpointSecurityError) as err:
            validate_endpoint_url(bad_url, allow_local=False, resolve_dns=False)
        assert err.value.code == "err_ssrf_private"

    with pytest.raises(EndpointSecurityError):
        validate_endpoint_url("https://user:pass@api.openai.com/v1", allow_local=False, resolve_dns=False)
    with pytest.raises(EndpointSecurityError):
        validate_endpoint_url("https://api.openai.com/v1?key=123", allow_local=False, resolve_dns=False)

    local_ok = validate_endpoint_url("http://localhost:11434/v1/", allow_local=True, resolve_dns=False)
    assert local_ok == "http://localhost:11434/v1"


def test_role_hierarchy_permission_rules() -> None:
    bot = _Member(id=999, top_role=_Role(name="TL;DR Bot", position=5))
    guild = _Guild(owner_id=1, me=bot)

    owner = _Member(id=1, top_role=_Role(name="@everyone", position=0))
    assert check_role_above_bot(owner, guild).allowed is True

    admin = _Member(id=2, top_role=_Role(name="Admin", position=6))
    assert check_role_above_bot(admin, guild).allowed is True

    equal_member = _Member(id=3, top_role=_Role(name="Equal", position=5))
    res_eq = check_role_above_bot(equal_member, guild)
    assert res_eq.allowed is False
    assert res_eq.bot_top_role_name == "TL;DR Bot"

    regular = _Member(id=4, top_role=_Role(name="Member", position=1))
    assert check_role_above_bot(regular, guild).allowed is False

    assert check_role_above_bot(admin, None).allowed is False


def test_secret_redactor() -> None:
    sample = (
        "Error with Authorization: Bearer abcdef1234567890xyz and "
        "sk-proj-secret1234567890abcdef and AIzaSyD1234567890abcdefghijk and "
        "AQ.FakeTestKeyExample1234567890abcdef and custom_secret_token_77"
    )
    cleaned = redact_secrets(sample, extra_secrets=("custom_secret_token_77",))
    assert "abcdef1234567890xyz" not in cleaned
    assert "sk-proj-secret" not in cleaned
    assert "AIzaSyD" not in cleaned
    assert "AQ.FakeTestKey" not in cleaned
    assert "custom_secret_token_77" not in cleaned


def test_guild_language_preferences_sqlite_and_unslop_copy(tmp_path: Path) -> None:
    db_file = tmp_path / "tldr_bot.db"
    store = GuildPreferencesStore(db_file, default_language="vi", persist=True)
    assert store.get_language(555) == "vi"

    store.set_language(555, "en")
    store.close()

    reloaded = GuildPreferencesStore(db_file, default_language="vi", persist=True)
    assert reloaded.get_language(555) == "en"
    reloaded.delete_guild(555)
    assert reloaded.get_language(555) == "vi"
    reloaded.close()

    assert "Permission denied" in t("en", "perm_role_too_low", bot_role="Bot")
    assert "Không đủ quyền" in t("vi", "perm_role_too_low", bot_role="Bot")

    # Unslop copy guard: zero exclamation marks in any UI catalog string
    for locale, strings in UI_STRINGS.items():
        for key, val in strings.items():
            assert "!" not in val, f"Exclamation mark found in UI_STRINGS[{locale}][{key}]"


@pytest.mark.asyncio
async def test_concurrency_guard_enforces_guild_and_global_limits() -> None:
    guard = ConcurrencyGuard(max_per_guild=1, max_global=2)

    async with guard.acquire(100):
        assert guard.guild_active(100) == 1
        assert guard.global_active == 1

        # Second concurrent request in same guild rejected immediately
        with pytest.raises(ConcurrencyLimitError) as guild_err:
            async with guard.acquire(100):
                pass
        assert guild_err.value.scope == "guild_busy"

        # Another guild succeeds up to global cap
        async with guard.acquire(200):
            assert guard.global_active == 2
            with pytest.raises(ConcurrencyLimitError) as global_err:
                async with guard.acquire(300):
                    pass
            assert global_err.value.scope == "global_busy"

    assert guard.global_active == 0
    assert guard.guild_active(100) == 0

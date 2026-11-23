"""Write-only per-guild API key vault backed by per-row AES-256-GCM in SQLite WAL or RAM-only mode."""

from __future__ import annotations

import json
import os
import secrets
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from tldr_bot.storage.database import open_sqlite_wal, restrict_file_permissions

_KDF_ITERATIONS = 310_000
_META_SALT_KEY = "kdf_salt_v2"
_META_CHECK_KEY = "key_check_v2"
_CHECK_AAD = b"tldr-bot-v2-key-check"
_CHECK_PLAINTEXT = b"tldr-bot-valid-master-key"


@dataclass(frozen=True)
class GuildLLMConfig:
    """Internal runtime credentials for a guild. Never exposes api_key via repr/str."""

    provider_id: str
    provider_name: str
    base_url: str
    model: str
    api_key: str
    updated_at_iso: str

    def __repr__(self) -> str:
        return (
            f"GuildLLMConfig(provider_id={self.provider_id!r}, "
            f"base_url={self.base_url!r}, model={self.model!r}, api_key='***REDACTED***')"
        )

    def __str__(self) -> str:
        return self.__repr__()


@dataclass(frozen=True)
class GuildConfigStatus:
    """Public non-secret status object safe for display in Discord embeds."""

    configured: bool
    storage_mode: str
    provider_id: str = ""
    provider_name: str = ""
    endpoint_host: str = ""
    model: str = ""
    updated_at_iso: str = ""


def _safe_host(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.netloc or url
    path = parsed.path.rstrip("/")
    return f"{host}{path}" if path else host


def _guild_aad(guild_id: int) -> bytes:
    """Bind ciphertext integrity strictly to its owning guild_id."""
    return f"tldr-bot-v2-guild-{guild_id}".encode("ascii")


def _derive_aes_key(master_secret: bytes, salt: bytes) -> bytes:
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=_KDF_ITERATIONS,
    )
    return kdf.derive(master_secret)


class KeyVault:
    """Write-only per-guild API key vault designed for multi-server scale.

    - `memory` mode: keys reside strictly in RAM with zero disk writes.
    - `encrypted` mode: stores one AES-256-GCM encrypted row per guild in SQLite WAL.
      Derives the KEK once at startup for sub-millisecond reads/writes under concurrency,
      uses a unique 12-byte nonce and guild-bound AAD per row, and never exposes a view-key API.
    """

    def __init__(
        self,
        *,
        mode: Literal["encrypted", "memory"] = "encrypted",
        db_file: Path = Path("data/tldr_bot.db"),
        master_key: str = "",
    ) -> None:
        self._mode: Literal["encrypted", "memory"] = mode
        self._db_file = db_file
        self._lock = threading.RLock()
        self._memory_store: dict[int, GuildLLMConfig] = {}
        self._conn: sqlite3.Connection | None = None
        self._aesgcm: AESGCM | None = None

        if self._mode == "encrypted":
            master_secret = self._resolve_master_secret(master_key)
            self._conn = open_sqlite_wal(db_file)
            self._aesgcm = self._init_cipher(master_secret)

    @property
    def storage_mode(self) -> str:
        return self._mode

    def _resolve_master_secret(self, master_key: str) -> bytes:
        if master_key and master_key.strip():
            return master_key.strip().encode("utf-8")

        key_file = self._db_file.parent / ".master.key"
        if key_file.exists():
            raw = key_file.read_text(encoding="utf-8").strip()
            if raw:
                return raw.encode("utf-8")

        key_file.parent.mkdir(parents=True, exist_ok=True)
        generated = secrets.token_urlsafe(32)
        key_file.write_text(generated, encoding="utf-8")
        restrict_file_permissions(key_file)
        return generated.encode("utf-8")

    def _init_cipher(self, master_secret: bytes) -> AESGCM:
        assert self._conn is not None
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM vault_meta WHERE key = ?", (_META_SALT_KEY,)
            ).fetchone()

            if row is None:
                salt = os.urandom(16)
                aesgcm = AESGCM(_derive_aes_key(master_secret, salt))
                check_nonce = os.urandom(12)
                check_blob = check_nonce + aesgcm.encrypt(
                    check_nonce, _CHECK_PLAINTEXT, _CHECK_AAD
                )
                self._conn.execute(
                    "INSERT INTO vault_meta (key, value) VALUES (?, ?)",
                    (_META_SALT_KEY, salt),
                )
                self._conn.execute(
                    "INSERT INTO vault_meta (key, value) VALUES (?, ?)",
                    (_META_CHECK_KEY, check_blob),
                )
                self._conn.commit()
                return aesgcm

            salt = bytes(row["value"])
            aesgcm = AESGCM(_derive_aes_key(master_secret, salt))
            check_row = self._conn.execute(
                "SELECT value FROM vault_meta WHERE key = ?", (_META_CHECK_KEY,)
            ).fetchone()
            if check_row is not None:
                blob = bytes(check_row["value"])
                nonce, ct = blob[:12], blob[12:]
                try:
                    plain = aesgcm.decrypt(nonce, ct, _CHECK_AAD)
                except InvalidTag as exc:
                    raise ValueError(
                        "Failed to unlock vault. Ensure VAULT_MASTER_KEY matches the key used to create the database."
                    ) from exc
                if plain != _CHECK_PLAINTEXT:
                    raise ValueError("Vault integrity verification failed.")
            return aesgcm

    def set_config(
        self,
        guild_id: int,
        *,
        provider_id: str,
        provider_name: str,
        base_url: str,
        model: str,
        api_key: str,
    ) -> GuildConfigStatus:
        """Write or overwrite a guild's LLM configuration. Never returns the API key."""
        clean_key = api_key.strip()
        if not clean_key:
            raise ValueError("API key must not be empty.")

        now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
        clean_provider_id = provider_id.strip()
        clean_provider_name = provider_name.strip()
        clean_base_url = base_url.strip()
        clean_model = model.strip()
        host = _safe_host(clean_base_url)

        with self._lock:
            if self._mode == "memory":
                self._memory_store[guild_id] = GuildLLMConfig(
                    provider_id=clean_provider_id,
                    provider_name=clean_provider_name,
                    base_url=clean_base_url,
                    model=clean_model,
                    api_key=clean_key,
                    updated_at_iso=now_iso,
                )
            else:
                assert self._conn is not None and self._aesgcm is not None
                secret_payload = json.dumps(
                    {"base_url": clean_base_url, "api_key": clean_key},
                    ensure_ascii=False,
                ).encode("utf-8")
                nonce = os.urandom(12)
                ciphertext = self._aesgcm.encrypt(
                    nonce, secret_payload, _guild_aad(guild_id)
                )
                self._conn.execute(
                    """
                    INSERT INTO guild_credentials (
                        guild_id, provider_id, provider_name, endpoint_host,
                        model, nonce, ciphertext, updated_at_iso
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(guild_id) DO UPDATE SET
                        provider_id = excluded.provider_id,
                        provider_name = excluded.provider_name,
                        endpoint_host = excluded.endpoint_host,
                        model = excluded.model,
                        nonce = excluded.nonce,
                        ciphertext = excluded.ciphertext,
                        updated_at_iso = excluded.updated_at_iso
                    """,
                    (
                        guild_id,
                        clean_provider_id,
                        clean_provider_name,
                        host,
                        clean_model,
                        nonce,
                        ciphertext,
                        now_iso,
                    ),
                )
                self._conn.commit()

        return self.get_public_status(guild_id)

    def update_model(self, guild_id: int, new_model: str) -> GuildConfigStatus:
        """Switch the active model for a configured guild without touching or exposing its API key."""
        clean_model = (new_model or "").strip()
        if not clean_model:
            raise ValueError("Model ID must not be empty.")

        now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._lock:
            if self._mode == "memory":
                existing = self._memory_store.get(guild_id)
                if existing is None:
                    raise KeyError(f"Guild {guild_id} is not configured.")
                self._memory_store[guild_id] = GuildLLMConfig(
                    provider_id=existing.provider_id,
                    provider_name=existing.provider_name,
                    base_url=existing.base_url,
                    model=clean_model,
                    api_key=existing.api_key,
                    updated_at_iso=now_iso,
                )
            else:
                assert self._conn is not None
                cur = self._conn.execute(
                    """
                    UPDATE guild_credentials
                    SET model = ?, updated_at_iso = ?
                    WHERE guild_id = ?
                    """,
                    (clean_model, now_iso, guild_id),
                )
                self._conn.commit()
                if cur.rowcount == 0:
                    raise KeyError(f"Guild {guild_id} is not configured.")

        return self.get_public_status(guild_id)

    def delete_config(self, guild_id: int) -> bool:
        """Delete a guild's credentials cleanly."""
        with self._lock:
            if self._mode == "memory":
                return self._memory_store.pop(guild_id, None) is not None
            assert self._conn is not None
            cur = self._conn.execute(
                "DELETE FROM guild_credentials WHERE guild_id = ?", (guild_id,)
            )
            self._conn.commit()
            return cur.rowcount > 0

    def get_public_status(self, guild_id: int) -> GuildConfigStatus:
        """Return safe non-secret display metadata without decrypting any secret key."""
        with self._lock:
            if self._mode == "memory":
                cfg = self._memory_store.get(guild_id)
                if cfg is None:
                    return GuildConfigStatus(configured=False, storage_mode=self._mode)
                return GuildConfigStatus(
                    configured=True,
                    storage_mode=self._mode,
                    provider_id=cfg.provider_id,
                    provider_name=cfg.provider_name,
                    endpoint_host=_safe_host(cfg.base_url),
                    model=cfg.model,
                    updated_at_iso=cfg.updated_at_iso,
                )

            assert self._conn is not None
            row = self._conn.execute(
                """
                SELECT provider_id, provider_name, endpoint_host, model, updated_at_iso
                FROM guild_credentials WHERE guild_id = ?
                """,
                (guild_id,),
            ).fetchone()
            if row is None:
                return GuildConfigStatus(configured=False, storage_mode=self._mode)
            return GuildConfigStatus(
                configured=True,
                storage_mode=self._mode,
                provider_id=row["provider_id"],
                provider_name=row["provider_name"],
                endpoint_host=row["endpoint_host"],
                model=row["model"],
                updated_at_iso=row["updated_at_iso"],
            )

    def get_runtime_credentials(self, guild_id: int) -> GuildLLMConfig | None:
        """Internal runtime access exclusively for LLM execution."""
        with self._lock:
            if self._mode == "memory":
                return self._memory_store.get(guild_id)

            assert self._conn is not None and self._aesgcm is not None
            row = self._conn.execute(
                """
                SELECT provider_id, provider_name, model, nonce, ciphertext, updated_at_iso
                FROM guild_credentials WHERE guild_id = ?
                """,
                (guild_id,),
            ).fetchone()
            if row is None:
                return None

            try:
                plaintext = self._aesgcm.decrypt(
                    bytes(row["nonce"]),
                    bytes(row["ciphertext"]),
                    _guild_aad(guild_id),
                )
            except InvalidTag as exc:
                raise ValueError(
                    f"Corrupted or tampered credentials for guild {guild_id}."
                ) from exc

            payload = json.loads(plaintext.decode("utf-8"))
            return GuildLLMConfig(
                provider_id=row["provider_id"],
                provider_name=row["provider_name"],
                base_url=payload["base_url"],
                model=row["model"],
                api_key=payload["api_key"],
                updated_at_iso=row["updated_at_iso"],
            )

    def configured_guild_count(self) -> int:
        with self._lock:
            if self._mode == "memory":
                return len(self._memory_store)
            assert self._conn is not None
            row = self._conn.execute("SELECT COUNT(*) AS cnt FROM guild_credentials").fetchone()
            return int(row["cnt"]) if row else 0

    def close(self) -> None:
        """Clear in-memory references and close database connection cleanly."""
        with self._lock:
            self._memory_store.clear()
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def clear_memory(self) -> None:
        """Alias retained for clean shutdown semantics."""
        self.close()

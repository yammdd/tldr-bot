"""SQLite WAL connection and schema initialization for multi-server scale."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS vault_meta (
    key TEXT PRIMARY KEY,
    value BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS guild_credentials (
    guild_id INTEGER PRIMARY KEY,
    provider_id TEXT NOT NULL,
    provider_name TEXT NOT NULL,
    endpoint_host TEXT NOT NULL,
    model TEXT NOT NULL,
    nonce BLOB NOT NULL,
    ciphertext BLOB NOT NULL,
    updated_at_iso TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS guild_preferences (
    guild_id INTEGER PRIMARY KEY,
    language TEXT NOT NULL,
    updated_at_iso TEXT NOT NULL
);
"""


def restrict_file_permissions(path: Path) -> None:
    """Best-effort 0600 permissions on POSIX hosts."""
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def open_sqlite_wal(db_path: Path) -> sqlite3.Connection:
    """Open a WAL-mode SQLite database with safe pragmas and initialized tables."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.executescript(_SCHEMA_SQL)
    conn.commit()
    restrict_file_permissions(db_path)
    return conn

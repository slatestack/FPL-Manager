"""Minimal, deterministic SQL migration runner.

Applies ``migrations/NNNN_*.sql`` in lexical order, each in its own transaction,
recording a SHA-256 checksum in ``schema_migrations``. A Postgres advisory lock
prevents two processes migrating at once; editing an already-applied file is an error.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import asyncpg

log = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"
_ADVISORY_LOCK_KEY = 0x46504C4D  # "FPLM"


class MigrationError(RuntimeError):
    pass


def discover(directory: Path = MIGRATIONS_DIR) -> list[tuple[str, str, str]]:
    """Return ``(version, sql, checksum)`` for every migration file, sorted."""
    out = []
    for path in sorted(directory.glob("*.sql")):
        sql = path.read_text(encoding="utf-8")
        out.append((path.stem, sql, hashlib.sha256(sql.encode()).hexdigest()))
    return out


async def migrate(dsn: str, directory: Path = MIGRATIONS_DIR) -> list[str]:
    """Apply pending migrations; returns the versions applied in this call."""
    conn = await asyncpg.connect(dsn)
    applied_now: list[str] = []
    try:
        await conn.execute("SELECT pg_advisory_lock($1)", _ADVISORY_LOCK_KEY)
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version    TEXT PRIMARY KEY,
                checksum   VARCHAR(64) NOT NULL,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        existing = {r["version"]: r["checksum"] for r in await conn.fetch("SELECT version, checksum FROM schema_migrations")}
        for version, sql, checksum in discover(directory):
            if version in existing:
                if existing[version] != checksum:
                    raise MigrationError(f"migration {version} was modified after being applied")
                continue
            log.info("applying migration %s", version)
            async with conn.transaction():
                await conn.execute(sql)
                await conn.execute(
                    "INSERT INTO schema_migrations (version, checksum) VALUES ($1, $2)", version, checksum
                )
            applied_now.append(version)
    finally:
        try:
            await conn.execute("SELECT pg_advisory_unlock($1)", _ADVISORY_LOCK_KEY)
        finally:
            await conn.close()
    return applied_now

"""asyncpg connection pool helpers."""

from __future__ import annotations

import json

import asyncpg


async def _init_connection(conn: asyncpg.Connection) -> None:
    # Transparent JSON <-> Python for json/jsonb columns.
    for typ in ("json", "jsonb"):
        await conn.set_type_codec(typ, encoder=json.dumps, decoder=json.loads, schema="pg_catalog")


async def create_pool(dsn: str, *, min_size: int = 1, max_size: int = 10) -> asyncpg.Pool:
    return await asyncpg.create_pool(dsn, min_size=min_size, max_size=max_size, init=_init_connection)

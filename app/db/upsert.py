"""Generic bulk upsert built on ``INSERT ... ON CONFLICT DO UPDATE``.

Rows are only rewritten when at least one column actually changed, so
``updated_at`` (and the history triggers on ``players``) reflect real changes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import asyncpg


def _q(ident: str) -> str:
    if not ident.replace("_", "").isalnum():
        raise ValueError(f"unsafe SQL identifier: {ident!r}")
    return f'"{ident}"'


def build_upsert_sql(
    table: str,
    columns: Sequence[str],
    conflict_columns: Sequence[str],
    *,
    touch_updated_at: bool = True,
) -> str:
    cols = ", ".join(_q(c) for c in columns)
    params = ", ".join(f"${i}" for i in range(1, len(columns) + 1))
    conflict = ", ".join(_q(c) for c in conflict_columns)
    update_cols = [c for c in columns if c not in conflict_columns]
    if not update_cols:
        return f"INSERT INTO {_q(table)} ({cols}) VALUES ({params}) ON CONFLICT ({conflict}) DO NOTHING"
    sets = [f"{_q(c)} = EXCLUDED.{_q(c)}" for c in update_cols]
    if touch_updated_at:
        sets.append('"updated_at" = NOW()')
    old = ", ".join(f"{_q(table)}.{_q(c)}" for c in update_cols)
    new = ", ".join(f"EXCLUDED.{_q(c)}" for c in update_cols)
    return (
        f"INSERT INTO {_q(table)} ({cols}) VALUES ({params}) "
        f"ON CONFLICT ({conflict}) DO UPDATE SET {', '.join(sets)} "
        f"WHERE ROW({old}) IS DISTINCT FROM ROW({new})"
    )


async def upsert_rows(
    conn: asyncpg.Connection,
    table: str,
    rows: Sequence[Mapping[str, Any]],
    conflict_columns: Sequence[str],
    *,
    touch_updated_at: bool = True,
) -> int:
    """Upsert ``rows`` (all with identical keys) into ``table``; returns rows submitted."""
    if not rows:
        return 0
    columns = list(rows[0].keys())
    for row in rows:
        if list(row.keys()) != columns:
            raise ValueError(f"inconsistent row shape for {table}: {list(row.keys())} != {columns}")
    sql = build_upsert_sql(table, columns, conflict_columns, touch_updated_at=touch_updated_at)
    await conn.executemany(sql, [tuple(r[c] for c in columns) for r in rows])
    return len(rows)

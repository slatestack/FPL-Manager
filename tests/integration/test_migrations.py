import asyncpg
import pytest

from app.db.migrate import MigrationError, discover, migrate


async def test_schema_contains_blueprint_and_ingestion_tables(pool):
    tables = {r["table_name"] for r in await pool.fetch(
        "SELECT table_name FROM information_schema.tables WHERE table_schema='public'"
    )}
    assert {"players", "player_predictions", "execution_logs", "system_config"} <= tables
    assert {"teams", "fixtures", "gameweeks", "player_gw_history", "fixture_odds", "player_xg_stats", "ingestion_runs"} <= tables


async def test_migrate_is_idempotent(dsn):
    assert await migrate(dsn) == []


async def test_modified_migration_is_detected(dsn, tmp_path):
    for version, sql, _ in discover():
        (tmp_path / f"{version}.sql").write_text(sql + "\n-- edited\n" if version.startswith("0001") else sql)
    with pytest.raises(MigrationError):
        await migrate(dsn, tmp_path)


async def test_blueprint_constraints_enforced(pool):
    with pytest.raises(asyncpg.CheckViolationError):
        await pool.execute(
            "INSERT INTO teams (team_id, code, name, short_name) VALUES (1, 3, 'Arsenal', 'ARS');"
            "INSERT INTO players (player_id, web_name, team_id, position, now_cost, status) VALUES (1, 'X', 1, 'GK', 50, 'a')"
        )

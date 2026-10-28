"""Migration runner against a real Postgres, in a throwaway database.

Skipped unless CAREPLAN_AI_TEST_POSTGRES_URL points at a server where the user
may create databases, e.g.
    CAREPLAN_AI_TEST_POSTGRES_URL=postgresql://postgres:root@localhost:5432/postgres
"""

import os
import uuid
from collections.abc import Iterator
from pathlib import Path

import psycopg
import pytest
from psycopg import sql

from careplan_ai.memory.migrate import DEFAULT_MIGRATIONS_DIR, migrate

ADMIN_URL = os.environ.get("CAREPLAN_AI_TEST_POSTGRES_URL")
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not ADMIN_URL, reason="CAREPLAN_AI_TEST_POSTGRES_URL not set"),
]


@pytest.fixture
def scratch_db() -> Iterator[str]:
    assert ADMIN_URL
    name = f"careplan_ai_test_{uuid.uuid4().hex[:8]}"
    with psycopg.connect(ADMIN_URL, autocommit=True) as conn:
        conn.execute(sql.SQL("create database {}").format(sql.Identifier(name)))
    try:
        yield psycopg.conninfo.make_conninfo(ADMIN_URL, dbname=name)
    finally:
        with psycopg.connect(ADMIN_URL, autocommit=True) as conn:
            conn.execute(sql.SQL("drop database {} with (force)").format(sql.Identifier(name)))


def test_applies_all_migrations_once(scratch_db: str) -> None:
    assert migrate(scratch_db) == ["0001_agent_memory"]
    assert migrate(scratch_db) == []

    with psycopg.connect(scratch_db) as conn:
        tables = {
            r[0]
            for r in conn.execute(
                "select table_name from information_schema.tables "
                "where table_schema = 'careplan_ai'"
            )
        }
        rls = conn.execute(
            "select relrowsecurity from pg_class where oid = 'careplan_ai.agent_sessions'::regclass"
        ).fetchone()
    assert {"agent_sessions", "care_plan_traces", "schema_migrations"} <= tables
    assert rls == (True,)


def test_refuses_an_edited_migration(scratch_db: str, tmp_path: Path) -> None:
    for f in DEFAULT_MIGRATIONS_DIR.glob("*.sql"):
        (tmp_path / f.name).write_text(f.read_text(encoding="utf-8"), encoding="utf-8")
    migrate(scratch_db, tmp_path)

    first = sorted(tmp_path.glob("*.sql"))[0]
    first.write_text(first.read_text(encoding="utf-8") + "\n-- edited\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="was changed after it was applied"):
        migrate(scratch_db, tmp_path)

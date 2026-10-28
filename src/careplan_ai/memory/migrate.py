"""Applies `migrations/*.sql` to the agent memory database, in order, once each.

The same runner is used everywhere, so local and Supabase schemas can't drift:

    local     python -m careplan_ai.memory.migrate --create-database
              (uses the CAREPLAN_AI_DB_* settings from .env -> local Postgres)
    Supabase  GitHub Actions (.github/workflows/db-migrate.yml) on push to main,
              with the Supabase URL from a repository secret. Never run it
              against Supabase by hand.

Each file runs in its own transaction and is recorded in
careplan_ai.schema_migrations with a checksum; an already-applied file that
has since been edited stops the run - add a new numbered file instead.

Deliberately imports nothing from the app (no Azure settings needed), so CI
only has to install psycopg and pydantic-settings.
"""

import argparse
import hashlib
import sys
from pathlib import Path

import psycopg
from psycopg import sql
from pydantic_settings import BaseSettings, SettingsConfigDict

from careplan_ai.memory.dsn import DbSettings, build_dsn

DEFAULT_MIGRATIONS_DIR = Path(__file__).resolve().parents[3] / "migrations"
# Arbitrary constant: stops two runners (e.g. overlapping CI jobs) racing.
_LOCK_ID = 7_351_001

_BOOKKEEPING_SQL = """
create schema if not exists careplan_ai;
create table if not exists careplan_ai.schema_migrations (
    version text primary key,
    checksum text not null,
    applied_at timestamptz not null default now()
);
"""


class _MigrationSettings(BaseSettings, DbSettings):
    """Only the DB settings (DB_* parts or MEMORY_DB_URL), from env or .env."""

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="CAREPLAN_AI_", env_file_encoding="utf-8", extra="ignore"
    )


def _checksum(text: str) -> str:
    return hashlib.sha256(text.replace("\r\n", "\n").encode("utf-8")).hexdigest()


def _create_database_if_missing(dsn: str) -> None:
    info = psycopg.conninfo.conninfo_to_dict(dsn)
    name = info.get("dbname")
    if not name:
        raise SystemExit("The database URL has no database name")
    admin_dsn = psycopg.conninfo.make_conninfo(dsn, dbname="postgres")
    with psycopg.connect(admin_dsn, autocommit=True, connect_timeout=10) as conn:
        exists = conn.execute("select 1 from pg_database where datname = %s", (name,)).fetchone()
        if not exists:
            conn.execute(sql.SQL("create database {}").format(sql.Identifier(str(name))))
            print(f"created database {name}")


def migrate(dsn: str, migrations_dir: Path = DEFAULT_MIGRATIONS_DIR) -> list[str]:
    """Apply pending migrations; returns the versions applied."""
    files = sorted(migrations_dir.glob("*.sql"))
    applied_now: list[str] = []
    with psycopg.connect(dsn, autocommit=True, connect_timeout=15) as conn:
        conn.execute("select pg_advisory_lock(%s)", (_LOCK_ID,))
        try:
            conn.execute(_BOOKKEEPING_SQL)
            done: dict[str, str] = dict(
                conn.execute("select version, checksum from careplan_ai.schema_migrations")
                .fetchall()
            )
            for path in files:
                version = path.stem
                text = path.read_text(encoding="utf-8")
                checksum = _checksum(text)
                if version in done:
                    if done[version] != checksum:
                        raise SystemExit(
                            f"{path.name} was changed after it was applied - "
                            "add a new migration file instead of editing this one"
                        )
                    continue
                with conn.transaction():
                    conn.execute(text)
                    conn.execute(
                        "insert into careplan_ai.schema_migrations (version, checksum) "
                        "values (%s, %s)",
                        (version, checksum),
                    )
                applied_now.append(version)
                print(f"applied {path.name}")
        finally:
            conn.execute("select pg_advisory_unlock(%s)", (_LOCK_ID,))
    if not applied_now:
        print("database is up to date")
    return applied_now


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--db-url",
        help="defaults to CAREPLAN_AI_MEMORY_DB_URL, else the CAREPLAN_AI_DB_* parts (env or .env)",
    )
    parser.add_argument("--dir", type=Path, default=DEFAULT_MIGRATIONS_DIR)
    parser.add_argument(
        "--create-database",
        action="store_true",
        help="create the database first if it doesn't exist (local Postgres)",
    )
    args = parser.parse_args(argv)

    dsn = args.db_url or build_dsn(_MigrationSettings())
    if not dsn:
        print(
            "No database configured: set CAREPLAN_AI_DB_HOST/DB_NAME (and DB_PASSWORD etc.), "
            "or CAREPLAN_AI_MEMORY_DB_URL, or pass --db-url",
            file=sys.stderr,
        )
        return 2

    if args.create_database:
        _create_database_if_missing(dsn)
    migrate(dsn, args.dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

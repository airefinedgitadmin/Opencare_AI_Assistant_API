from psycopg.conninfo import conninfo_to_dict
from pydantic import SecretStr

from careplan_ai.memory.dsn import DbSettings, build_dsn


def test_builds_from_separate_parts() -> None:
    dsn = build_dsn(
        DbSettings(
            db_host="localhost",
            db_port=5432,
            db_name="opencare-ai-dev",
            db_user="postgres",
            db_password=SecretStr("pa ss'word"),
        )
    )

    assert dsn is not None
    assert conninfo_to_dict(dsn) == {
        "host": "localhost",
        "port": "5432",
        "dbname": "opencare-ai-dev",
        "user": "postgres",
        "password": "pa ss'word",  # quoted safely
    }


def test_url_wins_over_parts() -> None:
    url = "postgresql://u:p@db.example.com:5432/x"

    assert build_dsn(DbSettings(memory_db_url=SecretStr(url), db_host="localhost",
                                db_name="y")) == url


def test_placeholder_url_falls_back_to_parts() -> None:
    dsn = build_dsn(DbSettings(memory_db_url=SecretStr("postgresql://<ref>@host/db"),
                               db_host="localhost", db_name="y", db_sslmode="require"))

    assert dsn is not None and conninfo_to_dict(dsn)["sslmode"] == "require"


def test_not_configured_returns_none() -> None:
    assert build_dsn(DbSettings()) is None
    assert build_dsn(DbSettings(db_host="localhost")) is None

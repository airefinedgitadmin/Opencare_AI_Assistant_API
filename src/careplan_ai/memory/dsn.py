"""Builds the agent memory database connection string from settings.

Two ways to configure it (both CAREPLAN_AI_-prefixed):

* separate parts - DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD, DB_SSLMODE
  (the usual local setup), or
* one MEMORY_DB_URL - takes precedence when set (GitHub Actions passes the
  Supabase connection string this way).

Kept free of app imports so the migration runner can use it in CI without any
Azure settings.
"""

from psycopg.conninfo import make_conninfo
from pydantic import BaseModel, SecretStr


class DbSettings(BaseModel):
    memory_db_url: SecretStr | None = None
    db_host: str | None = None
    db_port: int = 5432
    db_name: str | None = None
    db_user: str = "postgres"
    db_password: SecretStr | None = None
    # e.g. "require" for Supabase / Azure Postgres; leave unset for local.
    db_sslmode: str | None = None


def build_dsn(settings: DbSettings) -> str | None:
    """The usable connection string, or None when nothing (or only a
    <placeholder>) is configured."""
    if settings.memory_db_url is not None:
        url = settings.memory_db_url.get_secret_value().strip()
        if url and "<" not in url and "[" not in url:
            return url
    if not settings.db_host or not settings.db_name:
        return None
    parts: dict[str, str | int] = {
        "host": settings.db_host,
        "port": settings.db_port,
        "dbname": settings.db_name,
        "user": settings.db_user,
    }
    if settings.db_password is not None:
        parts["password"] = settings.db_password.get_secret_value()
    if settings.db_sslmode:
        parts["sslmode"] = settings.db_sslmode
    # make_conninfo quotes values safely (passwords with spaces/quotes etc.).
    return make_conninfo("", **parts)

"""Agent memory: drafting sessions and saved care plan traces.

Two backends behind one interface:

* PostgresStore - used whenever CAREPLAN_AI_MEMORY_DB_URL is set (Supabase
  now; any Postgres, e.g. Azure Database for PostgreSQL, later - only the
  connection string changes).
* InMemoryStore - tests and local dev. Lost on restart and not shared between
  replicas, so never rely on it in a deployed environment.

What is stored (agents/care_plan/stored.py): no record text. A session keeps
references to the evidence (ids, backend record ids, keyed fingerprints) and
the suggestions and staff decisions, with all AI wording encrypted; the
backend sends the evidence again with every call. It is deleted
CAREPLAN_AI_AGENT_SESSION_TTL_HOURS after it starts. A trace keeps, for a saved
care plan, its statements (encrypted) and references to the evidence behind
them, so "View" works once the backend sends that evidence.

Tables live in their own `careplan_ai` schema with row level security on, so
Supabase's auto-generated public API can't read them; this service connects
as the database owner, which bypasses RLS. The schema is owned by
`migrations/*.sql` (see memory/migrate.py) - this module never creates or
alters tables.
"""

import asyncio
import json
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from typing import Protocol

from careplan_ai.agents.care_plan.schemas import CarePlanSession, CarePlanTrace
from careplan_ai.agents.care_plan.stored import (
    session_from_stored,
    session_to_stored,
    subject_of,
    trace_from_stored,
    trace_to_stored,
)
from careplan_ai.api.schemas.agent import StartSessionRequest
from careplan_ai.core.config import get_settings
from careplan_ai.core.exceptions import MemoryStoreError
from careplan_ai.core.logging import get_logger

logger = get_logger(__name__)


Subject = tuple[int, datetime | None]


class SessionStore(Protocol):
    backend: str

    async def save_session(self, session: CarePlanSession) -> None: ...

    async def get_session(
        self, session_id: str, evidence: StartSessionRequest | None
    ) -> CarePlanSession | None: ...

    async def session_subject(self, session_id: str) -> Subject | None: ...

    async def save_trace(self, trace: CarePlanTrace) -> None: ...

    async def get_trace(
        self, care_plan_id: int, evidence: StartSessionRequest | None
    ) -> CarePlanTrace | None: ...

    async def trace_subject(self, care_plan_id: int) -> Subject | None: ...


class InMemoryStore:
    """Holds exactly what the database would - references and sealed wording."""

    backend = "in-memory"

    def __init__(self) -> None:
        self._sessions: dict[str, str] = {}
        self._traces: dict[int, str] = {}

    async def save_session(self, session: CarePlanSession) -> None:
        self._sessions[session.session_id] = json.dumps(session_to_stored(session))

    async def get_session(
        self, session_id: str, evidence: StartSessionRequest | None
    ) -> CarePlanSession | None:
        raw = self._sessions.get(session_id)
        return session_from_stored(json.loads(raw), evidence) if raw else None

    async def session_subject(self, session_id: str) -> Subject | None:
        raw = self._sessions.get(session_id)
        return subject_of(json.loads(raw)) if raw else None

    async def save_trace(self, trace: CarePlanTrace) -> None:
        self._traces[trace.care_plan_id] = json.dumps(trace_to_stored(trace))

    async def get_trace(
        self, care_plan_id: int, evidence: StartSessionRequest | None
    ) -> CarePlanTrace | None:
        raw = self._traces.get(care_plan_id)
        return trace_from_stored(json.loads(raw), evidence) if raw else None

    async def trace_subject(self, care_plan_id: int) -> Subject | None:
        raw = self._traces.get(care_plan_id)
        return subject_of(json.loads(raw)) if raw else None


class PostgresStore:
    """Sync psycopg run in a worker thread: works on any event loop (psycopg's
    async driver can't use Windows' default loop) and is plenty for this
    traffic. One short connection per call (via the session pooler on Supabase)."""

    backend = "postgres"

    def __init__(self, dsn: str, session_ttl: timedelta) -> None:
        self._dsn = dsn
        self._ttl = session_ttl

    def _connect(self):  # type: ignore[no-untyped-def]
        import psycopg

        try:
            return psycopg.connect(self._dsn, connect_timeout=10, autocommit=True)
        except psycopg.Error as exc:
            logger.error("memory_store_connect_failed", error=type(exc).__name__)
            raise MemoryStoreError("Agent memory store is unavailable") from exc

    def _run(self, fn):  # type: ignore[no-untyped-def]
        import psycopg

        def wrapper():  # type: ignore[no-untyped-def]
            with self._connect() as conn:
                try:
                    return fn(conn)
                except (psycopg.errors.UndefinedTable, psycopg.errors.InvalidSchemaName) as exc:
                    logger.error("memory_store_not_migrated")
                    raise MemoryStoreError(
                        "Agent memory tables are missing - run "
                        "`python -m careplan_ai.memory.migrate`"
                    ) from exc
                except psycopg.Error as exc:
                    logger.error("memory_store_query_failed", error=type(exc).__name__)
                    raise MemoryStoreError("Agent memory store query failed") from exc

        return asyncio.to_thread(wrapper)

    async def save_session(self, session: CarePlanSession) -> None:
        expires_at = session.created_at + self._ttl

        def _save(conn) -> None:  # type: ignore[no-untyped-def]
            # Housekeeping piggybacks on writes - no scheduler needed.
            conn.execute("delete from careplan_ai.agent_sessions where expires_at < now()")
            conn.execute(
                """
                insert into careplan_ai.agent_sessions
                    (session_id, assessment_template_user_id, status, state,
                     created_at, updated_at, expires_at)
                values (%s, %s, %s, %s::jsonb, %s, %s, %s)
                on conflict (session_id) do update set
                    status = excluded.status,
                    state = excluded.state,
                    updated_at = excluded.updated_at
                """,
                (
                    session.session_id,
                    session.assessment_template_user_id,
                    session.status.value,
                    json.dumps(session_to_stored(session)),
                    session.created_at,
                    session.updated_at,
                    expires_at,
                ),
            )

        await self._run(_save)

    def _session_row(self, session_id: str):  # type: ignore[no-untyped-def]
        def _get(conn):  # type: ignore[no-untyped-def]
            # Expired sessions go on reads too, not only when another is saved.
            conn.execute("delete from careplan_ai.agent_sessions where expires_at < now()")
            row = conn.execute(
                "select state from careplan_ai.agent_sessions where session_id = %s",
                (session_id,),
            ).fetchone()
            return _as_dict(row[0]) if row else None

        return self._run(_get)

    async def get_session(
        self, session_id: str, evidence: StartSessionRequest | None
    ) -> CarePlanSession | None:
        raw = await self._session_row(session_id)
        return session_from_stored(raw, evidence) if raw else None

    async def session_subject(self, session_id: str) -> Subject | None:
        raw = await self._session_row(session_id)
        return subject_of(raw) if raw else None

    async def save_trace(self, trace: CarePlanTrace) -> None:
        def _save(conn) -> None:  # type: ignore[no-untyped-def]
            conn.execute(
                """
                insert into careplan_ai.care_plan_traces
                    (care_plan_id, session_id, assessment_template_user_id, trace, created_at)
                values (%s, %s, %s, %s::jsonb, %s)
                on conflict (care_plan_id) do update set
                    session_id = excluded.session_id,
                    trace = excluded.trace
                """,
                (
                    trace.care_plan_id,
                    trace.session_id,
                    trace.assessment_template_user_id,
                    json.dumps(trace_to_stored(trace)),
                    trace.created_at,
                ),
            )

        await self._run(_save)

    def _trace_row(self, care_plan_id: int):  # type: ignore[no-untyped-def]
        def _get(conn):  # type: ignore[no-untyped-def]
            row = conn.execute(
                "select trace from careplan_ai.care_plan_traces where care_plan_id = %s",
                (care_plan_id,),
            ).fetchone()
            return _as_dict(row[0]) if row else None

        return self._run(_get)

    async def get_trace(
        self, care_plan_id: int, evidence: StartSessionRequest | None
    ) -> CarePlanTrace | None:
        raw = await self._trace_row(care_plan_id)
        return trace_from_stored(raw, evidence) if raw else None

    async def trace_subject(self, care_plan_id: int) -> Subject | None:
        raw = await self._trace_row(care_plan_id)
        return subject_of(raw) if raw else None


def _as_dict(value: object) -> dict:  # type: ignore[type-arg]
    # psycopg returns jsonb as a dict already; tolerate text just in case.
    return json.loads(value) if isinstance(value, str) else value  # type: ignore[return-value]


@lru_cache
def get_session_store() -> SessionStore:
    settings = get_settings()
    dsn = settings.memory_db_dsn
    if dsn is None:
        if settings.environment in ("staging", "production"):
            logger.warning("agent_memory_in_memory", environment=settings.environment)
        return InMemoryStore()
    return PostgresStore(dsn, timedelta(hours=settings.agent_session_ttl_hours))


def utcnow() -> datetime:
    return datetime.now(UTC)

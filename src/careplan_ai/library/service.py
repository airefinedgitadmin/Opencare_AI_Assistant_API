"""Style library operations: index, remove, exclude, and find examples."""

from careplan_ai.agents.care_plan.evidence import to_sentences
from careplan_ai.core.config import get_settings
from careplan_ai.core.exceptions import SessionNotFoundError
from careplan_ai.core.logging import get_logger
from careplan_ai.library.anonymise import anonymise
from careplan_ai.library.models import (
    LibraryCarePlanIn,
    LibraryEntry,
    LibraryIndexResult,
    StyleExample,
    entry_id,
    pathway_key,
)
from careplan_ai.library.store import get_vector_store
from careplan_ai.memory.store import get_session_store, utcnow

logger = get_logger(__name__)

# Example text and retrieval query are kept short: style needs a sample, not
# the whole plan, and the query is only used to find similar plans.
_MAX_QUERY_CHARS = 1500


def _plain(text: str | None) -> str:
    return "\n".join(to_sentences(text))


def _entry(plan: LibraryCarePlanIn, home_name: str | None = None) -> LibraryEntry:
    names = {"resident_names": plan.resident_names, "other_names": plan.other_names}
    parts = []
    if plan.title:
        parts.append(f"Title: {anonymise(_plain(plan.title), **names)}")
    parts.append(f"Action:\n{anonymise(_plain(plan.action), **names)}")
    if plan.outcome:
        parts.append(f"Outcome:\n{anonymise(_plain(plan.outcome), **names)}")
    return LibraryEntry(
        id=entry_id(plan.care_plan_id),
        care_plan_id=plan.care_plan_id,
        text="\n".join(parts),
        pathway=plan.pathway,
        pathway_key=pathway_key(plan.pathway),
        assessment_template_id=plan.assessment_template_id,
        care_plan_date=plan.care_plan_date,
        author_role=plan.author_role,
        status=plan.status,
        source=plan.source,
        edited=plan.edited,
        indexed_at=utcnow().isoformat(),
        home_name=" ".join(home_name.split()) if home_name and home_name.strip() else None,
    )


def _normalise(text: str) -> str:
    return " ".join(_plain(text).split()).casefold()


def staff_changed(plan_text: str, ai_statements: list[str]) -> bool:
    """True when the saved plan isn't just the AI statements staff added -
    a statement is missing (reworded) or there is more than a few words of
    staff-written text around them."""
    remaining = _normalise(plan_text)
    for statement in ai_statements:
        norm = _normalise(statement)
        if norm not in remaining:
            return True
        remaining = remaining.replace(norm, " ", 1)
    return len(remaining.split()) > 3


async def _apply_ai_source(plans: list[LibraryCarePlanIn]) -> None:
    """A plan saved through the agent has a trace: mark it AI-assisted, and
    edited when staff changed the AI wording (ranked accordingly)."""
    sessions = get_session_store()
    for plan in plans:
        try:
            # Only the statements' wording is needed here, not the evidence.
            trace = await sessions.get_trace(plan.care_plan_id, None)
        except Exception as exc:  # noqa: BLE001 - provenance is a ranking hint only
            logger.warning("library_trace_lookup_failed", error=type(exc).__name__)
            return
        if trace is None:
            continue
        plan.source = "ai_assisted"
        actions = [s.text for s in trace.statements if s.field == "action"]
        outcomes = [s.text for s in trace.statements if s.field == "outcome"]
        plan.edited = (
            any(s.edited for s in trace.statements)
            or staff_changed(plan.action, actions)
            or staff_changed(plan.outcome or "", outcomes)
        )


async def mark_ai_assisted(home_id: int | None, care_plan_id: int, edited: bool) -> None:
    """Called when an agent session is finalized: if the plan was already
    indexed by the save hook (it usually is), record that it was AI-assisted.
    Never raises - the trace is saved either way."""
    store = get_vector_store()
    if store is None or home_id is None:
        return
    eid = entry_id(care_plan_id)
    try:
        existing = (await store.get(home_id, [eid])).get(eid)
        if existing is not None and existing.source != "ai_assisted":
            existing.source = "ai_assisted"
            existing.edited = edited
            await store.upsert(home_id, [existing])
    except Exception as exc:  # noqa: BLE001
        logger.warning("library_mark_ai_failed", home_id=home_id, error=type(exc).__name__)


async def index_care_plans(
    home_id: int, plans: list[LibraryCarePlanIn], *, home_name: str | None = None
) -> LibraryIndexResult:
    store = get_vector_store()
    if store is None:
        return LibraryIndexResult(enabled=False)
    await _apply_ai_source(plans)
    entries = [_entry(plan, home_name) for plan in plans]
    # An admin's exclusion survives the plan being edited and re-indexed.
    existing = await store.get(home_id, [e.id for e in entries])
    for entry in entries:
        if entry.id in existing and existing[entry.id].excluded:
            entry.excluded = True
    await store.upsert(home_id, entries)
    logger.info("library_indexed", home_id=home_id, count=len(entries))
    return LibraryIndexResult(
        enabled=True,
        indexed=[e.id for e in entries],
        excluded=[e.id for e in entries if e.excluded],
    )


async def remove_care_plan(home_id: int, care_plan_id: int) -> LibraryIndexResult:
    store = get_vector_store()
    if store is None:
        return LibraryIndexResult(enabled=False)
    await store.delete(home_id, [entry_id(care_plan_id)])
    return LibraryIndexResult(enabled=True)


async def set_excluded(home_id: int, care_plan_id: int, excluded: bool) -> LibraryIndexResult:
    store = get_vector_store()
    if store is None:
        return LibraryIndexResult(enabled=False)
    eid = entry_id(care_plan_id)
    if not await store.set_excluded(home_id, eid, excluded):
        raise SessionNotFoundError(f"Care plan {care_plan_id} is not in this home's library")
    logger.info("library_exclusion_set", home_id=home_id, entry_id=eid, excluded=excluded)
    return LibraryIndexResult(enabled=True, excluded=[eid] if excluded else [])


def _rank(example: StyleExample) -> int:
    # Staff-written first, then AI-assisted that staff edited, then unedited AI.
    if example.source == "staff_written":
        return 0
    return 1 if example.edited else 2


async def find_examples(
    home_id: int | None, pathway: str | None, query: str, *, resident_names: list[str]
) -> list[StyleExample]:
    """Top-N style examples from this home and pathway; [] when the library is
    off, the home or pathway is unknown, or the home has none yet (the draft
    then uses the neutral house style). Never raises - style is optional."""
    settings = get_settings()
    store = get_vector_store()
    if store is None or home_id is None or not pathway:
        return []
    # The query leaves this service (embedding), so it is anonymised too.
    safe_query = anonymise(f"{pathway}\n{query}", resident_names=resident_names)
    top_k = settings.library_top_k
    try:
        hits = await store.search(
            home_id,
            pathway_key=pathway_key(pathway),
            query=safe_query[:_MAX_QUERY_CHARS],
            top_k=top_k * 3,  # headroom for re-ranking by source
        )
    except Exception as exc:  # noqa: BLE001 - a library outage must not block drafting
        logger.warning("library_search_failed", home_id=home_id, error=type(exc).__name__)
        return []
    examples = [
        StyleExample(
            id=entry.id,
            care_plan_id=entry.care_plan_id,
            text=entry.text,
            source=entry.source,
            edited=entry.edited,
            score=score,
        )
        for entry, score in hits
    ]
    if not settings.library_include_unedited_ai:
        examples = [e for e in examples if _rank(e) < 2]
    examples.sort(key=lambda e: (_rank(e), -e.score))
    return examples[:top_k]

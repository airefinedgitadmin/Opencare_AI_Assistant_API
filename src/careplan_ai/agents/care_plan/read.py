"""Reader agents: nothing in the resident's record goes unread.

The care plan agent's pack (packs.py) holds what fits. Every other record is
split into chunks and each chunk is read by a reader agent - one structured
model call, all chunks in parallel - which classifies every record for this
pathway:

    new_requirement / update / status   -> relevant: handed to the care plan
                                           agent; a new_requirement must then
                                           be covered by an action or is shown
                                           word for word (validation.py)
    not_relevant                        -> listed for staff, with nothing lost

Code, not the model, keeps the ledger: every record must come back exactly
once. Records a reader skips are asked about once more; any still unread (or
whose reader failed) are reported as omitted so staff check them. Nothing runs
when everything already fits the pack.
"""

import asyncio
import time
from dataclasses import dataclass, field

from careplan_ai.agents.care_plan import llm
from careplan_ai.agents.care_plan.context import count_tokens
from careplan_ai.agents.care_plan.prompts import group_repeats
from careplan_ai.agents.care_plan.schemas import EvidenceItem, ScreenedRecord, SourceType
from careplan_ai.core.config import get_settings
from careplan_ai.core.logging import get_logger

logger = get_logger(__name__)

NOT_RELEVANT = "not_relevant"
_KIND_ORDER = {"new_requirement": 0, "update": 1, "status": 2}


@dataclass
class Reading:
    # What the readers made of each record they read, in evidence order.
    screened: list[ScreenedRecord] = field(default_factory=list)
    # Relevant records for the care plan agent, in evidence order.
    added: list[EvidenceItem] = field(default_factory=list)
    # new_requirement records (current only), one group per repeated record:
    # an action must cite one of each group's ids.
    required: list[list[str]] = field(default_factory=list)
    # Records no reader read.
    unread: list[str] = field(default_factory=list)
    calls: int = 0
    seconds: float = 0.0


def _context(shown: list[EvidenceItem], budget: int) -> list[EvidenceItem]:
    """What the pathway already says - its assessment answers and previous
    actions - for judging relevance. Context only, so it is kept short."""
    context: list[EvidenceItem] = []
    for item in shown:
        if item.source_type == SourceType.ASSESSMENT or (
            item.source_type == SourceType.PREVIOUS_CARE_PLAN and item.id.startswith("P")
        ):
            cost = count_tokens(item)
            if cost > budget:
                break
            context.append(item)
            budget -= cost
    return context


def _chunks(groups: list[list[EvidenceItem]], chunk_tokens: int) -> list[list[list[EvidenceItem]]]:
    chunks: list[list[list[EvidenceItem]]] = []
    size = 0
    for group in groups:
        cost = count_tokens(group[0]) + 2 * (len(group) - 1)  # the extra ids and dates
        if chunks and size + cost <= chunk_tokens:
            chunks[-1].append(group)
            size += cost
        else:
            chunks.append([group])
            size = cost
    return chunks


async def _read_chunk(
    chunk: list[list[EvidenceItem]],
    *,
    resident_name: str | None,
    pathway: str | None,
    context: list[EvidenceItem],
    limit: asyncio.Semaphore,
    reading: Reading,
) -> dict[str, tuple[str, str]]:
    """First id of each group -> (kind, reason). Missing = unread."""
    found: dict[str, tuple[str, str]] = {}
    left = chunk
    for _attempt in range(2):  # skipped lines are asked about once more
        owner = {item.id: group[0].id for group in left for item in group}
        try:
            async with limit:
                reading.calls += 1
                out = await llm.read_records(
                    resident_name=resident_name,
                    pathway=pathway,
                    context=context,
                    records=[item for group in left for item in group],
                )
        except Exception as exc:  # noqa: BLE001 - reported as unread, never fails the draft
            logger.warning("care_plan_reader_failed", records=len(owner), error=type(exc).__name__)
            break
        for rel in out.relevant:
            first = owner.get(rel.evidence_id.strip())
            if first is not None:
                # Read more than once: the most demanding kind wins.
                old = found.get(first)
                if old is None or _KIND_ORDER[rel.kind] < _KIND_ORDER.get(old[0], 9):
                    found[first] = (rel.kind, " ".join(rel.reason.split()))
        for raw in out.not_relevant:
            first = owner.get(raw.strip())
            if first is not None and first not in found:  # unsure = relevant wins
                found[first] = (NOT_RELEVANT, "")
        left = [group for group in left if group[0].id not in found]
        if not left:
            break
    return found


async def read_rest(
    evidence: list[EvidenceItem],
    shown: list[EvidenceItem],
    *,
    resident_name: str | None,
    pathway: str | None,
) -> Reading:
    """Reads every record not in `shown`, in parallel chunks."""
    settings = get_settings()
    shown_ids = {e.id for e in shown}
    rest = [e for e in evidence if e.id not in shown_ids]
    reading = Reading()
    if not rest:
        return reading
    if not settings.agent_read_all:
        reading.unread = [e.id for e in rest]
        return reading

    started = time.monotonic()
    groups = group_repeats(rest)
    context = _context(shown, settings.agent_reader_context_tokens)
    limit = asyncio.Semaphore(max(1, settings.agent_reader_concurrency))
    results = await asyncio.gather(*(
        _read_chunk(chunk, resident_name=resident_name, pathway=pathway, context=context,
                    limit=limit, reading=reading)
        for chunk in _chunks(groups, settings.agent_reader_chunk_tokens)
    ))
    verdicts = {first: verdict for found in results for first, verdict in found.items()}

    kinds: dict[str, tuple[str, str]] = {}
    for group in groups:
        verdict = verdicts.get(group[0].id)
        if verdict is None:
            reading.unread += [e.id for e in group]
            continue
        kind, reason = verdict
        if kind == "new_requirement" and (group[0].historical or group[0].supporting):
            # History explains, and a supporting chart is context (care team
            # rules): neither becomes a need on its own.
            kind = "status"
        if kind == "new_requirement":
            reading.required.append([e.id for e in group])
        for item in group:
            kinds[item.id] = (kind, reason)

    # Care needs always go to the care plan agent; information-only records
    # while the budget lasts (the rest stay searchable, and are listed).
    budget = settings.agent_reader_added_tokens
    added: set[str] = set()
    for group in sorted(
        (g for g in groups if kinds.get(g[0].id, (NOT_RELEVANT,))[0] != NOT_RELEVANT),
        key=lambda g: _KIND_ORDER[kinds[g[0].id][0]],
    ):
        cost = sum(count_tokens(e) for e in group)
        if kinds[group[0].id][0] != "status":
            added.update(e.id for e in group)
        elif cost <= budget:
            added.update(e.id for e in group)
            budget -= cost

    reading.screened = [
        ScreenedRecord(evidence_id=e.id, kind=kinds[e.id][0], reason=kinds[e.id][1])
        for e in rest if e.id in kinds
    ]
    reading.added = [e for e in rest if e.id in added]
    order = {e.id: i for i, e in enumerate(evidence)}
    reading.unread.sort(key=order.__getitem__)
    reading.seconds = round(time.monotonic() - started, 2)
    logger.info(
        "care_plan_records_read",
        records=len(rest),
        calls=reading.calls,
        relevant=sum(1 for s in reading.screened if s.kind != NOT_RELEVANT),
        added=len(reading.added),
        required=len(reading.required),
        unread=len(reading.unread),
        seconds=reading.seconds,
    )
    return reading

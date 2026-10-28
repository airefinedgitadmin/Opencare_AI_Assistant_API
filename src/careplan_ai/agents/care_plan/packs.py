"""The care plan agent's starting evidence - plain code, deterministic.

Long prompts make a model skip a different item each run, so instead of
handing it everything, it gets:

1. every must-read record (assessment, previous care plans, notes written
   against this pathway) - refused, never cut, if these alone are too large;
2. for each need among them (risk answer, previous action, pathway note), its
   best keyword matches across ALL the resident's other records - current
   notes and history - taken in turns so every need gets its best match first;
3. then the most pathway-related remaining notes, newest first, to the budget.

Everything not in the pack stays reachable through the agent's
search_records tool; records it never sees are reported on the session.
"""

from dataclasses import dataclass

from careplan_ai.agents.care_plan.context import (
    count_tokens,
    is_must_read,
    rank_for_pathway,
    total_tokens,
)
from careplan_ai.agents.care_plan.schemas import EvidenceItem, SourceType
from careplan_ai.agents.care_plan.search import RecordIndex
from careplan_ai.core.exceptions import EvidenceTooLargeError


@dataclass
class Pack:
    shown: list[EvidenceItem]  # in evidence order
    index: RecordIndex  # over ALL records, for the search tool


def _needs(must_read: list[EvidenceItem]) -> list[EvidenceItem]:
    return [
        e for e in must_read
        if e.source_type == SourceType.ASSESSMENT
        or (e.source_type == SourceType.PREVIOUS_CARE_PLAN and e.id.startswith("P"))
        or e.is_note
    ]


def build_pack(
    evidence: list[EvidenceItem],
    *,
    pathway: str | None,
    max_must_read_tokens: int,
    pack_tokens: int,
    matches_per_need: int,
) -> Pack:
    must_read = [e for e in evidence if is_must_read(e)]
    must_tokens = total_tokens(must_read)
    if must_tokens > max_must_read_tokens:
        raise EvidenceTooLargeError(
            f"The assessment, previous care plans and pathway notes need about {must_tokens} "
            f"tokens, over the {max_must_read_tokens} limit - they are never cut, so this "
            "resident's history is too large to draft from in one go"
        )
    others = [e for e in evidence if not is_must_read(e)]
    others_index = RecordIndex(others)

    # Each need's matches, ranked; then taken round-robin (rank 1 of every need,
    # then rank 2, ...) so no need is crowded out by another's matches.
    per_need = [
        [item for item, _ in others_index.search(f"{n.label} {n.text}", k=matches_per_need)]
        for n in _needs(must_read)
    ]
    ordered: list[EvidenceItem] = []
    seen: set[str] = set()
    for rank in range(matches_per_need):
        for matches in per_need:
            if rank < len(matches) and matches[rank].id not in seen:
                seen.add(matches[rank].id)
                ordered.append(matches[rank])
    ordered += [e for e in rank_for_pathway(must_read, others, pathway) if e.id not in seen]

    budget = max(pack_tokens - must_tokens, 0)
    chosen = {e.id for e in must_read}
    for item in ordered:
        cost = count_tokens(item)
        if cost <= budget:
            chosen.add(item.id)
            budget -= cost
    return Pack(shown=[e for e in evidence if e.id in chosen], index=RecordIndex(evidence))

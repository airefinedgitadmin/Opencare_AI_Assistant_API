"""Shared helpers for what the care plan agent reads (packs.py builds the pack):
token estimates, date order, what is must-read, and pathway ranking.

* must-read: the assessment, previous care plans and notes written against
  this pathway - always read, never cut.
* the rest: notes on the resident's record only (most carer notes) and history
  (older notes, archived / other-pathway care plans) - matched to each need,
  ranked by pathway, and searchable by the agent.
"""

import re
from datetime import date, datetime

import tiktoken

from careplan_ai.agents.care_plan.prompts import format_evidence_line
from careplan_ai.agents.care_plan.schemas import EvidenceItem, SourceType
from careplan_ai.agents.care_plan.validation import content_stems

# Estimate only - consistent and close enough, with headroom in the budgets.
_ENCODING = tiktoken.get_encoding("cl100k_base")
_DATE_FORMATS = ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y")


def count_tokens(item: EvidenceItem) -> int:
    return len(_ENCODING.encode(format_evidence_line(item))) + 1


def total_tokens(items: list[EvidenceItem]) -> int:
    return sum(count_tokens(item) for item in items)


def _parse_date(value: str | None) -> date:
    if value:
        head = re.split(r"[ T]", value.strip())[0]
        for fmt in _DATE_FORMATS:
            try:
                return datetime.strptime(head, fmt).date()
            except ValueError:
                continue
    return date.min  # undated notes count as oldest


def newest_first(items: list[EvidenceItem]) -> list[EvidenceItem]:
    # Stable: same-day notes keep the backend's order.
    return sorted(items, key=lambda item: _parse_date(item.date), reverse=True)


def is_must_read(item: EvidenceItem) -> bool:
    if item.historical:
        return False  # background - matched and searched like resident-level notes
    return not item.is_note or item.linked_to_pathway


def rank_for_pathway(
    must_read: list[EvidenceItem], candidates: list[EvidenceItem], pathway: str | None
) -> list[EvidenceItem]:
    """Candidates that share a content word with this pathway (its name, its
    assessment questions and answers, previous plans, pathway notes) first,
    the rest after; newest first within each group. Order only - nothing is
    removed here."""
    vocabulary = content_stems(
        [pathway or ""]
        + [e.label for e in must_read if e.source_type == SourceType.ASSESSMENT]
        + [e.text for e in must_read]
    )
    ordered = newest_first(candidates)
    return sorted(ordered, key=lambda e: not (content_stems([e.text]) & vocabulary))

"""Turns the backend's evidence pack into numbered, quotable evidence items.

Deterministic - no model involved. Ids are short and typed so both the model
and the code checks can tell sources apart at a glance:

    A1..  assessment answers
    P1..  previous care plan ACTION sentences (each must be re-suggested or
          explicitly superseded by a note - deck slides 7 and 9)
    O1..  previous care plan OUTCOME sentences (support for new outcomes)
    R1..  review note sentences
    C1..  carer (daily) note sentences
    N1..  nursing note sentences
    G1..  external professional note sentences (GP, district nurse, ...)
    F1..  relative communication sentences
    H1..  related care plan sentences (this pathway's archived plans, other
          pathways' plans) - history: context and patterns, never required
    V1..  observation chart entries (fluid, food, TPR, falls diary, ...) - one
          item per entry, its fields as "Field: value; ..."; supporting only
    T1..  one summary line per chart, computed here from its entries (count,
          days, each number's range and latest value) - so a long chart is
          always represented exactly, whatever is read of its entries

Nothing is capped or clipped here: deck slide 20 says no information may be
removed or skipped. What the model reads is decided by token budget in
context.py, and anything it can't read is reported, not dropped.
"""

import re
from html.parser import HTMLParser

from careplan_ai.agents.care_plan.schemas import EvidenceItem, SourceType
from careplan_ai.api.schemas.agent import (
    NoteIn,
    ObservationIn,
    ReviewNoteIn,
    StartSessionRequest,
)

_NOTE_SOURCES: dict[str, tuple[SourceType, str]] = {
    "carer": (SourceType.CARER_NOTE, "C"),
    "nursing": (SourceType.NURSING_NOTE, "N"),
    "professional": (SourceType.PROFESSIONAL_NOTE, "G"),
    "relative": (SourceType.RELATIVE_NOTE, "F"),
}

_BLOCK_TAGS = {"p", "div", "li", "br", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6", "tr"}
# Split after . ! ? when the next sentence starts with a capital, digit or quote.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'“(])")
_WHITESPACE = re.compile(r"\s+")


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def to_sentences(text: str | None) -> list[str]:
    """Plain text or Quill HTML -> clean sentences, in order.

    Paragraphs and list items are kept apart even when they lack a full stop
    (Quill bullets often do)."""
    if not text:
        return []
    if "<" in text and ">" in text:
        extractor = _TextExtractor()
        extractor.feed(text)
        extractor.close()
        text = "".join(extractor.parts)
    sentences: list[str] = []
    for line in text.replace("\r", "\n").split("\n"):
        line = _WHITESPACE.sub(" ", line).strip()
        if not line:
            continue
        sentences.extend(s.strip() for s in _SENTENCE_BOUNDARY.split(line) if len(s.strip()) > 2)
    return sentences


def _note_sentences(note: ReviewNoteIn | NoteIn) -> list[str]:
    sentences = to_sentences(note.note)
    # A title can carry facts of its own ("Fall in the lounge"); keep it unless
    # the note already says it.
    title = " ".join((note.title or "").split())
    if len(title) > 2 and title.casefold() not in " ".join(sentences).casefold():
        sentences.insert(0, title)
    return sentences


def _note_label(note: NoteIn) -> str:
    if note.note_type == "professional":
        return f"{(note.professional_type or 'External professional').strip()} note"
    if note.note_type == "relative":
        return f"Relative ({note.relation.strip()})" if note.relation else "Relative"
    return {"carer": "Carer note", "nursing": "Nursing note"}[note.note_type]


def unanswered_questions(request: StartSessionRequest) -> list[str]:
    if request.assessment is None:
        return []
    return [
        a.question.strip()
        for a in request.assessment.answers
        if (a.answer is None or not a.answer.strip()) and a.question.strip()
    ]


def build_evidence(request: StartSessionRequest) -> list[EvidenceItem]:
    items: list[EvidenceItem] = []

    a_count = 0
    if request.assessment is not None:
        for answer in request.assessment.answers:
            if answer.answer is None or not answer.answer.strip():
                continue  # an unanswered question can't flag a risk
            a_count += 1
            items.append(
                EvidenceItem(
                    id=f"A{a_count}",
                    source_type=SourceType.ASSESSMENT,
                    source_ref=answer.question_id,
                    label=answer.question.strip(),
                    date=request.assessment.completed_date,
                    text=answer.answer.strip(),
                )
            )

    p_count = o_count = 0
    for plan in request.previous_care_plans:
        ref = str(plan.care_plan_id) if plan.care_plan_id is not None else None
        for sentence in to_sentences(plan.action):
            p_count += 1
            items.append(
                EvidenceItem(
                    id=f"P{p_count}",
                    source_type=SourceType.PREVIOUS_CARE_PLAN,
                    source_ref=ref,
                    label="Previous care plan - action",
                    date=plan.care_plan_date,
                    text=sentence,
                )
            )
        for sentence in to_sentences(plan.outcome):
            o_count += 1
            items.append(
                EvidenceItem(
                    id=f"O{o_count}",
                    source_type=SourceType.PREVIOUS_CARE_PLAN,
                    source_ref=ref,
                    label="Previous care plan - outcome",
                    date=plan.care_plan_date,
                    text=sentence,
                )
            )

    r_count = 0
    for review in request.review_notes:
        for sentence in _note_sentences(review):
            r_count += 1
            items.append(
                EvidenceItem(
                    id=f"R{r_count}",
                    source_type=SourceType.REVIEW_NOTE,
                    source_ref=str(review.note_id) if review.note_id is not None else None,
                    label="Review note",
                    date=review.created_date,
                    text=sentence,
                    linked_to_pathway=review.pathway_linked,
                    historical=review.historical,
                )
            )

    counts = {prefix: 0 for _, prefix in _NOTE_SOURCES.values()}
    for note in request.notes:
        source_type, prefix = _NOTE_SOURCES[note.note_type]
        for sentence in _note_sentences(note):
            counts[prefix] += 1
            items.append(
                EvidenceItem(
                    id=f"{prefix}{counts[prefix]}",
                    source_type=source_type,
                    source_ref=str(note.note_id) if note.note_id is not None else None,
                    label=_note_label(note),
                    date=note.visit_date or note.created_date,
                    text=sentence,
                    linked_to_pathway=note.pathway_linked,
                    historical=note.historical,
                )
            )

    h_count = 0
    for related in request.related_care_plans:
        ref = str(related.care_plan_id) if related.care_plan_id is not None else None
        what = ", ".join(p for p in (related.pathway, related.status) if p)
        for part, text in (("action", related.action), ("outcome", related.outcome)):
            for sentence in to_sentences(text):
                h_count += 1
                items.append(
                    EvidenceItem(
                        id=f"H{h_count}",
                        source_type=SourceType.RELATED_CARE_PLAN,
                        source_ref=ref,
                        label=f"Earlier care plan ({what}) - {part}" if what
                        else f"Earlier care plan - {part}",
                        date=related.care_plan_date,
                        text=sentence,
                        linked_to_pathway=False,
                        historical=True,
                    )
                )

    v_count = 0
    recorded: list[ObservationIn] = []
    for entry in request.observations:
        text = observation_text(entry)
        if not text:
            continue  # an empty entry records nothing
        v_count += 1
        recorded.append(entry)
        items.append(
            EvidenceItem(
                id=f"V{v_count}",
                source_type=SourceType.OBSERVATION,
                source_ref=entry.entry_id,
                label=_chart_name(entry),
                date=entry.recorded_date,
                text=text,
                linked_to_pathway=entry.pathway_linked,
                historical=entry.historical,
                supporting=entry.is_supporting,
            )
        )

    return items + chart_summaries(recorded)


def _chart_name(entry: ObservationIn) -> str:
    return entry.observation_name.strip() or "Observation"


_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def _day(value: str | None) -> tuple[int, int, int] | None:
    """(year, month, day) of a 'dd/MM/yyyy' date, for ordering."""
    parts = (value or "").strip()[:10].split("/")
    if len(parts) == 3 and all(p.isdigit() for p in parts):
        return int(parts[2]), int(parts[1]), int(parts[0])
    return None


def chart_summaries(entries: list[ObservationIn]) -> list[EvidenceItem]:
    """One line per chart (current entries and history apart), from 2+ entries:
    how many, on how many days, and for every numeric field its lowest and
    highest value and the latest one - figures exactly as recorded, nothing
    added up or averaged, so the line is as quotable as the entries."""
    charts: dict[tuple[str, bool], list[ObservationIn]] = {}
    for entry in entries:
        charts.setdefault((_chart_name(entry), entry.historical), []).append(entry)

    summaries: list[EvidenceItem] = []
    for (name, historical), group in charts.items():
        if len(group) < 2:
            continue  # one entry says it all
        # Stable: same-day entries keep the backend's (oldest first) order.
        ordered = sorted(group, key=lambda e: _day(e.recorded_date) or (0, 0, 0))
        dated = [e.recorded_date for e in ordered if _day(e.recorded_date)]
        days = len({_day(d) for d in dated})
        parts = [f"{len(group)} entries"]
        if dated:
            span = dated[0] if dated[0] == dated[-1] else f"{dated[0]} to {dated[-1]}"
            parts[0] += f" on {days} day{'s' if days != 1 else ''} ({span})"

        figures: dict[str, list[str]] = {}
        for entry in ordered:
            for v in entry.values:
                value = _WHITESPACE.sub(" ", v.value or "").strip()
                field = _WHITESPACE.sub(" ", v.field).strip()
                if field and _NUMBER.fullmatch(value):
                    figures.setdefault(field, []).append(value)
        for field, values in figures.items():
            low = min(values, key=float)
            high = max(values, key=float)
            if float(low) == float(high):
                parts.append(f"{field}: {values[-1]} every time")
            else:
                parts.append(f"{field}: {low} to {high}, latest {values[-1]}")

        summaries.append(
            EvidenceItem(
                id=f"T{len(summaries) + 1}",
                source_type=SourceType.OBSERVATION_SUMMARY,
                label=name,
                date=dated[-1] if dated else None,
                text="; ".join(parts),
                linked_to_pathway=any(e.pathway_linked for e in group),
                historical=historical,
                supporting=all(e.is_supporting for e in group),
            )
        )
    return summaries


def observation_text(entry: ObservationIn) -> str:
    """'Field: value; Field: value' for the fields that were filled in."""
    parts = []
    for v in entry.values:
        value = _WHITESPACE.sub(" ", v.value or "").strip()
        field = _WHITESPACE.sub(" ", v.field).strip()
        if value:
            parts.append(f"{field}: {value}" if field else value)
    return "; ".join(parts)

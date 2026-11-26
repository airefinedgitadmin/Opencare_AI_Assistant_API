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
          item per entry: its date first (dd/mm/yyyy), then "Field: value · ..."; supporting only
    T1..  one summary line per chart, computed here from its entries (count,
          days, each number's range and latest value) - so a long chart is
          always represented exactly, whatever is read of its entries
    K1..  the care standard a recorded score calls for (standards.py) - e.g.
          the steps for the MUST band in the assessment

Nothing is capped or clipped here: deck slide 20 says no information may be
removed or skipped. What the model reads is decided by token budget in
context.py, and anything it can't read is reported, not dropped.
"""

import re
from datetime import datetime
from html.parser import HTMLParser
from zoneinfo import ZoneInfo

from careplan_ai.agents.care_plan.schemas import EvidenceItem, SourceType
from careplan_ai.agents.care_plan.standards import care_standards, daily_target
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
# OPENCARE-80: text staff typed as a placeholder ("test", "tbc", "xxx", "asdf") is not a fact
# about the resident - it is never evidence, so it can't be cited or carried over.
_PLACEHOLDER_TEXT = re.compile(
    r"(?:test(?:ing)?\s*\d*|tbc|tba|todo|to do|xx+|asdf\w*|qwerty|dummy|lorem ipsum\b.*)[.!?]*",
    re.IGNORECASE,
)


# OPENCARE-78: words that say nothing when they are the whole of a care plan or note line. Not
# applied to assessment answers or chart values, where "None" or "N/A" is a real answer.
_EMPTY_FREE_TEXT = re.compile(r"(?:n\s*/\s*a|na|ok|nil|none)[.!?]*|[\W_]+", re.IGNORECASE)


def is_empty_free_text(sentence: str) -> bool:
    """Placeholder text, or a care plan / note line that is only "n/a", "ok", "nil", "none" or
    punctuation ("...", "-")."""
    text = " ".join(sentence.split())
    return is_placeholder(text) or bool(_EMPTY_FREE_TEXT.fullmatch(text))


def is_placeholder(sentence: str) -> bool:
    """A short sentence that says nothing ("test", "Test.", "tbc", "xxx"). Placeholders are short;
    anything longer is real text and is never dropped."""
    text = " ".join(sentence.split())
    return len(text) <= 20 and bool(_PLACEHOLDER_TEXT.fullmatch(text))


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


def to_sentences(
    text: str | None, *, drop_placeholders: bool = False, dropped: list[str] | None = None
) -> list[str]:
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
        for s in _SENTENCE_BOUNDARY.split(line):
            s = s.strip()
            if len(s) <= 2:
                continue
            if drop_placeholders and is_empty_free_text(s):
                if dropped is not None:
                    dropped.append(s)
                continue
            sentences.append(s)
    return sentences


def _note_sentences(note: ReviewNoteIn | NoteIn, dropped: list[str] | None = None) -> list[str]:
    sentences = to_sentences(note.note, drop_placeholders=True, dropped=dropped)
    # A title can carry facts of its own ("Fall in the lounge"); keep it unless
    # the note already says it - or it is placeholder text (OPENCARE-80).
    title = " ".join((note.title or "").split())
    if len(title) > 2 and is_empty_free_text(title):
        if dropped is not None:
            dropped.append(title)
    elif len(title) > 2 and title.casefold() not in " ".join(sentences).casefold():
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


def build_evidence(
    request: StartSessionRequest, dropped_records: set[str] | None = None
) -> list[EvidenceItem]:
    """dropped_records, if given, collects one key per record that had placeholder text left out
    (OPENCARE-80) - so staff are told how many records, counted by the same rule."""
    items: list[EvidenceItem] = []

    def note_drop(key: str, dropped: list[str]) -> None:
        if dropped and dropped_records is not None:
            dropped_records.add(key)

    a_count = 0
    if request.assessment is not None:
        for answer in request.assessment.answers:
            if answer.answer is None or not answer.answer.strip():
                continue  # an unanswered question can't flag a risk
            if is_placeholder(answer.answer):
                note_drop(f"answer:{answer.question_id or answer.question}", [answer.answer])
                continue  # "test" / "tbc" says nothing about the resident (OPENCARE-80)
            a_count += 1
            items.append(
                EvidenceItem(
                    id=f"A{a_count}",
                    source_type=SourceType.ASSESSMENT,
                    source_ref=answer.question_id,
                    label=answer.question.strip(),
                    date=request.assessment.completed_date,
                    text=answer.answer.strip(),
                    risk_flagged=answer.risk_flagged,
                    risk_rule=answer.risk_rule,
                )
            )

    p_count = o_count = 0
    for plan in request.previous_care_plans:
        ref = str(plan.care_plan_id) if plan.care_plan_id is not None else None
        plan_dropped: list[str] = []
        for sentence in to_sentences(plan.action, drop_placeholders=True, dropped=plan_dropped):
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
        for sentence in to_sentences(plan.outcome, drop_placeholders=True, dropped=plan_dropped):
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
        note_drop(f"plan:{ref}:{id(plan)}", plan_dropped)

    r_count = 0
    for review in request.review_notes:
        review_dropped: list[str] = []
        review_sentences = _note_sentences(review, review_dropped)
        note_drop(f"note:{review.note_id}:{id(review)}", review_dropped)
        for sentence in review_sentences:
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
        note_dropped: list[str] = []
        sentences = _note_sentences(note, note_dropped)
        note_drop(f"note:{note.note_id}:{id(note)}", note_dropped)
        for sentence in sentences:
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
        related_dropped: list[str] = []
        for part, text in (("action", related.action), ("outcome", related.outcome)):
            for sentence in to_sentences(text, drop_placeholders=True, dropped=related_dropped):
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
        note_drop(f"plan:{ref}:{id(related)}", related_dropped)

    v_count = 0
    recorded: list[ObservationIn] = []
    # OPENCARE-99: a value every entry of a chart repeats (the daily target, "Record type:
    # Intake") is said once, in the chart's summary line, not on each entry; and an amount
    # counted against a daily target carries the day's running total.
    repeated = _repeated_fields(request.observations)
    running = _running_totals(request.observations)
    chart_types: dict[str, str | None] = {}
    chart_fields: dict[str, list[tuple[str, str]]] = {}
    for entry in request.observations:
        entry_dropped: list[str] = []
        text = observation_text(
            entry, entry_dropped,
            leave_out=repeated.get((_chart_name(entry), entry.historical), set()),
            running=running.get(id(entry)),
        )
        note_drop(f"observation:{entry.entry_id}:{id(entry)}", entry_dropped)
        if not text:
            continue  # an empty entry records nothing
        v_count += 1
        recorded.append(entry)
        chart_types[f"V{v_count}"] = entry.observation_type
        chart_fields[f"V{v_count}"] = _fields(entry)
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

    items += chart_summaries(recorded)
    # OPENCARE-99/115/116: the care the recorded score, level or reading calls for - K items.
    return items + care_standards(
        items,
        pathway=request.pathway,
        resident_name=request.resident_name,
        chart_types=chart_types,
        chart_fields=chart_fields,
        below_target=below_target(recorded),
    )


def below_target(entries: list[ObservationIn]) -> set[str]:
    """OPENCARE-116: current charts whose latest chart day adds up to less than their daily
    target - the amounts as recorded, added up per day as the chart summary does."""
    low: set[str] = set()
    for (name, historical), group in _charts(entries).items():
        if historical:
            continue
        target = _chart_target(group)
        if target is None:
            continue
        _field, amount, unit = target
        per_day: dict[tuple[int, int, int], float] = {}
        for entry in group:
            day = _entry_sort_key(entry)[0]
            fields = dict(_fields(entry))
            for field in _counted_fields(entry, unit):
                per_day[day] = per_day.get(day, 0.0) + float(fields[field])
        if per_day and per_day[max(per_day)] < amount:
            low.add(name)
    return low


def _chart_name(entry: ObservationIn) -> str:
    return entry.observation_name.strip() or "Observation"


def _clean(text: str | None) -> str:
    return _WHITESPACE.sub(" ", text or "").strip()


def _fields(entry: ObservationIn) -> list[tuple[str, str]]:
    """The entry's filled-in (field, value) pairs, as recorded."""
    return [(_clean(v.field), _clean(v.value)) for v in entry.values if _clean(v.value)]


def _charts(entries: list[ObservationIn]) -> dict[tuple[str, bool], list[ObservationIn]]:
    """Entries by chart - current and history apart, as the summary lines are."""
    charts: dict[tuple[str, bool], list[ObservationIn]] = {}
    for entry in entries:
        charts.setdefault((_chart_name(entry), entry.historical), []).append(entry)
    return charts


def _is_entry_when(field: str, value: str) -> bool:
    """The field is the entry's own date and time ("Date & Time"), not another date."""
    return (readable_date(value) != value and bool(_ENTRY_WHEN.search(field))
            and not _OTHER_DATE.search(field))


def _repeated_fields(entries: list[ObservationIn]) -> dict[tuple[str, bool], set[str]]:
    """Per chart of 2+ entries: the fields every entry has with the same value."""
    repeated: dict[tuple[str, bool], set[str]] = {}
    for key, group in _charts(entries).items():
        if len(group) < 2:
            continue
        common = dict(_fields(group[0]))
        for entry in group[1:]:
            values = dict(_fields(entry))
            common = {f: v for f, v in common.items() if values.get(f) == v}
        same = {f for f, v in common.items() if f and not _is_entry_when(f, v)}
        if same:
            repeated[key] = same
    return repeated


def _entry_sort_key(entry: ObservationIn) -> tuple[tuple[int, int, int], str]:
    """(day, time) of an entry: its own Date & Time, else the date it was recorded."""
    for field, value in _fields(entry):
        if _is_entry_when(field, value):
            shown = readable_date(value)
            day = _day(shown) or (0, 0, 0)
            return day, shown[11:16]
    return _day(entry.recorded_date) or (0, 0, 0), ""


def _counted_fields(entry: ObservationIn, unit: str) -> list[str]:
    """Amount fields counted against a daily target in `unit`: same unit, not the target, a total
    or an output ("Current intake (ml)", not "Output (ml)"). An output entry ("Record type:
    Output") counts nothing towards intake."""
    if any(re.fullmatch(r"output", v, re.IGNORECASE) for _f, v in _fields(entry)):
        return []
    return [
        f for f, v in _fields(entry)
        if f.casefold().endswith(f"({unit.casefold()})")
        and not re.search(r"\b(?:target|total|output|balance)\b", f, re.IGNORECASE)
        and _NUMBER.fullmatch(v)
    ]


def _chart_target(group: list[ObservationIn]) -> tuple[str, float, str] | None:
    for entry in group:
        target = daily_target(_fields(entry))
        if target is not None and target[2]:
            return target
    return None


def _running_totals(entries: list[ObservationIn]) -> dict[int, dict[str, str]]:
    """id(entry) -> {field: " (running total 200 ml)"} for amounts on a chart with a daily target,
    added up per day in the order the chart dates them. Code adds them, so the model never has
    to (it may not calculate)."""
    running: dict[int, dict[str, str]] = {}
    for group in _charts(entries).values():
        target = _chart_target(group)
        if target is None:
            continue
        unit = target[2]
        totals: dict[tuple[tuple[int, int, int], str], float] = {}
        for entry in sorted(group, key=_entry_sort_key):
            day = _entry_sort_key(entry)[0]
            for field in _counted_fields(entry, unit):
                value = float(dict(_fields(entry))[field])
                totals[(day, field)] = totals.get((day, field), 0.0) + value
                running.setdefault(id(entry), {})[field] = (
                    f" (running total {totals[(day, field)]:g} {unit})"
                )
    return running


def _daily_totals(group: list[ObservationIn]) -> str | None:
    """"700 ml against a 1500 ml target" (one day) or "daily total against a 1500 ml target:
    15/11/2026 900 ml, 16/11/2026 700 ml" - the amounts added up per chart day."""
    target = _chart_target(group)
    if target is None:
        return None
    _field, amount, unit = target
    per_day: dict[tuple[int, int, int], float] = {}
    shown_day: dict[tuple[int, int, int], str] = {}
    for entry in group:
        day = _entry_sort_key(entry)[0]
        fields = dict(_fields(entry))
        for field in _counted_fields(entry, unit):
            per_day[day] = per_day.get(day, 0.0) + float(fields[field])
            shown_day[day] = f"{day[2]:02d}/{day[1]:02d}/{day[0]}"
    if not per_day:
        return None
    goal = f"a {amount:g} {unit} target"
    if len(per_day) == 1:
        return f"{next(iter(per_day.values())):g} {unit} against {goal}"
    days = ", ".join(f"{shown_day[d]} {per_day[d]:g} {unit}" for d in sorted(per_day))
    return f"daily total against {goal}: {days}"


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
    summaries: list[EvidenceItem] = []
    repeated = _repeated_fields(entries)
    for (name, historical), group in _charts(entries).items():
        if len(group) < 2:
            continue  # one entry says it all
        # OPENCARE-99: in the order the chart dates them (its own Date & Time), so "latest" is
        # the last drink given, not the last one typed in. Stable for equal times.
        ordered = sorted(group, key=_entry_sort_key)
        dated = [e.recorded_date for e in ordered if _day(e.recorded_date)]
        days = len({_day(d) for d in dated})
        parts = [f"{len(group)} entries"]
        if dated:
            if days == 1:
                parts[0] += f" on {dated[0]}"
            else:
                parts[0] += f" on {days} days ({dated[0]} to {dated[-1]})"
        totals = _daily_totals(group)
        if totals:
            parts.append(totals)

        figures: dict[str, list[str]] = {}
        for entry in ordered:
            for field, value in _fields(entry):
                if field and _NUMBER.fullmatch(value):
                    figures.setdefault(field, []).append(value)
        same = repeated.get((name, historical), set())
        for field, values in figures.items():
            low = min(values, key=float)
            high = max(values, key=float)
            if float(low) == float(high) and len(values) == len(group):
                parts.append(f"{field}: {values[-1]} every time")
            elif float(low) == float(high):
                parts.append(f"{field}: {values[-1]}")
            else:
                parts.append(f"{field}: {low} to {high}, latest {values[-1]}")
        # Words every entry repeats ("Record type: Intake"), said once here - not on each entry.
        for field, value in _fields(ordered[0]):
            if field in same and not _NUMBER.fullmatch(value):
                parts.append(f"{field}: {value} every time")

        summaries.append(
            EvidenceItem(
                id=f"T{len(summaries) + 1}",
                source_type=SourceType.OBSERVATION_SUMMARY,
                label=name,
                date=dated[-1] if dated else None,
                text=" · ".join(parts),
                linked_to_pathway=any(e.pathway_linked for e in group),
                historical=historical,
                supporting=all(e.is_supporting for e in group),
            )
        )
    return summaries


# A date or date-time as SurveyJS saves it: "2026-11-09", "2026-11-09T21:40", "2026-11-09 21:40:00".
_ISO_DATE = re.compile(
    r"(\d{4})-(\d{2})-(\d{2})"
    r"(?:[T ](\d{2}):(\d{2})(?::\d{2}(?:\.\d+)?)?)?(?:Z|[+-]\d{2}:?\d{2})?"
)


_UK = ZoneInfo("Europe/London")
# A field that IS the entry's own date and time ("Date & Time of Fall", "Date & Time") leads the
# entry without its question; any other date ("Date of last GP review") keeps its name.
_ENTRY_WHEN = re.compile(r"\b(?:date|time)\b", re.IGNORECASE)
_OTHER_DATE = re.compile(
    r"\b(?:last|next|review|due|previous|birth|admission|appointment|given|started|stopped)\b",
    re.IGNORECASE,
)


def readable_date(value: str) -> str:
    """OPENCARE-98: "2026-11-09T21:40" -> "09/11/2026 21:40". A time saved without a zone is shown
    as written; one with a zone ("...Z") in UK time. Anything else unchanged."""
    text = value.strip()
    m = _ISO_DATE.fullmatch(text)
    if not m:
        return value
    if m.group(4) and (text.endswith(("Z", "z")) or re.search(r"[+-]\d{2}:?\d{2}$", text)):
        try:
            moment = datetime.fromisoformat(text.replace("Z", "+00:00").replace("z", "+00:00"))
            return moment.astimezone(_UK).strftime("%d/%m/%Y %H:%M")
        except ValueError:
            pass
    day = f"{m.group(3)}/{m.group(2)}/{m.group(1)}"
    return f"{day} {m.group(4)}:{m.group(5)}" if m.group(4) else day


def observation_text(
    entry: ObservationIn,
    dropped: list[str] | None = None,
    *,
    leave_out: set[str] | frozenset[str] = frozenset(),
    running: dict[str, str] | None = None,
) -> str:
    """The fields that were filled in, as staff read them (OPENCARE-98): the entry's date and time
    first, without its question, then "Field: value" - "09/11/2026 21:40 · Location of Fall:
    Bathroom · What Happened?: Slipped on the bathroom floor". ISO dates are shown dd/mm/yyyy.
    OPENCARE-99: `leave_out` = fields every entry of the chart repeats (said once in its summary);
    `running` = field -> " (running total 200 ml)" for an amount counted against a daily target."""
    when: str | None = None
    parts = []
    for v in entry.values:
        value = _WHITESPACE.sub(" ", v.value or "").strip()
        field = _WHITESPACE.sub(" ", v.field).strip()
        if value and is_placeholder(value):
            if dropped is not None:
                dropped.append(value)
            continue
        if not value or (field and field in leave_out):
            continue
        shown = readable_date(value)
        if (when is None and shown != value and _ENTRY_WHEN.search(field)
                and not _OTHER_DATE.search(field)):
            when = shown  # the entry's own date and time leads, its question adds nothing
            continue
        shown += (running or {}).get(field, "")
        parts.append(f"{field}: {shown}" if field else shown)
    if leave_out and not parts:
        # Every field is one the chart repeats: the entry still shows what it recorded.
        return observation_text(entry, None, running=running)
    return " · ".join(([when] if when else []) + parts)

"""OPENCARE-99: the care standard a recorded score calls for - plain code, deterministic.

A MUST score sets the routine care (BAPEN "MUST" management guidelines, care home version: Low =
monthly, Medium = observe and record intake, High = treat and refer). The records rarely spell
these steps out ("weigh weekly", "rescreen monthly"), so a model kept to the records could not
write them - and one free to add them would guess. Here each step of the band the records show
becomes a K item: evidence the model cites like any other (the verifier checks the action against
it) and every step must be covered by an action.

    K1..  one care standard step, for the MUST band in the assessment (or, without one, the latest
          MUST chart entry), and the drink step for a chart with a daily fluid target

No band in the records = no MUST steps: the band is never guessed.

The step wording follows OPENCARE-99's expected draft and awaits clinical sign-off with the
OPENCARE-68 mapping (Linda, Stobhill). A home's own rules can replace this table later.
"""

import re

from careplan_ai.agents.care_plan.schemas import EvidenceItem, SourceType

# The bands, lowest first: (name, lowest score, steps). "{name}" is the resident.
_MUST_BANDS: list[tuple[str, int, list[str]]] = [
    ("Low risk", 0, [
        "Weigh {name} monthly on the same scales.",
        "Rescreen MUST monthly, or sooner if {name}'s weight drops.",
    ]),
    ("Medium risk", 1, [
        "Weigh {name} weekly on the same scales.",
        "Record everything {name} eats and drinks on the Food and Fluid charts.",
        "Rescreen MUST monthly, or sooner if {name}'s weight drops.",
        "Tell the nurse or GP, and follow up any dietitian referral, if {name}'s weight keeps "
        "falling or {name} eats less than half of each meal for 3 days.",
    ]),
    ("High risk", 2, [
        "Refer {name} to the dietitian, or follow up the referral, and follow the advice given.",
        "Weigh {name} weekly on the same scales.",
        "Record everything {name} eats and drinks on the Food and Fluid charts.",
        "Rescreen MUST monthly, or sooner if {name}'s weight drops.",
        "Tell the nurse or GP if {name}'s weight keeps falling or {name} eats less than half of "
        "each meal for 3 days.",
    ]),
]

_FLUID_STEP = (
    "Offer {name} a drink every 1-2 hours while awake to reach {target} a day, and record each "
    "drink on the {chart}."
)

_MUST_PATHWAY = re.compile(r"\b(?:must\b|malnutrition)", re.IGNORECASE)
_MUST_CHART = "OF_MUST"
# "MUST score", "Total MUST score", "Overall score" - the score question, not a step score
# ("Step 2 - weight loss score").
_SCORE_QUESTION = re.compile(r"\bscore\b", re.IGNORECASE)
_STEP_SCORE = re.compile(r"\bstep\s*\d|\bbmi\b|\bweight\s+loss\b|\bacute\b", re.IGNORECASE)
_TOTAL_HINT = re.compile(r"\b(?:must|total|overall|final)\b", re.IGNORECASE)
_BAND_TEXT = re.compile(r"\b(low|medium|high)\b(?:\s+risk)?", re.IGNORECASE)
# The overall MUST risk question ("Risk category", "MUST risk", "Overall risk level") - not any
# question with "level" in it ("Activity level: Low").
_RISK_QUESTION = re.compile(r"\brisk\b", re.IGNORECASE)
_INTEGER = re.compile(r"^\s*(\d+)\b")
# A chart field that is a daily target: "Fluid target per day (ml)".
_DAILY_TARGET = re.compile(r"\b(?:per\s+day|daily)\b", re.IGNORECASE)
_TARGET = re.compile(r"\btarget\b", re.IGNORECASE)
_UNIT = re.compile(r"\(([^()]+)\)\s*$")
_NUMBER = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([a-zA-Z]*)\s*$")


def _band_for_score(score: int) -> int:
    index = 0
    for i, (_, lowest, _steps) in enumerate(_MUST_BANDS):
        if score >= lowest:
            index = i
    return index


def _band_from_text(text: str) -> int | None:
    m = _BAND_TEXT.search(text)
    if not m:
        return None
    return {"low": 0, "medium": 1, "high": 2}[m.group(1).casefold()]


def _band_from(pairs: list[tuple[str, str, str]]) -> tuple[int, str] | None:
    """(band index, evidence id) from (id, question, answer) pairs: the total score first (a
    question with "score" and "MUST"/"total", or the only score question), else a risk band
    answer ("Medium risk")."""
    scores = [
        (i, q, a) for i, q, a in pairs
        if _SCORE_QUESTION.search(q) and not _STEP_SCORE.search(q) and _INTEGER.match(a)
    ]
    totals = [s for s in scores if _TOTAL_HINT.search(s[1])] or (scores if len(scores) == 1 else [])
    if totals:
        i, _q, a = totals[-1]
        m = _INTEGER.match(a)
        if m:
            return _band_for_score(int(m.group(1))), i
    # A risk band answer: a question naming MUST first, then any overall risk question. A step's
    # own score ("Step 2 - weight loss score: High") is never the band.
    candidates = [
        (i, q, a) for i, q, a in pairs
        if (_RISK_QUESTION.search(q) or _SCORE_QUESTION.search(q)) and not _STEP_SCORE.search(q)
    ]
    candidates.sort(key=lambda c: not re.search(r"\bmust\b", c[1], re.IGNORECASE))
    for i, _q, a in candidates:
        band = _band_from_text(a)
        if band is not None:
            return band, i
    return None


def _sortable(day: str | None) -> tuple[int, int, int]:
    """'dd/mm/yyyy' -> (yyyy, mm, dd); undated sorts oldest."""
    parts = (day or "")[:10].split("/")
    if len(parts) == 3 and all(p.isdigit() for p in parts):
        return int(parts[2]), int(parts[1]), int(parts[0])
    return 0, 0, 0


def _chart_pairs(entry_id: str, text: str) -> list[tuple[str, str, str]]:
    """'Field: value · Field: value' -> (id, field, value) pairs."""
    pairs = []
    for part in text.split(" · "):
        field, sep, value = part.partition(": ")
        if sep:
            pairs.append((entry_id, field.strip(), value.strip()))
    return pairs


def must_band(
    items: list[EvidenceItem], chart_types: dict[str, str | None]
) -> tuple[str, int, str] | None:
    """(band name, band index, id of the evidence that shows it) - from the assessment's total
    score or risk band, else from the latest current MUST chart entry. None = not recorded."""
    answers = [(e.id, e.label, e.text) for e in items if e.source_type == SourceType.ASSESSMENT]
    found = _band_from(answers)
    if found is None:
        entries = [
            e for e in items
            if e.is_observation and not e.historical and chart_types.get(e.id) == _MUST_CHART
        ]
        # Newest first by the chart's own date (dd/mm/yyyy); stable for the same day.
        for entry in sorted(entries, key=lambda e: _sortable(e.date), reverse=True):
            found = _band_from(_chart_pairs(entry.id, entry.text))
            if found:
                break
    if found is None:
        return None
    index, evidence_id = found
    return _MUST_BANDS[index][0], index, evidence_id


def daily_target(fields: list[tuple[str, str]]) -> tuple[str, float, str] | None:
    """(field, amount, unit) of a chart's daily target ("Fluid target per day (ml)": "1500")."""
    for field, value in fields:
        if _TARGET.search(field) and _DAILY_TARGET.search(field):
            m = _NUMBER.match(value or "")
            if m:
                unit_match = _UNIT.search(field)
                unit = (unit_match.group(1) if unit_match else m.group(2)).strip()
                return field, float(m.group(1)), unit
    return None


def _amount(value: float, unit: str) -> str:
    number = f"{value:g}"
    return f"{number} {unit}".strip()


def care_standards(
    items: list[EvidenceItem],
    *,
    pathway: str | None,
    resident_name: str | None,
    chart_types: dict[str, str | None],
    chart_fields: dict[str, list[tuple[str, str]]],
) -> list[EvidenceItem]:
    """The K items for this draft. `chart_types`: V id -> chart type ("OF_MUST");
    `chart_fields`: V id -> its (field, value) pairs as recorded."""
    name = (resident_name or "").strip() or "the resident"
    standards: list[tuple[str, str]] = []  # (label, text)

    if pathway and _MUST_PATHWAY.search(pathway):
        band = must_band(items, chart_types)
        if band is not None:
            band_name, index, _evidence_id = band
            label = f"MUST care standard - {band_name}"
            standards += [(label, step.format(name=name)) for step in _MUST_BANDS[index][2]]

    # A current primary chart with a daily fluid target: how often to offer a drink.
    seen_charts: set[str] = set()
    for item in items:
        if not item.is_observation or item.historical or item.supporting:
            continue
        if item.label in seen_charts:
            continue
        target = daily_target(chart_fields.get(item.id, []))
        if target is None or "fluid" not in f"{item.label} {target[0]}".casefold():
            continue
        seen_charts.add(item.label)
        _field, amount, unit = target
        standards.append((
            "Fluid care standard",
            _FLUID_STEP.format(name=name, target=_amount(amount, unit), chart=item.label),
        ))

    return [
        EvidenceItem(
            id=f"K{i}",
            source_type=SourceType.CARE_STANDARD,
            label=label,
            text=text,
        )
        for i, (label, text) in enumerate(standards, start=1)
    ]

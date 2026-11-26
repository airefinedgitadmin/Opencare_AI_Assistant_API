"""The care standard a recorded score, level or reading calls for - plain code, deterministic.

OPENCARE-99: a MUST score sets the routine care (BAPEN "MUST" management guidelines, care home
version: Low = monthly, Medium = observe and record intake, High = treat and refer). The records
rarely spell these steps out ("weigh weekly", "rescreen monthly"), so a model kept to the records
could not write them - and one free to add them would guess. Here each step of the band the
records show becomes a K item: evidence the model cites like any other (the verifier checks the
action against it) and every step must be covered by an action.

OPENCARE-115 adds the same for Waterlow (by total score band) and Choking (by the IDDSI diet and
fluid levels in the notes or assessment). OPENCARE-116 limits the drink step to MUST and Bladder
and Bowel plans whose intake is below target, and adds the TPR & BP steps a reading calls for (a
temperature of 37.8 °C or more or new confusion; a drop in blood pressure on standing, or
dizziness).

    K1..  one care standard step

Nothing is guessed: no score, level or reading in the records = no step. The wording and bands
live in care_standards.json (CAREPLAN_AI_CARE_STANDARDS_FILE points at another copy), so the
clinical wording can change without a release. It awaits clinical sign-off (Linda, Stobhill).
"""

import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from careplan_ai.agents.care_plan.schemas import EvidenceItem, SourceType
from careplan_ai.core.exceptions import ConfigurationError

_PACKAGED = Path(__file__).parent / "care_standards.json"


@lru_cache
def _load(path: str) -> dict[str, Any]:
    try:
        data: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ConfigurationError(f"Care standards file not readable: {path}") from exc
    return data


def _configured_path() -> str | None:
    # CAREPLAN_AI_CARE_STANDARDS_FILE (declared in core/config.py). Read from the environment
    # here, not through get_settings(): building evidence must not fix the cached settings.
    return os.environ.get("CAREPLAN_AI_CARE_STANDARDS_FILE") or None


def config() -> dict[str, Any]:
    configured = _configured_path()
    return _load(configured if configured else str(_PACKAGED))


def _matches(pattern: str | None, pathway: str | None) -> bool:
    return bool(pattern and pathway and re.search(pattern, pathway, re.IGNORECASE))


_MUST_CHART = "OF_MUST"
# "MUST score", "Total MUST score", "Overall score" - the score question, not a step score
# ("Step 2 - weight loss score").
_SCORE_QUESTION = re.compile(r"\bscore\b", re.IGNORECASE)
_STEP_SCORE = re.compile(r"\bstep\s*\d|\bbmi\b|\bweight\s+loss\b|\bacute\b", re.IGNORECASE)
_MUST_TOTAL_HINT = r"\b(?:must|total|overall|final)\b"
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


def _bands(standard: str) -> list[dict[str, Any]]:
    bands: list[dict[str, Any]] = config()[standard]["bands"]
    return sorted(bands, key=lambda b: int(b["from"]))


def _band_for_score(score: int, bands: list[dict[str, Any]]) -> int | None:
    """Index of the highest band whose lowest score the score reaches; None below the first."""
    index = None
    for i, band in enumerate(bands):
        if score >= int(band["from"]):
            index = i
    return index


def _band_from_text(text: str) -> int | None:
    m = _BAND_TEXT.search(text)
    if not m:
        return None
    return {"low": 0, "medium": 1, "high": 2}[m.group(1).casefold()]


def _total_score(pairs: list[tuple[str, str, str]], hint: str) -> tuple[int, str] | None:
    """(score, evidence id) of the total score question: a question with "score" and the hint
    (e.g. "MUST" or "total"), or the only score question."""
    scores = [
        (i, q, a) for i, q, a in pairs
        if _SCORE_QUESTION.search(q) and not _STEP_SCORE.search(q) and _INTEGER.match(a)
    ]
    totals = [s for s in scores if re.search(hint, s[1], re.IGNORECASE)] or (
        scores if len(scores) == 1 else []
    )
    if not totals:
        return None
    i, _q, a = totals[-1]
    m = _INTEGER.match(a)
    return (int(m.group(1)), i) if m else None


def _must_band_from(pairs: list[tuple[str, str, str]]) -> tuple[int, str] | None:
    """(band index, evidence id) from (id, question, answer) pairs: the total score first, else a
    risk band answer ("Medium risk")."""
    bands = _bands("must")
    total = _total_score(pairs, _MUST_TOTAL_HINT)
    if total is not None:
        band = _band_for_score(total[0], bands)
        if band is not None:
            return band, total[1]
    # A risk band answer: a question naming MUST first, then any overall risk question. A step's
    # own score ("Step 2 - weight loss score: High") is never the band.
    candidates = [
        (i, q, a) for i, q, a in pairs
        if (_RISK_QUESTION.search(q) or _SCORE_QUESTION.search(q)) and not _STEP_SCORE.search(q)
    ]
    candidates.sort(key=lambda c: not re.search(r"\bmust\b", c[1], re.IGNORECASE))
    for i, _q, a in candidates:
        found = _band_from_text(a)
        if found is not None:
            return found, i
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


def _answers(items: list[EvidenceItem]) -> list[tuple[str, str, str]]:
    return [(e.id, e.label, e.text) for e in items if e.source_type == SourceType.ASSESSMENT]


def must_band(
    items: list[EvidenceItem], chart_types: dict[str, str | None]
) -> tuple[str, int, str] | None:
    """(band name, band index, id of the evidence that shows it) - from the assessment's total
    score or risk band, else from the latest current MUST chart entry. None = not recorded."""
    found = _must_band_from(_answers(items))
    if found is None:
        entries = [
            e for e in items
            if e.is_observation and not e.historical and chart_types.get(e.id) == _MUST_CHART
        ]
        # Newest first by the chart's own date (dd/mm/yyyy); stable for the same day.
        for entry in sorted(entries, key=lambda e: _sortable(e.date), reverse=True):
            found = _must_band_from(_chart_pairs(entry.id, entry.text))
            if found:
                break
    if found is None:
        return None
    index, evidence_id = found
    return _bands("must")[index]["name"], index, evidence_id


def waterlow_band(items: list[EvidenceItem]) -> tuple[str, int, str] | None:
    """OPENCARE-115: (band name, band index, evidence id) from the Waterlow total score. None = no
    total score recorded, or below the first band (not at risk)."""
    total = _total_score(_answers(items), config()["waterlow"]["score_question"])
    if total is None:
        return None
    bands = _bands("waterlow")
    index = _band_for_score(total[0], bands)
    if index is None:
        return None
    return bands[index]["name"], index, total[1]


# --- OPENCARE-115: IDDSI diet and fluid levels, as the records write them ---

# "IDDSI Level 6 soft and bite-sized", "Level 5 minced and moist", "level 4 pureed".
_DIET_LEVEL = re.compile(
    r"\b(?:IDDSI\s+)?level\s*([3-7])\b(?:\s*[-–:,]?\s*(?:and\s+)?(?:liquidised|pur[eé]ed|minced|moist"
    r"|soft|bite[-\s]sized|easy\s+to\s+chew|regular))*",
    re.IGNORECASE,
)
# "Level 2 mildly thick", "mildly thick fluids", "thickened fluids".
_THICK = re.compile(
    r"\b(?:(?:IDDSI\s+)?level\s*([1-4])\s*[-–:]?\s*)?(slightly|mildly|moderately|extremely)\s+"
    r"thick(?:ened)?(?:\s+(?:fluids?|drinks?))?"
    r"|\bthickened\s+(?:fluids?|drinks?)\b",
    re.IGNORECASE,
)
# "normal fluids", "thin fluids", "IDDSI Level 0".
_THIN = re.compile(
    r"\b(?:normal|thin)\s+(?:fluids?|drinks?)\b|\b(?:IDDSI\s+)?level\s*0\b(?:\s+thin)?",
    re.IGNORECASE,
)
_FLUID_WORD = re.compile(r"\b(?:thick\w*|fluids?|drinks?)\b", re.IGNORECASE)


@dataclass(frozen=True)
class ChokingLevels:
    diet: str | None  # "IDDSI Level 5 minced and moist" as written
    diet_level: int | None
    fluids: str | None  # "Level 2 mildly thick fluids" / "normal fluids" as written
    thick: str | None  # "mildly thick" - set only when fluids are thickened
    record_ids: tuple[str, ...]

    @property
    def modified(self) -> bool:
        return (self.diet_level is not None and self.diet_level <= 6) or self.thick is not None


def _tidy(text: str) -> str:
    return " ".join(text.split()).rstrip(" .,;")


def choking_levels(items: list[EvidenceItem]) -> ChokingLevels:
    """The latest diet and fluid levels in the current notes and assessment answers."""
    diet = fluids = thick = None
    diet_level = None
    diet_id = fluid_id = None
    records = [
        e for e in items
        if not e.historical and (e.is_note or e.source_type == SourceType.ASSESSMENT)
    ]
    for record in sorted(records, key=lambda e: _sortable(e.date)):
        text = f"{record.label}: {record.text}"
        for m in _DIET_LEVEL.finditer(text):
            if _FLUID_WORD.search(m.group(0)) or _THICK.match(text, m.start()):
                continue
            diet, diet_level, diet_id = _tidy(m.group(0)), int(m.group(1)), record.id
        t = _THICK.search(text)
        if t:
            fluids, fluid_id = _tidy(t.group(0)), record.id
            thick = f"{t.group(2)} thick".casefold() if t.group(2) else "thickened"
        elif _THIN.search(text):
            m2 = _THIN.search(text)
            assert m2 is not None
            fluids, fluid_id, thick = _tidy(m2.group(0)), record.id, None
    if fluids is not None and not _FLUID_WORD.search(fluids):
        fluids += " fluids"
    ids = tuple(i for i in (diet_id, fluid_id) if i)
    return ChokingLevels(diet=diet, diet_level=diet_level, fluids=fluids, thick=thick,
                         record_ids=ids)


def choking_not_recorded(
    items: list[EvidenceItem], *, pathway: str | None, resident_name: str | None
) -> str | None:
    """OPENCARE-115: the notice for a Choking plan with no IDDSI level in the records."""
    standard = config().get("choking")
    if not standard or not _matches(standard.get("pathway"), pathway):
        return None
    levels = choking_levels(items)
    if levels.diet is not None or levels.fluids is not None:
        return None
    name = (resident_name or "").strip() or "the resident"
    return str(standard["not_recorded"]).format(name=name)


def thickened_level(items: list[EvidenceItem], pathway: str | None) -> str | None:
    """OPENCARE-115: in a Choking plan with thickened fluids, the words every drink action must
    carry ("mildly thick"); None otherwise."""
    standard = config().get("choking")
    if not standard or not _matches(standard.get("pathway"), pathway):
        return None
    return choking_levels(items).thick


# --- daily fluid target (OPENCARE-99, OPENCARE-116) ---


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


# --- OPENCARE-116: TPR & BP readings ---

_TEMPERATURE = re.compile(
    r"\btemp\w*[^:·]*:\s*(\d{2}(?:\.\d+)?)|(\d{2}\.\d+)\s*°\s*C", re.IGNORECASE
)
_LYING = re.compile(r"\blying\b[^·\d]*?(\d{2,3})\s*/\s*\d{2,3}", re.IGNORECASE)
_STANDING = re.compile(r"\bstanding\b[^·\d]*?(\d{2,3})\s*/\s*\d{2,3}", re.IGNORECASE)
_DIZZY = re.compile(r"\b(?:dizz\w*|light[-\s]?headed)\b", re.IGNORECASE)
_CONFUSED = re.compile(r"\b(?:more|new(?:ly)?|sudden(?:ly)?|increas\w*)\s+confus\w*", re.IGNORECASE)


def _tpr_findings(
    items: list[EvidenceItem], chart_types: dict[str, str | None], pathway: str | None
) -> list[tuple[str, str, str]]:
    """(label finding, step template, V id) for each TPR rule a current reading meets - the
    newest reading that meets it."""
    standard = config().get("tpr")
    if not standard:
        return []
    entries = sorted(
        (e for e in items
         if e.is_observation and not e.historical and chart_types.get(e.id) == standard["chart"]),
        key=lambda e: _sortable(e.date), reverse=True,
    )
    found: list[tuple[str, str, str]] = []
    fever = standard.get("fever")
    if fever and _matches(fever.get("pathway"), pathway):
        threshold = float(fever["from_celsius"])
        for entry in entries:
            temps = [float(a or b) for a, b in _TEMPERATURE.findall(entry.text)]
            hot = [t for t in temps if t >= threshold]
            if hot:
                finding = str(fever["finding"]).format(value=f"{max(hot):g}")
            elif _CONFUSED.search(entry.text):
                finding = str(fever["confusion_finding"])
            else:
                continue
            step = str(fever["step"]).replace("{threshold}", f"{threshold:g}")
            found.append((finding, step, entry.id))
            break
    drop = standard.get("postural_drop")
    if drop and _matches(drop.get("pathway"), pathway):
        for entry in entries:
            lying, standing = _LYING.search(entry.text), _STANDING.search(entry.text)
            dropped = bool(
                lying and standing
                and int(lying.group(1)) - int(standing.group(1)) >= int(drop["systolic_drop"])
            )
            if dropped or (standing and _DIZZY.search(entry.text)):
                found.append((str(drop["finding"]), str(drop["step"]), entry.id))
                break
    return found


def care_standards(
    items: list[EvidenceItem],
    *,
    pathway: str | None,
    resident_name: str | None,
    chart_types: dict[str, str | None],
    chart_fields: dict[str, list[tuple[str, str]]],
    below_target: set[str] | None = None,
) -> list[EvidenceItem]:
    """The K items for this draft. `chart_types`: V id -> chart type ("OF_MUST");
    `chart_fields`: V id -> its (field, value) pairs as recorded; `below_target`: the charts whose
    latest day's total is below their daily target (OPENCARE-116)."""
    name = (resident_name or "").strip() or "the resident"
    conf = config()
    # (label, text, the record the step answers - so Sources show it)
    standards: list[tuple[str, str, str | None]] = []

    must = conf.get("must")
    if must and _matches(must.get("pathway"), pathway):
        band = must_band(items, chart_types)
        if band is not None:
            band_name, index, _evidence_id = band
            label = str(must["label"]).format(band=band_name)
            standards += [(label, step.format(name=name), None)
                          for step in _bands("must")[index]["steps"]]

    waterlow = conf.get("waterlow")
    if waterlow and _matches(waterlow.get("pathway"), pathway):
        wband = waterlow_band(items)
        if wband is not None:
            band_name, index, evidence_id = wband
            label = str(waterlow["label"]).format(band=band_name)
            standards += [(label, step.format(name=name), evidence_id)
                          for step in _bands("waterlow")[index]["steps"]]

    choking = conf.get("choking")
    if choking and _matches(choking.get("pathway"), pathway):
        levels = choking_levels(items)
        ref = levels.record_ids[0] if levels.record_ids else None
        if levels.modified:
            part = choking["modified"]
            band_name = levels.diet or levels.fluids or "Modified diet"
            label = str(choking["label"]).format(band=band_name)
            if levels.diet:
                standards.append((label, str(part["diet_step"]).format(name=name,
                                                                       diet=levels.diet), ref))
            if levels.fluids:
                standards.append((label, str(part["fluid_step"]).format(name=name,
                                                                        fluids=levels.fluids),
                                  levels.record_ids[-1]))
            standards += [(label, step.format(name=name), ref) for step in part["steps"]]
        else:
            part = choking["low"]
            label = str(choking["label"]).format(band=part["name"])
            standards += [(label, step.format(name=name), ref) for step in part["steps"]]

    # A current primary chart with a daily fluid target: how often to offer a drink - only in the
    # pathways the config names (OPENCARE-116: never in a Choking plan), and only when the latest
    # day's intake is below target if the config says so.
    fluid = conf.get("fluid")
    if fluid and _matches(fluid.get("pathway"), pathway):
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
            if fluid.get("only_below_target") and item.label not in (below_target or set()):
                continue
            _field, amount, unit = target
            standards.append((
                str(fluid["label"]),
                str(fluid["step"]).format(name=name, target=_amount(amount, unit),
                                          chart=item.label),
                None,
            ))

    tpr = conf.get("tpr")
    if tpr:
        for finding, step, entry_id in _tpr_findings(items, chart_types, pathway):
            standards.append((str(tpr["label"]).format(finding=finding), step.format(name=name),
                              entry_id))

    return [
        EvidenceItem(
            id=f"K{i}",
            source_type=SourceType.CARE_STANDARD,
            # The record the step answers: cited with it, so Sources show the reading or score.
            backed_by=ref,
            label=label,
            text=text,
        )
        for i, (label, text, ref) in enumerate(standards, start=1)
    ]

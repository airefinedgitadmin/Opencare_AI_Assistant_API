"""OPENCARE-113: earlier care plan lines that the current records make wrong - plain code rules.

A note that says in words that an earlier line no longer applies is found by the model (it puts
the line under not_used, citing the note). A change that shows only in a SCORE or a DIET LEVEL was
missed: "Staff to cut up his meat" was copied into the actions of a resident now on IDDSI Level 5
minced and moist food. These rules decide those lines in code, so they cannot be missed:

- Choking: the diet is IDDSI Level 3, 4 or 5, or fluids are thickened -> an earlier line about a
  normal diet, thin fluids or cutting up food is not used.
- MUST: the score is 1 or more (the MUST care standard is Medium or High) -> an earlier "weigh
  monthly" line is not used (weekly applies).
- Any pathway: an earlier line that describes how the resident IS ("Willie's skin is intact",
  "Nell walks independently") is not something staff do, so it is never an action.

Each rule only fires on words found in the records; the reason quotes the record, never a guess.
"""

import re
from dataclasses import dataclass

from careplan_ai.agents.care_plan.schemas import EvidenceItem, SourceType

_CHOKING_PATHWAY = re.compile(r"\b(?:chok\w*|swallow\w*|dysphagia)\b", re.IGNORECASE)
_MUST_PATHWAY = re.compile(r"\b(?:must\b|malnutrition)", re.IGNORECASE)

# "IDDSI Level 5 minced and moist", "Level 4 pureed", "level 3 liquidised" - a modified diet.
_MODIFIED_DIET = re.compile(
    r"\b(?:IDDSI\s+)?level\s*([345])\b(?:\s*[-–:]?\s*(?:liquidised|pur[eé]ed|minced(?:\s+and\s+"
    r"moist)?|moist))*(?:\s+(?:diet|food|texture|consistency))?",
    re.IGNORECASE,
)
# "Level 2 mildly thick", "mildly thick fluids", "thickened fluids", "IDDSI level 1 drinks".
_THICK_FLUIDS = re.compile(
    r"\b(?:(?:IDDSI\s+)?level\s*[1-4]\s*[-–:]?\s*)?(?:slightly|mildly|moderately|extremely)\s+"
    r"thick(?:ened)?(?:\s+(?:fluids?|drinks?))?"
    r"|\bthickened\s+(?:fluids?|drinks?)\b"
    r"|\b(?:IDDSI\s+)?level\s*[1-4]\s+(?:fluids?|drinks?)\b",
    re.IGNORECASE,
)
# The DIET level lines must not mention a fluid level as the diet ("Level 2 mildly thick fluids").
_FLUID_WORD = re.compile(r"\b(?:thick|fluids?|drinks?)\b", re.IGNORECASE)

# Earlier lines a modified diet or thickened fluids make wrong.
_UNSAFE_FOR_MODIFIED_DIET = re.compile(
    r"\bnormal\s+(?:diet|food|texture)\b|\bthin\s+(?:fluids?|drinks?)\b"
    r"|\bcut(?:s|ting)?\s+up\b|\bcut(?:s|ting)?\s+(?:[\w'’]+\s+){0,3}(?:food|meat|meals?)\b",
    re.IGNORECASE,
)
# Retest 26/11 (Rab, IDDSI Level 5): "Staff should cut and prepare Rab's food before serving it",
# from the assessment answer "Requires assistance with food cutting". At Level 3-5 the food is
# already liquidised, pureed or minced, so cutting it is not care - whatever the earlier plan said.
_CUT_FOOD = re.compile(
    r"\bcut(?:s|ting)?\s+up\b|\bcut(?:s|ting)?\s+(?:[\w'’]+\s+){0,3}(?:food|meat|meals?)\b",
    re.IGNORECASE,
)


def cutting_food_at_level(
    text: str, evidence: list[EvidenceItem], pathway: str | None
) -> str | None:
    """The modified diet level ("IDDSI Level 5 minced and moist diet") that makes cutting the
    food in `text` wrong, or None."""
    if not pathway or not _CHOKING_PATHWAY.search(pathway) or not _CUT_FOOD.search(text):
        return None
    # The DIET level only: thickened fluids alone (a normal diet) may still need cutting up.
    level = None
    for record in _current_records(evidence):
        for m in _MODIFIED_DIET.finditer(record.text):
            if not _FLUID_WORD.search(m.group(0)):
                level = m.group(0)
    return _sentence_case(level) if level else None
_WEIGH_MONTHLY = re.compile(r"\bweigh\w*\b[^.;]*\bmonthly\b", re.IGNORECASE)

# A line that says what someone or something IS, not what staff do: it starts with the resident
# (or he / she / they / his / her) and gives no instruction.
_PRONOUNS = {"he", "she", "they", "his", "her", "their"}
_INSTRUCTION = re.compile(
    r"\b(?:staff|should|must|please|ensure|encourage|offer|assist|support|remind|check|help"
    r"|give|weigh|record|monitor|observe|supervise|refer|report|tell|inform|prompt|use|apply"
    r"|change|reposition|ask|avoid|provide|keep|make)\b",
    re.IGNORECASE,
)
_FIRST_WORD = re.compile(r"^\W*([A-Za-z][A-Za-z'’-]*)")


@dataclass(frozen=True)
class RuleNotUsed:
    evidence_id: str
    superseded_by: str
    reason: str
    # The words an action must not carry once this line is not used (None: the overlap check).
    forbidden: re.Pattern[str] | None = None


def _current_records(evidence: list[EvidenceItem]) -> list[EvidenceItem]:
    return [
        e for e in evidence
        if not e.historical and (e.is_note or e.source_type == SourceType.ASSESSMENT)
    ]


_DIET_WORD = re.compile(r"\b(?:diet|food|texture|consistency|fluids?|drinks?)\b", re.IGNORECASE)


def _sentence_case(text: str) -> str:
    text = " ".join(text.split()).rstrip(" .,;")
    if not _DIET_WORD.search(text):
        text += " diet"  # "IDDSI Level 5 minced and moist" is a diet level
    return text[:1].upper() + text[1:]


def _modified_diet(records: list[EvidenceItem]) -> tuple[str, str] | None:
    """(record id, "IDDSI Level 5 minced and moist") - the first current record naming a modified
    diet (Level 3-5) or thickened fluids, newest records are later in the list."""
    found: tuple[str, str] | None = None
    for record in records:
        for m in _MODIFIED_DIET.finditer(record.text):
            if _FLUID_WORD.search(m.group(0)):
                continue
            found = (record.id, m.group(0))
        if found is None:
            thick = _THICK_FLUIDS.search(record.text)
            if thick:
                found = (record.id, thick.group(0))
    return found


def _must_band(evidence: list[EvidenceItem]) -> tuple[str, str] | None:
    """(K id, band name) of the MUST care standard when it is Medium risk or higher."""
    for item in evidence:
        if item.source_type == SourceType.CARE_STANDARD and item.label.startswith("MUST "):
            band = item.label.rsplit(" - ", 1)[-1]
            if not band.casefold().startswith("low"):
                return item.id, band
    return None


def _names(resident_name: str | None) -> set[str]:
    return {w.casefold() for w in re.findall(r"[A-Za-z][A-Za-z'’-]*", resident_name or "")}


def describes_state(text: str, resident_name: str | None) -> bool:
    """'Willie's skin is intact.' / 'Nell walks independently.' - not something staff do."""
    m = _FIRST_WORD.match(text)
    if not m or _INSTRUCTION.search(text):
        return False
    first = m.group(1).casefold().replace("’", "'")
    if first.endswith("'s"):
        first = first[:-2]
    return first in _PRONOUNS or first in _names(resident_name)


def rule_not_used(
    evidence: list[EvidenceItem], *, pathway: str | None, resident_name: str | None
) -> list[RuleNotUsed]:
    """The earlier care plan lines (P items) these rules take out, with the record that shows why.
    Lines that only describe the resident are returned by `state_lines`, not here."""
    previous = [
        e for e in evidence
        if e.source_type == SourceType.PREVIOUS_CARE_PLAN and e.id.startswith("P")
    ]
    if not previous:
        return []
    out: list[RuleNotUsed] = []
    if pathway and _CHOKING_PATHWAY.search(pathway):
        diet = _modified_diet(_current_records(evidence))
        if diet is not None:
            record_id, level = diet
            for p in previous:
                if _UNSAFE_FOR_MODIFIED_DIET.search(p.text):
                    out.append(RuleNotUsed(
                        evidence_id=p.id, superseded_by=record_id,
                        reason=f"Now on {_sentence_case(level)}.",
                        forbidden=_UNSAFE_FOR_MODIFIED_DIET,
                    ))
    if pathway and _MUST_PATHWAY.search(pathway):
        band = _must_band(evidence)
        if band is not None:
            k_id, band_name = band
            for p in previous:
                if _WEIGH_MONTHLY.search(p.text):
                    out.append(RuleNotUsed(
                        evidence_id=p.id, superseded_by=k_id,
                        reason=f"MUST is now {band_name}, weigh weekly.",
                        forbidden=_WEIGH_MONTHLY,
                    ))
    return out


def state_lines(evidence: list[EvidenceItem], resident_name: str | None) -> set[str]:
    """Ids of earlier care plan lines that describe the resident rather than say what staff do."""
    return {
        e.id for e in evidence
        if e.source_type == SourceType.PREVIOUS_CARE_PLAN and e.id.startswith("P")
        and describes_state(e.text, resident_name)
    }

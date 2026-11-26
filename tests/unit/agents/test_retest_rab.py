"""Retest 26/11/2026 (Rab, 81363): Choking and Oral Health drafts. Every action below is the exact
text of the draft; the checks hold for any resident - they read that resident's own records."""

import pytest

from careplan_ai.agents.care_plan.schemas import (
    ActionOut,
    DraftPlanOut,
    EvidenceItem,
    SourceType,
)
from careplan_ai.agents.care_plan.superseded import cutting_food_at_level
from careplan_ai.agents.care_plan.validation import copies_answer, strip_reason, validate_draft


def _a(i: int, label: str, text: str) -> EvidenceItem:
    return EvidenceItem(id=f"A{i}", source_type=SourceType.ASSESSMENT, label=label, text=text,
                        risk_flagged=True)


SLT = EvidenceItem(id="N1", source_type=SourceType.NURSING_NOTE, label="Nursing note",
                   text="SLT visit: IDDSI Level 5 minced and moist diet and Level 2 mildly thick "
                        "fluids.")
CUTTING = _a(1, "Requires assistance with food cutting or preparing prior to consuming", "Yes")
ORAL = [
    _a(2, "Oral Cleanliness", "Particles/tartar in most areas; severe halitosis"),
    _a(3, "Tongue", "Patchy, fissured, red and coated"),
    _a(4, "Saliva", "Very little/no saliva present. Resident reports dry mouth"),
    _a(5, "Lips", "Dry, chapped, red at corners"),
]


def _draft(texts: list[str], ids: list[str]) -> DraftPlanOut:
    return DraftPlanOut(actions=[ActionOut(text=t, evidence_ids=ids) for t in texts],
                        risks=[], note_classes=[], not_used=[])


# --- Choking: cutting food at a modified diet level ---

def test_cutting_food_at_level_5_is_dropped() -> None:
    text = "Staff should cut and prepare Rab's food before serving it."
    assert cutting_food_at_level(text, [CUTTING, SLT], "Choking Risk Assessment") is not None
    result = validate_draft([CUTTING, SLT], _draft([text], ["A1"]), resident_name="Rab",
                            pathway="Choking Risk Assessment", with_outcomes=False)
    assert result.cards == [] or all("cut" not in c.text.casefold() for c in result.cards)
    assert any("cutting is not part of the care" in p for p in result.problems)


@pytest.mark.parametrize("records", [
    # Thickened fluids only (a normal diet): food may still need cutting up.
    [EvidenceItem(id="N1", source_type=SourceType.NURSING_NOTE, label="Nursing note",
                  text="Level 2 mildly thick fluids.")],
    # IDDSI Level 6 soft and bite-sized is cut up.
    [EvidenceItem(id="N1", source_type=SourceType.NURSING_NOTE, label="Nursing note",
                  text="IDDSI Level 6 soft and bite-sized diet.")],
    # No diet level recorded.
    [],
])
def test_cutting_food_is_care_without_a_level_3_to_5_diet(records: list[EvidenceItem]) -> None:
    text = "Staff should cut up Kate's food into small pieces."
    assert cutting_food_at_level(text, [CUTTING, *records], "Choking Risk Assessment") is None


def test_preparing_food_at_the_level_is_not_cutting() -> None:
    text = "Staff should make sure Rab's food is prepared at IDDSI Level 5 minced and moist."
    assert cutting_food_at_level(text, [CUTTING, SLT], "Choking Risk Assessment") is None


def test_outside_a_choking_plan_nothing_changes() -> None:
    text = "Staff should cut up Rab's food."
    assert cutting_food_at_level(text, [CUTTING, SLT], "MUST") is None


# --- pointing at the records instead of the care ---

@pytest.mark.parametrize("text,expected", [
    ("Staff should supervise Rab while eating, as the records state, at every meal.",
     "Staff should supervise Rab while eating at every meal."),
    ("Staff should offer assistance during meals when Rab gets tired and stops, as shown on the "
     "Food chart entries.",
     "Staff should offer assistance during meals when Rab gets tired and stops."),
    ("Staff should offer Level 2 mildly thick drinks throughout the day and encourage intake, as "
     "the Fluid chart shows 650 ml against a 1500 ml target.",
     "Staff should offer Level 2 mildly thick drinks throughout the day and encourage intake."),
])
def test_record_references_are_taken_out(text: str, expected: str) -> None:
    assert strip_reason(text) == expected


def test_record_the_drink_on_the_chart_is_still_care() -> None:
    text = "Staff should record each drink on the Fluid chart."
    assert strip_reason(text) == text


# --- Oral Health: assessment answers copied ---

RAB_ORAL = [
    "Staff should carry out the oral care review: brush Rab's teeth twice a day with a soft brush "
    "and check his right cheek for food after meals, clearing any food left there where "
    "particles/tartar are in most areas and severe halitosis is present.",
    "Staff should gently clean Rab's tongue during brushing to address his patchy, fissured, red "
    "and coated tongue.",
    "Staff should offer Rab mouth care for a very dry mouth where very little or no saliva is "
    "present and he reports a dry mouth, and support him with IDDSI Level 2 mildly thick drinks "
    "during mouth care.",
]


def test_copied_answers_are_found() -> None:
    assert [copies_answer(t, ORAL).id for t in RAB_ORAL] == ["A2", "A3", "A4"]  # type: ignore[union-attr]


@pytest.mark.parametrize("text", [
    "Staff should brush Rab's teeth twice a day with a soft brush and gently clean his tongue.",
    "Staff should check Rab's right cheek after every meal and clear any food left there.",
    "Staff should give Rab mouth care for his dry mouth several times a day.",
    "Staff should apply lip balm to keep Rab's lips moist.",
    # A Yes/No answer has no words to copy.
    "Staff should help Rab with food cutting or preparing.",
])
def test_plain_actions_do_not_copy_an_answer(text: str) -> None:
    assert copies_answer(text, ORAL + [CUTTING]) is None


def test_the_oral_draft_is_sent_back_and_the_review_lead_in_taken_out() -> None:
    review = EvidenceItem(
        id="R1", source_type=SourceType.REVIEW_NOTE, label="Review note",
        text="Oral care review: staff now brush Rab's teeth twice a day with a soft brush and "
             "check his right cheek for food after meals.")
    result = validate_draft(ORAL + [review], _draft(RAB_ORAL, ["A2", "A3", "A4", "R1"]),
                            resident_name="Rab", pathway="Oral Health Assessment",
                            with_outcomes=False)
    problems = "\n".join(result.problems)
    for ref in ("[A2]", "[A3]", "[A4]"):
        assert f"repeats the wording of the assessment answer {ref}" in problems
    assert result.cards[0].text.startswith("Staff should brush Rab's teeth twice a day")
    assert len(result.cards) == 3  # nothing dropped: the agent rewrites them

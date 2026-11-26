"""OPENCARE-114: raw form questions, scores, chart lines and copied notes never reach Actions.

Every string below is the exact text the 24/11/2026 retest found in the Action box.
"""

import pytest

from careplan_ai.agents.care_plan.schemas import (
    ActionOut,
    CardOrigin,
    DraftPlanOut,
    EvidenceItem,
    SourceType,
)
from careplan_ai.agents.care_plan.validation import not_an_instruction, validate_draft

NELL_NOTE = EvidenceItem(
    id="C1", source_type=SourceType.CARER_NOTE, label="Carer Note",
    text="Nell eats quickly and sometimes needs a reminder to slow down and take smaller "
         "mouthfuls.",
)
WILLIE_REVIEW = EvidenceItem(
    id="R1", source_type=SourceType.REVIEW_NOTE, label="Review note",
    text="Staff should reposition Willie every 2 hours and use his pressure cushion.",
)
RECORDS = [NELL_NOTE, WILLIE_REVIEW]

NOT_ACTIONS = [
    # A. Lines that are not actions at all
    "Is the Resident oriented to their new environment?: Yes",
    "2 entries on 16/11/2026",
    "Sex: F",
    "Age: 81+",
    "Special Risks – Medicines: 2",
    "24/11/2026 15:33 · Observation: Found Awake",
    "Eats rapidly: Yes",
    # B. Real actions padded with form text, scores or reasons
    "Staff should take into account Robert's sex (M) and age (81+) when providing his care.",
    "Staff should support Catherine when supporting her comfort and movement: she is F and 81+, "
    "with Special Risks – Neurological Deficit 4 and Medicines 2.",
    "Staff should weigh Agnes weekly in response to a MUST Score 1 - Medium risk",
    "Staff should record Agnes's weight on the MUST chart and a Monthly record type",
    "Staff should support Agnes with bowel care; normal bowel habit: No; prone to constipation: "
    "Yes; regulated with diet: No; takes aperients: Yes; takes a laxative: Yes.",
    "Staff should offer William help with his oral care. His overall score is 3.",
]


@pytest.mark.parametrize("text", NOT_ACTIONS)
def test_every_line_the_retest_found_is_rejected(text: str) -> None:
    assert not_an_instruction(text, RECORDS) is not None


def test_a_carer_note_pasted_as_an_action_is_rejected() -> None:
    assert not_an_instruction(NELL_NOTE.text, RECORDS) is not None


@pytest.mark.parametrize("text", [
    "Staff should reposition Willie every 2 hours and use his pressure cushion.",
    "Remind Nell to slow down and take smaller mouthfuls at every meal.",
    "Staff should weigh Agnes weekly on the same scales.",
    "Offer Kate a seat before she stands and check her lying and standing blood pressure.",
    "Staff should check whether Agnes needs her laxative each morning.",
    "Staff should arrange a same-sex carer for personal care, as Rab prefers.",
])
def test_real_instructions_are_kept(text: str) -> None:
    # A review note written as an instruction may be followed closely: that is care.
    assert not_an_instruction(text, RECORDS) is None


def _evidence() -> list[EvidenceItem]:
    return [
        EvidenceItem(id="A1", source_type=SourceType.ASSESSMENT, label="Sex", text="F",
                     risk_flagged=True),
        EvidenceItem(id="A2", source_type=SourceType.ASSESSMENT, label="Eats rapidly",
                     text="Yes", risk_flagged=True),
        NELL_NOTE,
    ]


def test_a_rejected_line_goes_back_to_the_model_and_is_never_an_action() -> None:
    draft = DraftPlanOut(risks=[], note_classes=[], not_used=[], actions=[
        ActionOut(text="Sex: F", evidence_ids=["A1"]),
        ActionOut(text="Remind Nell to slow down and take small mouthfuls at meals.",
                  evidence_ids=["A2", "C1"]),
    ])

    result = validate_draft(_evidence(), draft, resident_name="Nell")

    assert [c.text for c in result.cards] == [
        "Remind Nell to slow down and take small mouthfuls at meals."
    ]
    assert any(p.startswith('Action "Sex: F" is an assessment question') for p in result.problems)
    assert result.dropped_action_count == 1


def test_uncovered_risks_and_notes_are_not_pasted_in_as_actions() -> None:
    # Nothing covers A1, A2 or the carer note.
    draft = DraftPlanOut(risks=[], note_classes=[], not_used=[], actions=[])

    result = validate_draft(_evidence(), draft, resident_name="Nell")

    # No "Sex: F", "Eats rapidly: Yes" or carer note in Actions ...
    assert not [c for c in result.cards if c.origin == CardOrigin.SOURCE_TEXT]
    # ... the answers stay risks, so staff see them under "Not covered in Action" ...
    assert {r.evidence_id for r in result.risks} == {"A1", "A2"}
    assert all(not c.risk_evidence_ids for c in result.cards)
    # ... and the note is listed for staff to check.
    assert result.not_covered == [
        "Open Care Buddy did not write an action for this record. Check it and add one if it is "
        'needed: "Carer Note: ' + NELL_NOTE.text + '"'
    ]
    assert result.backfilled_count == 3


def test_an_uncovered_chart_is_listed_not_pasted() -> None:
    summary = EvidenceItem(
        id="T1", source_type=SourceType.OBSERVATION_SUMMARY, label="Food Chart",
        text="2 entries on 16/11/2026", linked_to_pathway=True,
    )
    entry = EvidenceItem(
        id="V1", source_type=SourceType.OBSERVATION, label="Food Chart",
        text="Breakfast: half eaten", linked_to_pathway=True,
    )
    draft = DraftPlanOut(risks=[], note_classes=[], not_used=[], actions=[])

    result = validate_draft([summary, entry], draft, resident_name="Agnes")

    assert not [c for c in result.cards if c.text == "2 entries on 16/11/2026"]
    assert any("2 entries on 16/11/2026" in line for line in result.not_covered)

"""Bladder and Bowel retest 26/11/2026 (Agnes, 81362): descriptions and reasons the earlier checks
missed. Every action is the exact text of the draft."""

import pytest

from careplan_ai.agents.care_plan.schemas import (
    ActionOut,
    DraftPlanOut,
    EvidenceItem,
    SourceType,
)
from careplan_ai.agents.care_plan.validation import strip_descriptions, validate_draft

WALKER = (
    "Staff should keep Agnes’s walker within reach at all times and place a sensor mat beside her "
    "bed at night while asleep with nocturia, responding as soon as it alerts; this supports her "
    "difficulty getting to the toilet and that she often stands up from her chair without her "
    "walker when she wants the toilet."
)
WETNESS = (
    "Staff should check for wetness as Agnes is not aware of leakage and change her pull-up pants "
    "when wet; she was upset when her pants were wet."
)
BOWELS = (
    "Staff should offer Agnes the toilet for bowels and support her when she needs it; she does "
    "not have a normal bowel habit, is prone to constipation, it is not regulated with diet and "
    "she takes an aperient."
)
FALLS_REVIEW = (
    "Staff should prompt Agnes and walk with her to the toilet, and do not let her walk on her own "
    "at night following the falls review that says she is no longer safe to walk on her own at "
    "night."
)


def test_descriptions_after_a_semicolon_are_taken_out() -> None:
    assert strip_descriptions(WALKER, "Agnes McPhail") == (
        "Staff should keep Agnes’s walker within reach at all times and place a sensor mat beside "
        "her bed at night while asleep with nocturia, responding as soon as it alerts."
    )
    assert strip_descriptions(BOWELS, "Agnes McPhail") == (
        "Staff should offer Agnes the toilet for bowels and support her when she needs it."
    )
    assert strip_descriptions(WETNESS, "Agnes McPhail").endswith("when wet.")


@pytest.mark.parametrize("text", [
    "Walk with Agnes to the toilet at night; she must not walk on her own at night.",
    "Offer Agnes the toilet every 2 hours; staff should walk with her.",
    "Help Agnes to the toilet; she needs one carer and prompting.",
    "Offer the toilet before bed; Agnes can be helped with her pull-up pants.",
    "Offer the toilet before bed, at night and after breakfast.",
])
def test_care_after_a_semicolon_is_kept(text: str) -> None:
    assert strip_descriptions(text, "Agnes McPhail") == text


def _run(texts: list[str]) -> list[str]:
    evidence = [EvidenceItem(id="A1", source_type=SourceType.ASSESSMENT,
                             label="Is the Resident aware of leakage?", text="No",
                             risk_flagged=True)]
    draft = DraftPlanOut(actions=[ActionOut(text=t, evidence_ids=["A1"]) for t in texts],
                         risks=[], note_classes=[], not_used=[])
    result = validate_draft(evidence, draft, resident_name="Agnes McPhail",
                            pathway="Bladder and Bowel Assessment", with_outcomes=False)
    assert len(result.cards) == len(texts)  # nothing dropped
    return result.problems


def test_a_reason_with_the_resident_name_is_sent_back() -> None:
    assert any('explains why ("as Agnes McPhail is' in p for p in _run([WETNESS]))


def test_the_review_as_the_reason_is_sent_back() -> None:
    assert any("following the review that says" in p for p in _run([FALLS_REVIEW]))


@pytest.mark.parametrize("text", [
    "Help Agnes to the toilet as Agnes prefers.",
    "Offer drinks as soon as Agnes wakes.",
    "Follow the dietitian's advice following the review visit.",
])
def test_care_is_not_taken_for_a_reason(text: str) -> None:
    assert not any("explains why" in p for p in _run([text]))

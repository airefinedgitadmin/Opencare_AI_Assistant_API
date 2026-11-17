"""OPENCARE-78: empty free text is ignored; actions give no reasons and are not paperwork."""

from careplan_ai.agents.care_plan.evidence import build_evidence, is_empty_free_text
from careplan_ai.agents.care_plan.prompts import AGENT_SYSTEM_PROMPT
from careplan_ai.agents.care_plan.schemas import SourceType
from careplan_ai.agents.care_plan.validation import _NOT_CARE
from careplan_ai.api.schemas.agent import StartSessionRequest


def test_empty_care_plan_and_note_lines_are_ignored() -> None:
    for empty in ["n/a", "N/A.", "n / a", "NA", "ok", "Nil", "None", "test"]:
        assert is_empty_free_text(empty), empty
    for real in ["None of her teeth are loose.", "OK with staff help.", "Nil by mouth after 8pm.",
                 "Independent.", "Dentures."]:
        assert not is_empty_free_text(real), real


def test_none_stays_a_real_assessment_answer_but_not_a_plan_line() -> None:
    request = StartSessionRequest.model_validate(
        {
            "assessmentTemplateUserId": 1,
            "assessment": {"answers": [{"questionId": "q1", "question": "Dental pain",
                                        "answer": "None"}]},
            "previousCarePlans": [{"carePlanId": 9, "action": "n/a", "outcome": "None"}],
        }
    )

    evidence = build_evidence(request)

    assert [(e.source_type, e.text) for e in evidence] == [(SourceType.ASSESSMENT, "None")]


def test_actions_with_a_reason_or_only_paperwork_are_caught() -> None:
    caught = [
        "Following review, staff should help Yoda brush his teeth.",
        "Staff should help Yoda brush, following his recent review.",
        "Staff should document that a dentist appointment has been requested.",
        "Staff should note that a referral has been sent to the dietitian.",
    ]
    for text in caught:
        assert _NOT_CARE.search(text), text
    kept = [
        "Staff should help Yoda brush his teeth morning and night.",
        "Staff should record every drink on the Fluid chart.",
        "Staff should support Yoda to attend the dentist appointment that has been requested.",
    ]
    for text in kept:
        assert not _NOT_CARE.search(text), text


def test_the_draft_rules_say_actions_give_no_reason_and_are_not_paperwork() -> None:
    assert "It never explains why" in AGENT_SYSTEM_PROMPT
    assert "never paperwork about the records" in AGENT_SYSTEM_PROMPT

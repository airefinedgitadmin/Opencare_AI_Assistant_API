"""OPENCARE-78: empty free text is ignored; actions give no reasons and are not paperwork."""

from careplan_ai.agents.care_plan.evidence import build_evidence, is_empty_free_text
from careplan_ai.agents.care_plan.prompts import AGENT_SYSTEM_PROMPT
from careplan_ai.agents.care_plan.schemas import SourceType
from careplan_ai.agents.care_plan.validation import _EQUIPMENT_STATUS, _NOT_CARE
from careplan_ai.api.schemas.agent import StartSessionRequest


def test_empty_care_plan_and_note_lines_are_ignored() -> None:
    for empty in ["n/a", "N/A.", "n / a", "NA", "ok", "Nil", "None", "test", "...", "-"]:
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
        "Staff should document that a dentist appointment has been requested.",
        "Staff should note that a referral has been sent to the dietitian.",
    ]
    for text in caught:
        assert _NOT_CARE.search(text), text
    kept = [
        "Staff should help Yoda brush his teeth morning and night.",
        "Staff should record every drink on the Fluid chart.",
        "Staff should support Yoda to attend the dentist appointment that has been requested.",
        "Staff should check Cathy's heels daily following review by the tissue viability nurse.",
    ]
    for text in kept:
        assert not _NOT_CARE.search(text), text


def test_the_draft_rules_say_actions_give_no_reason_and_are_not_paperwork() -> None:
    assert "It never explains why" in AGENT_SYSTEM_PROMPT
    assert "never paperwork about the records" in AGENT_SYSTEM_PROMPT


def test_equipment_actions_say_how_it_is_used_not_its_status() -> None:
    # OPENCARE-79
    for text in ["Staff should arrange a sensor mat as requested.",
                 "Staff should order a new walking frame as discussed",
                 "Staff should request a pressure cushion as agreed."]:
        assert _EQUIPMENT_STATUS.search(text), text
    for text in ["Staff should help Cathy get dressed each morning as agreed.",
                 "Staff should reposition Cathy two-hourly in order to protect her skin, "
                 "as agreed.",
                 "Staff should place the sensor mat by Yoda's bed at night and respond as soon "
                 "as it alerts.",
                 "Staff should help Yoda as requested by his daughter when he walks to the "
                 "lounge."]:
        assert not _EQUIPMENT_STATUS.search(text), text
    assert "An action about equipment says how staff use it" in AGENT_SYSTEM_PROMPT

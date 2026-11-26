"""OPENCARE-118: an answer the question's rule says is not a risk is never counted as one."""

from careplan_ai.agents.care_plan.evidence import build_evidence
from careplan_ai.agents.care_plan.prompts import format_evidence_line
from careplan_ai.agents.care_plan.schemas import DraftPlanOut, RiskFlagOut, SourceType
from careplan_ai.agents.care_plan.validation import validate_draft
from careplan_ai.api.schemas.agent import StartSessionRequest

AGNES_FALLS = StartSessionRequest.model_validate({
    "assessmentTemplateUserId": 1, "pathway": "Falls Assessment", "residentName": "Agnes",
    "assessment": {"completedDate": "18/11/2026", "answers": [
        {"questionId": "falls", "question": "Any recent falls?", "answer": "Yes",
         "riskFlagged": True, "riskRule": None},
        {"questionId": "oriented", "question": "Is the Resident oriented to their new "
         "environment?", "answer": "Yes", "riskFlagged": False, "riskRule": "no_risk"},
        {"questionId": "footwear", "question": "Suitable clothing and footwear?", "answer": "Yes",
         "riskFlagged": False, "riskRule": "no_risk"},
        {"questionId": "comment", "question": "Comment", "answer": "Walks with a frame",
         "riskFlagged": False, "riskRule": "info"},
    ]},
})


def test_answers_ruled_not_a_risk_are_never_counted_even_if_the_model_lists_them() -> None:
    evidence = build_evidence(AGNES_FALLS)
    ids = {e.label: e.id for e in evidence if e.source_type == SourceType.ASSESSMENT}
    # The model lists every answer as a risk (as it did on the retest).
    draft = DraftPlanOut(note_classes=[], not_used=[], actions=[], risks=[
        RiskFlagOut(evidence_id=i, reason="listed") for i in ids.values()])

    for _run in range(3):
        result = validate_draft(evidence, draft, resident_name="Agnes", pathway="Falls Assessment")
        # Only the real risk counts - the same every run.
        assert [r.evidence_id for r in result.risks] == [ids["Any recent falls?"]]


def test_the_prompt_marks_answers_that_are_not_risks() -> None:
    evidence = build_evidence(AGNES_FALLS)
    lines = {e.label: format_evidence_line(e) for e in evidence
             if e.source_type == SourceType.ASSESSMENT}
    assert "[FLAGGED RISK]" in lines["Any recent falls?"]
    assert "[NOT A RISK]" in lines["Comment"]
    assert "[NOT A RISK]" in lines["Suitable clothing and footwear?"]

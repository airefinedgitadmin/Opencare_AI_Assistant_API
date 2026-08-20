from careplan_ai.api.schemas.request import (
    AssessmentResponseItem,
    CarePlanGenerationRequest,
    PreviousCarePlanContext,
)
from careplan_ai.llm.prompts import build_generation_prompt


def test_build_generation_prompt_includes_assessment_responses() -> None:
    payload = CarePlanGenerationRequest(
        assessment_template_user_id=1,
        assessment_type="Falls Risk",
        assessment_responses=[
            AssessmentResponseItem(
                question="Has the resident fallen recently?",
                answer="Yes, twice in the last month",
            )
        ],
    )

    prompt = build_generation_prompt(payload)

    assert "Falls Risk" in prompt
    assert "Has the resident fallen recently?" in prompt
    assert "Yes, twice in the last month" in prompt


def test_build_generation_prompt_includes_previous_care_plans() -> None:
    payload = CarePlanGenerationRequest(
        assessment_template_user_id=1,
        previous_care_plans=[
            PreviousCarePlanContext(
                care_plan_title="Falls Prevention",
                action="Weekly mobility check",
                outcome="No falls since last review",
            )
        ],
    )

    prompt = build_generation_prompt(payload)

    assert "Falls Prevention" in prompt
    assert "Weekly mobility check" in prompt


def test_build_generation_prompt_handles_empty_context() -> None:
    payload = CarePlanGenerationRequest(assessment_template_user_id=1)

    prompt = build_generation_prompt(payload)

    assert "None provided" in prompt

"""Wire-format request models for the care plan generation API.

These define the contract the Java backend must satisfy when calling this
service. They stay decoupled from the internal LLM output schema
(`careplan_ai.domain.care_plan`) so the two can evolve independently - this
is context the caller sends us, not what the model produces.
"""

from pydantic import Field

from careplan_ai.api.schemas.base import CamelModel


class AssessmentResponseItem(CamelModel):
    question: str
    answer: str | None = None


class PreviousCarePlanContext(CamelModel):
    care_plan_title: str | None = None
    action: str
    outcome: str | None = None


class ReviewNoteContext(CamelModel):
    note: str
    created_date: str | None = None


class PreviousAssessmentCompletion(CamelModel):
    """One earlier COMPLETED attempt at the same assessment template by the
    same resident (distinct from assessment_responses, which is always the
    latest completion) - lets the model see how this resident's own answers
    have changed over time. Never another resident's data."""

    completed_date: str | None = None
    responses: list[AssessmentResponseItem] = Field(default_factory=list)


class CarePlanGenerationRequest(CamelModel):
    assessment_template_user_id: int
    assessment_template_id: int | None = None
    assessment_type: str | None = None
    assessment_responses: list[AssessmentResponseItem] = Field(default_factory=list)
    previous_care_plans: list[PreviousCarePlanContext] = Field(default_factory=list)
    review_notes: list[ReviewNoteContext] = Field(default_factory=list)
    previous_assessment_completions: list[PreviousAssessmentCompletion] = Field(
        default_factory=list
    )
    care_plan_title: str | None = None

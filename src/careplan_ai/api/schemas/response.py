"""Wire-format response model for the care plan generation API.

Field names mirror `AssessmentCarePlanOutDTO`/`AssessmentCarePlanResponseDTO`
in the Java `care-plan-main-backend` service so the caller can forward this
payload (after human review) straight into `POST /care-plan` with no
remapping. `rationale` is the one additive field outside that schema.
"""

from pydantic import Field

from careplan_ai.api.schemas.base import CamelModel


class CarePlanGenerationResponse(CamelModel):
    assessment_template_user_id: int
    assessment_template_id: int | None = None
    action: str
    outcome: str | None = None
    care_plan_title: str | None = None
    rationale: str | None = Field(
        default=None,
        description="Not part of the Java care-plan schema; additive context for human reviewers.",
    )

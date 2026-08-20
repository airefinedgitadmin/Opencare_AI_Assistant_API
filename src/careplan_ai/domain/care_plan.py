"""LLM-facing output schema for a generated care plan draft.

This is the structured-output target bound to the chat model via
`with_structured_output`. `extra="forbid"` means a field the model invents
surfaces as a validation error instead of being silently accepted - kept
separate from `api.schemas` so the wire contract can change independently of
what we ask the model to produce.
"""

from pydantic import BaseModel, ConfigDict, Field


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CarePlanDraft(_StrictModel):
    action: str = Field(..., min_length=1, max_length=2000)
    outcome: str | None = Field(default=None, max_length=2000)
    care_plan_title: str | None = Field(default=None, max_length=500)
    rationale: str | None = Field(
        default=None,
        max_length=1000,
        description="Brief justification tying the draft back to the given assessment/history.",
    )

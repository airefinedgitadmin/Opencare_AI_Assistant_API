"""Renders request context into the plain-text sections of the generation prompt.

Kept separate from the system prompt and few-shot examples so the "how do we
turn a CarePlanGenerationRequest into text" logic can be read/edited on its
own.
"""

from careplan_ai.api.schemas.request import (
    AssessmentResponseItem,
    CarePlanGenerationRequest,
    PreviousAssessmentCompletion,
    PreviousCarePlanContext,
    ReviewNoteContext,
)


def _format_qa(items: list[AssessmentResponseItem]) -> str:
    if not items:
        return "None provided"
    return "\n".join(
        f"- Q: {item.question}\n  A: {item.answer or 'Not answered'}" for item in items
    )


def _format_care_plans(items: list[PreviousCarePlanContext]) -> str:
    if not items:
        return "None provided"
    lines = []
    for item in items:
        title = item.care_plan_title or "Untitled"
        outcome = item.outcome or "Not recorded"
        lines.append(f"- [{title}] Action: {item.action} | Outcome: {outcome}")
    return "\n".join(lines)


def _format_review_notes(items: list[ReviewNoteContext]) -> str:
    if not items:
        return "None provided"
    lines = []
    for item in items:
        when = f" ({item.created_date})" if item.created_date else ""
        lines.append(f"- {item.note}{when}")
    return "\n".join(lines)


def _format_previous_completions(items: list[PreviousAssessmentCompletion]) -> str:
    if not items:
        return "None provided"
    blocks = []
    for item in items:
        when = item.completed_date or "date unknown"
        blocks.append(
            f"- Completed {when}:\n"
            + "\n".join(f"    {line}" for line in _format_qa(item.responses).splitlines())
        )
    return "\n".join(blocks)


def build_generation_prompt(payload: CarePlanGenerationRequest) -> str:
    return f"""## Assessment Type
{payload.assessment_type or "Not specified"}

## Assessment Responses (latest completion)
{_format_qa(payload.assessment_responses)}

## This Resident's Earlier Completions Of This Same Assessment
{_format_previous_completions(payload.previous_assessment_completions)}

## This Resident's Previous Care Plans
{_format_care_plans(payload.previous_care_plans)}

## Recent Review Notes
{_format_review_notes(payload.review_notes)}

## Title Hint
{payload.care_plan_title or "None provided - propose a short, descriptive title if appropriate"}
"""

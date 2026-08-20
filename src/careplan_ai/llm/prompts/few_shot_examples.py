"""Worked examples shown to the model before the real request.

Kept separate from the system prompt and formatting code so the example data
itself - which is long and mostly prose - doesn't crowd out the actual logic
in the other two files.

Three examples calibrate tone/conciseness/groundedness beyond what the system
prompt's rules alone achieve. Built from real request/response objects (not
hand-written duplicate strings) so they can't drift out of sync with the
actual wire format, and deliberately use ONLY fields the Java backend actually
populates - it never sends a dedicated "resident's name" field, so a name only
appears here where it would realistically occur in real data (inside a review
note's free text), and the other examples correctly use generic
"the resident"/"they" language.
"""

import json

from careplan_ai.api.schemas.request import (
    AssessmentResponseItem,
    CarePlanGenerationRequest,
    PreviousAssessmentCompletion,
    ReviewNoteContext,
)
from careplan_ai.domain.care_plan import CarePlanDraft
from careplan_ai.llm.prompts.formatters import build_generation_prompt


def _example_response_json(draft: CarePlanDraft) -> str:
    return json.dumps(draft.model_dump(exclude_none=True))


FEW_SHOT_EXAMPLES: list[tuple[str, str]] = [
    (
        build_generation_prompt(
            CarePlanGenerationRequest(
                assessment_template_user_id=0,
                assessment_type="Oral Health Assessment",
                assessment_responses=[
                    AssessmentResponseItem(
                        question="Does the resident have dentures?", answer="Yes, upper dentures"
                    ),
                    AssessmentResponseItem(
                        question="Oral cleanliness",
                        answer="Some particles/tartar/plaque in a few areas",
                    ),
                ],
                review_notes=[
                    ReviewNoteContext(
                        note="Emily has been finding it harder to manage her dentures "
                        "independently since her hand grip weakened"
                    )
                ],
            )
        ),
        _example_response_json(
            CarePlanDraft(
                action="Support Emily with her denture care each day: offer to help her take her "
                "upper dentures out and clean them with a soft brush and denture cleanser, since "
                "her grip has made this harder to manage alone. After meals, encourage and assist "
                "Emily to rinse her mouth and check around her gums for any trapped food, given "
                "the plaque noted at her last check.",
                outcome="Emily's dentures are cleaned and cared for daily even as her grip "
                "changes, and her mouth stays comfortable with reduced plaque build-up.",
                care_plan_title="Denture and Oral Care",
                rationale="Assessment notes dentures and some plaque; the review note explains why "
                "hands-on support, not just prompting, is now needed, and gives the resident's "
                "name.",
            )
        ),
    ),
    (
        build_generation_prompt(CarePlanGenerationRequest(assessment_template_user_id=0)),
        _example_response_json(
            CarePlanDraft(
                action="There isn't enough information yet to draft a specific action - no "
                "completed assessment, care-plan history, or review notes are available for this "
                "resident. Please complete a relevant assessment first, or add details here about "
                "the specific need this care plan should address.",
                outcome="Once a specific need is identified, an outcome can be set that reflects "
                "it.",
                rationale="No assessment responses, previous care plans, or review notes were "
                "provided, so there's nothing concrete to base a specific plan on.",
            )
        ),
    ),
    (
        build_generation_prompt(
            CarePlanGenerationRequest(
                assessment_template_user_id=0,
                assessment_type="Falls Risk Assessment",
                assessment_responses=[
                    AssessmentResponseItem(
                        question="Has the resident fallen in the last 3 months?", answer="Yes, once"
                    ),
                    AssessmentResponseItem(question="Uses a walking aid?", answer="Yes, a frame"),
                ],
                previous_assessment_completions=[
                    PreviousAssessmentCompletion(
                        completed_date="2026-02-01",
                        responses=[
                            AssessmentResponseItem(
                                question="Has the resident fallen in the last 3 months?",
                                answer="No",
                            ),
                            AssessmentResponseItem(question="Uses a walking aid?", answer="No"),
                        ],
                    )
                ],
            )
        ),
        _example_response_json(
            CarePlanDraft(
                action="The resident has started using a walking frame and had a fall in the last "
                "3 months, which is a change since their last falls assessment when they weren't "
                "using any aid and hadn't fallen. Support them to use their frame for all walking, "
                "keep their usual routes clear of clutter and trailing cables, and stay close "
                "during transfers in case they need steadying. Check in on their confidence with "
                "the frame and let the senior on duty know if they seem unsteady or reluctant to "
                "use it.",
                outcome="The resident moves around safely with their frame, their confidence with "
                "it improves, and they have no further falls.",
                care_plan_title="Falls Prevention with Walking Frame",
                rationale="The latest assessment shows a new fall and new frame use compared to "
                "the previous assessment (no falls, no aid) - this is a meaningful change in risk "
                "that the action addresses directly.",
            )
        ),
    ),
]

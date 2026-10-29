"""Wire-format requests for the care plan agent (X-CarePlan-Version: 2).

The start request IS the evidence pack: the backend sends the deck's three
sources ("Care Plan AI integration ppt final.pdf") plus the resident's other
notes (carer, nursing, external professional, relative) and observation chart
entries once, and the agent
keeps them in session memory for every later step (outcomes, redrafts, the
saved trace). Send everything relevant - nothing is cut silently here.
"""

from datetime import datetime
from typing import Literal

from pydantic import Field

from careplan_ai.api.schemas.base import CamelModel


class AssessmentAnswerIn(CamelModel):
    # SurveyJS question name, so the UI can link back to the question.
    question_id: str | None = None
    question: str
    # The answer as staff saw it - choice TEXT, not the stored choice value.
    answer: str | None = None


class AssessmentEvidenceIn(CamelModel):
    """The most recently completed version of the assessment (source 1)."""

    completed_date: str | None = None
    answers: list[AssessmentAnswerIn] = Field(default_factory=list)


class PreviousCarePlanIn(CamelModel):
    """An active previous care plan for the same pathway (source 2).
    action/outcome may be Quill HTML; the agent splits them into sentences."""

    care_plan_id: int | None = None
    care_plan_date: str | None = None
    title: str | None = None
    action: str
    outcome: str | None = None


class ReviewNoteIn(CamelModel):
    """A review note since the previous care plan (source 3)."""

    note_id: int | None = None
    created_date: str | None = None
    title: str | None = None
    note: str
    # True = written against this pathway's assessment; False = a note on the
    # resident's record only (screened for relevance to this pathway first).
    pathway_linked: bool = True
    # True = from before the current period (before the latest care plan):
    # background and patterns only, screened like a resident-level note.
    historical: bool = False


class RelatedCarePlanIn(CamelModel):
    """Background care plans: this pathway's ARCHIVED plans and any of the
    resident's plans for OTHER pathways - "has this happened before?"."""

    care_plan_id: int | None = None
    care_plan_date: str | None = None
    pathway: str | None = None
    status: Literal["active", "archived"] = "archived"
    title: str | None = None
    action: str
    outcome: str | None = None


NoteType = Literal["carer", "nursing", "professional", "relative"]


class NoteIn(CamelModel):
    """Any other note about the resident (Assessment_Note types 1, 3, 4, 5)."""

    note_id: int | None = None
    note_type: NoteType
    created_date: str | None = None
    # External professional visits: when the visit happened, if recorded.
    visit_date: str | None = None
    title: str | None = None
    note: str
    # External professional notes: e.g. "GP", "District nurse", "Physiotherapist".
    professional_type: str | None = None
    # Relative communications: e.g. "Daughter".
    relation: str | None = None
    pathway_linked: bool = False
    historical: bool = False


class ObservationValueIn(CamelModel):
    # The chart field's title, e.g. "Amount (ml)".
    field: str
    # The value as staff saw it - choice TEXT, not the stored choice value.
    value: str | None = None


class ObservationIn(CamelModel):
    """One entry on an observation chart (fluid, food, TPR, MUST, falls diary,
    ...). A recorded reading, not an instruction: it can support a suggestion
    but never supersedes a previous care plan action."""

    entry_id: str | None = None
    # Ref_Assessment_Type identifier, e.g. "OF_FLC".
    observation_type: str | None = None
    # The chart's name, e.g. "Fluid Chart".
    observation_name: str
    recorded_date: str | None = None
    values: list[ObservationValueIn] = Field(default_factory=list)
    # True = one of this pathway's primary charts.
    pathway_linked: bool = False
    historical: bool = False
    # The care team's rule for this pathway (backend CarePlanObservationRules):
    # primary = the pathway's own evidence; supporting = context only, never
    # acted on alone. Unset (older backends): primary when pathway_linked.
    role: Literal["primary", "supporting"] | None = None

    @property
    def is_supporting(self) -> bool:
        return self.role == "supporting" or (self.role is None and not self.pathway_linked)


class StartSessionRequest(CamelModel):
    assessment_template_user_id: int
    assessment_template_id: int | None = None
    pathway: str | None = None
    # The care home (organisation) - selects its style library (OPENCARE-51).
    organisation_id: int | None = None
    # How suggestions refer to the resident (e.g. "Cathy"); "the resident" if unset.
    resident_name: str | None = None
    assessment: AssessmentEvidenceIn | None = None
    previous_care_plans: list[PreviousCarePlanIn] = Field(default_factory=list)
    review_notes: list[ReviewNoteIn] = Field(default_factory=list)
    notes: list[NoteIn] = Field(default_factory=list)
    related_care_plans: list[RelatedCarePlanIn] = Field(default_factory=list)
    observations: list[ObservationIn] = Field(default_factory=list)


class EvidenceRequest(CamelModel):
    """Agent memory keeps references, not record text (agents/care_plan/stored.py):
    every call after the start carries the evidence pack again, rebuilt by the
    backend as of the session's start. Without it, records show as changed."""

    evidence: StartSessionRequest | None = None


class SubjectOut(CamelModel):
    """Whose evidence a session or trace needs, and as of when."""

    assessment_template_user_id: int
    evidence_as_of: datetime | None = None


class DecisionRequest(EvidenceRequest):
    target: Literal["action", "outcome"] = "action"
    # reset = undo a previous add/dismiss (the UI's Undo).
    decision: Literal["add", "dismiss", "reset"]
    # Staff's edited wording, when they changed it before adding.
    text: str | None = None


class RedraftRequest(EvidenceRequest):
    target: Literal["action", "outcome"] = "action"


class FinalizeRequest(EvidenceRequest):
    """Sent after the backend has saved the care plan, so the trace can be
    stored against the real care plan id."""

    care_plan_id: int

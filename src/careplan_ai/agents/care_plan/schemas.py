"""Care plan agent data models.

Two groups:

* `*Out` models - the exact JSON the language model must return. They hold
  wording and evidence ids only; nothing the model returns is ever shown as a
  source quote.
* Session models - the agent's working memory for one drafting session,
  persisted by `careplan_ai.memory` and returned (camelCase) to the backend.
"""

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

from careplan_ai.api.schemas.base import CamelModel
from careplan_ai.library.models import StyleExample

# --------------------------------------------------------------------------
# Language model output
# --------------------------------------------------------------------------

NoteKind = Literal["new_requirement", "update", "status", "not_relevant"]


class RiskFlagOut(BaseModel):
    evidence_id: str = Field(description="Id of the assessment answer (A...) that shows the risk.")
    reason: str = Field(description="Short phrase naming the risk, using only the answer's words.")


class NoteClassOut(BaseModel):
    evidence_id: str = Field(description="Id of a note sentence (R, C, N, G or F...).")
    kind: NoteKind = Field(
        description=(
            "new_requirement: raises a care need or action for this pathway not already in "
            "the evidence; update: changes or confirms an existing need; status: information "
            "only; not_relevant: nothing to do with this pathway."
        )
    )


class NotUsedOut(BaseModel):
    evidence_id: str = Field(description="Id of a previous care plan sentence (P...).")
    superseded_by: str = Field(
        description="Id of the note sentence (R, C, N, G or F...) that says it no longer applies."
    )
    reason: str


class ActionOut(BaseModel):
    text: str = Field(description="The suggested action, reworded only from the cited evidence.")
    evidence_ids: list[str] = Field(description="Every evidence id this action relies on.")
    # Outcomes are written later, only for the actions staff add (outcome step).
    outcome_text: str = Field(default="", description="Leave empty.")
    outcome_evidence_ids: list[str] = Field(default_factory=list, description="Leave empty.")


class OutcomeOut(BaseModel):
    ref: str = Field(description="The action's ref exactly as given, e.g. 'S1'.")
    text: str = Field(description="The intended outcome for the resident, one sentence.")
    evidence_ids: list[str] = Field(description="Ids of the quotes the outcome relies on.")


class OutcomesOut(BaseModel):
    outcomes: list[OutcomeOut]


class PatternOut(BaseModel):
    summary: str = Field(
        description="One short sentence naming the pattern, using the items' own words."
    )
    evidence_ids: list[str] = Field(description="At least two ids of the items that show it.")


class DraftPlanOut(BaseModel):
    risks: list[RiskFlagOut]
    note_classes: list[NoteClassOut]
    not_used: list[NotUsedOut]
    actions: list[ActionOut]
    patterns: list[PatternOut] = Field(default_factory=list)
    carry_over: list[str] = Field(
        default_factory=list,
        description=(
            "Ids of P items that state no care you can write as an action without guessing. "
            "Staff are shown them exactly as written."
        ),
    )


class SearchRecords(BaseModel):
    """Search ALL of this resident's records (notes, history, earlier care plans)
    by keywords. Returns the best matching records with ids and dates."""

    query: str = Field(description="Keywords, e.g. 'refused teeth brushing' or 'hoist transfer'.")


class GetRecords(BaseModel):
    """Read records by id (only ids you have been shown)."""

    ids: list[str] = Field(description="Record ids, e.g. ['C12', 'G3'].")


class SubmitPlan(DraftPlanOut):
    """Hand in the WHOLE care plan. Code checks it; if it is not accepted you are
    told exactly what to fix - then submit the whole corrected plan again."""


AGENT_TOOLS: list[type[BaseModel]] = [SearchRecords, GetRecords, SubmitPlan]


class RedraftOut(BaseModel):
    text: str = Field(description="The reworded sentence(s) with exactly the same facts.")


Verdict = Literal["supported", "overstated", "unsupported", "contradicts"]


class VerdictOut(BaseModel):
    ref: str = Field(description="The statement's ref exactly as given, e.g. 'S1' or 'S1.outcome'.")
    verdict: Verdict
    reason: str = Field(description="Empty if supported; otherwise the exact words at fault.")


class VerifyOut(BaseModel):
    verdicts: list[VerdictOut]


ReadKind = Literal["new_requirement", "update", "status"]


class RelevantRecordOut(BaseModel):
    evidence_id: str = Field(description="The record's id (the first id of its line).")
    kind: ReadKind = Field(
        description=(
            "new_requirement: raises a care need or action for this pathway; update: changes "
            "or confirms an existing need; status: useful information only."
        )
    )
    reason: str = Field(description="At most 12 words, using the record's own words.")


class ReadOut(BaseModel):
    """A reader agent's verdict on its chunk: every record, exactly once."""

    relevant: list[RelevantRecordOut]
    not_relevant: list[str] = Field(description="Ids of records with nothing for this pathway.")


# --------------------------------------------------------------------------
# Session (agent memory)
# --------------------------------------------------------------------------


class SourceType(StrEnum):
    ASSESSMENT = "assessment"
    PREVIOUS_CARE_PLAN = "previous_care_plan"
    REVIEW_NOTE = "review_note"
    CARER_NOTE = "carer_note"
    NURSING_NOTE = "nursing_note"
    PROFESSIONAL_NOTE = "professional_note"
    RELATIVE_NOTE = "relative_note"
    # Archived care plans of this pathway and care plans of other pathways -
    # background ("has this happened before?"), never a current requirement.
    RELATED_CARE_PLAN = "related_care_plan"
    # One entry on an observation chart (fluid, food, TPR, falls diary, ...) -
    # a recorded reading. Supporting evidence only: not classified like a note,
    # never supersedes a previous action, never required coverage on its own.
    OBSERVATION = "observation"
    # One line per chart, computed by code from its entries (how many, which
    # days, each figure's range and latest value) - never by a model.
    OBSERVATION_SUMMARY = "observation_summary"


NOTE_SOURCES = frozenset(
    {
        SourceType.REVIEW_NOTE,
        SourceType.CARER_NOTE,
        SourceType.NURSING_NOTE,
        SourceType.PROFESSIONAL_NOTE,
        SourceType.RELATIVE_NOTE,
    }
)


class EvidenceItem(CamelModel):
    """One quotable fact, exactly as the backend supplied it - never clipped."""

    id: str
    source_type: SourceType
    # Backend identifiers so the UI/audit can point at the real record.
    source_ref: str | None = None
    # Human label: the question, or "Previous care plan - action", etc.
    label: str
    date: str | None = None
    text: str
    # Notes and observations only: False = on the resident's record, not this
    # pathway's.
    linked_to_pathway: bool = True
    # Before the current period (older notes, earlier/other care plans):
    # context and patterns only - never a requirement on its own.
    historical: bool = False
    # Observations only: a SUPPORTING chart for this pathway (care team rules)
    # - context that can back an action alongside other evidence, never alone.
    supporting: bool = False
    # The record changed or is no longer in the backend's evidence since the AI
    # read it (agent memory keeps references, not text - agents/care_plan/stored.py).
    changed: bool = False
    # The fingerprint the record had when the AI read it - kept through every
    # load and save, so a call without the evidence can't overwrite it.
    fingerprint: str | None = Field(default=None, exclude=True)

    @property
    def is_note(self) -> bool:
        return self.source_type in NOTE_SOURCES

    @property
    def is_observation(self) -> bool:
        return self.source_type == SourceType.OBSERVATION

    @property
    def is_chart_summary(self) -> bool:
        return self.source_type == SourceType.OBSERVATION_SUMMARY


class Pattern(CamelModel):
    """A trend across the resident's records (e.g. refusals increasing), found
    by the model but only kept when it cites at least two real items and adds
    no number or name they don't contain. Shown with its quotes, like a card."""

    summary: str
    evidence_ids: list[str]


class MissingDetail(CamelModel):
    """Words from a cited source that the suggestion's wording doesn't carry -
    a prompt for staff to check nothing was dropped when combining (slide 20)."""

    evidence_id: str
    words: list[str]


class ScreenedRecord(CamelModel):
    """A record a reader agent read because it didn't fit the care plan
    agent's pack: what it found. Relevant ones go on to the care plan agent."""

    evidence_id: str
    # new_requirement | update | status | not_relevant
    kind: str
    reason: str = ""


class CardStatus(StrEnum):
    PENDING = "pending"
    ADDED = "added"
    DISMISSED = "dismissed"
    # Outcome of an action that was dismissed - never shown (deck slide 13).
    NOT_APPLICABLE = "not_applicable"


class CardOrigin(StrEnum):
    AI = "ai"
    # Added back word for word by the completeness check - no model involved.
    SOURCE_TEXT = "source_text"


class Confidence(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Suggestion(CamelModel):
    text: str
    # The latest AI (or source) wording, kept to detect staff edits.
    original_text: str
    evidence_ids: list[str]
    status: CardStatus = CardStatus.PENDING
    edited: bool = False
    confidence: Confidence = Confidence.MEDIUM
    warnings: list[str] = Field(default_factory=list)
    possibly_missing: list[MissingDetail] = Field(default_factory=list)
    # The verifier agent's verdict against the cited quotes: supported,
    # overstated, not_verified (verifier off or unavailable) or source_text
    # (the source's own words - nothing to verify).
    verification: str = "not_verified"
    # Earlier wordings, so a redraft doesn't repeat itself.
    wording_history: list[str] = Field(default_factory=list)


class ActionCard(Suggestion):
    id: str
    origin: CardOrigin = CardOrigin.AI
    # The flagged risks (assessment evidence ids) this action addresses.
    risk_evidence_ids: list[str] = Field(default_factory=list)
    outcome: Suggestion | None = None
    # The outcome step ran for this card (it may still have no outcome, if the
    # verifier rejected the one written).
    outcome_attempted: bool = False
    # Which draft action this card came from (internal - not stored or sent).
    draft_index: int | None = Field(default=None, exclude=True)


class RiskFlag(CamelModel):
    evidence_id: str
    reason: str


class NotUsedItem(CamelModel):
    evidence_id: str
    superseded_by: str
    reason: str


class SessionStatus(StrEnum):
    OPEN = "open"
    FINALIZED = "finalized"


class GenerationInfo(CamelModel):
    model_id: str
    fallback_used: bool = False
    invalid_reference_count: int = 0
    backfilled_count: int = 0
    # Suggestions dropped for borrowing details from a style example.
    copied_count: int = 0
    # Audit (OPENCARE-51): the library entries used as style examples.
    example_ids: list[str] = Field(default_factory=list)
    # How many times the care plan agent revised its draft on feedback.
    revisions: int = 0
    # Audit: what the care plan agent searched the resident's records for.
    searches: list[str] = Field(default_factory=list)
    # Statements the verifier agent rejected (dropped; sources shown instead).
    rejected_count: int = 0
    verifier_ran: bool = False
    # Reader agents: model calls made, and seconds the reading took.
    reader_calls: int = 0
    reader_seconds: float = 0.0


class CarePlanSession(CamelModel):
    session_id: str
    agent_version: int = 2
    status: SessionStatus = SessionStatus.OPEN
    assessment_template_user_id: int
    assessment_template_id: int | None = None
    pathway: str | None = None
    organisation_id: int | None = None
    resident_name: str | None = None
    created_at: datetime
    updated_at: datetime
    # Everything the backend sent, numbered - including notes the model was
    # never shown (see screened_out / omitted), so nothing silently vanishes.
    evidence: list[EvidenceItem]
    # Records the care plan agent didn't read itself, as a reader agent judged
    # them (relevant ones were then given to the care plan agent).
    screened: list[ScreenedRecord] = Field(default_factory=list)
    # Records NO agent read (a reader failed) - staff are told to check them.
    omitted: list[str] = Field(default_factory=list)
    # Plain-language messages for staff - e.g. which assessment questions are
    # unanswered, or why there are no suggestions.
    notices: list[str] = Field(default_factory=list)
    risks: list[RiskFlag] = Field(default_factory=list)
    patterns: list[Pattern] = Field(default_factory=list)
    not_used: list[NotUsedItem] = Field(default_factory=list)
    cards: list[ActionCard] = Field(default_factory=list)
    style_examples: list[StyleExample] = Field(default_factory=list)
    generation: GenerationInfo
    care_plan_id: int | None = None


# ---- navigation helpers for the one-by-one UI (deck slides 11-13) ----


def next_action_card_id(session: CarePlanSession) -> str | None:
    return next((c.id for c in session.cards if c.status == CardStatus.PENDING), None)


def next_outcome_card_id(session: CarePlanSession) -> str | None:
    return next(
        (
            c.id
            for c in session.cards
            if c.status == CardStatus.ADDED
            and c.outcome is not None
            and c.outcome.status == CardStatus.PENDING
        ),
        None,
    )


def outcomes_to_write(session: CarePlanSession) -> list[str]:
    """Added actions whose outcome hasn't been written yet (outcome step)."""
    return [
        c.id
        for c in session.cards
        if c.status == CardStatus.ADDED and c.outcome is None and not c.outcome_attempted
    ]


class SessionView(CarePlanSession):
    """What the API returns: the session plus where the UI is up to."""

    next_action_card_id: str | None = None
    next_outcome_card_id: str | None = None
    # Added actions still waiting for their outcome - the UI calls the outcome
    # step (POST .../outcomes) when staff open the Outcome field.
    outcomes_to_write: list[str] = Field(default_factory=list)
    # True once every action and outcome prompt has been added or dismissed -
    # the UI then shows "That's all the AI prompts for now!".
    all_prompts_handled: bool = False

    @classmethod
    def of(cls, session: CarePlanSession) -> "SessionView":
        next_action = next_action_card_id(session)
        next_outcome = next_outcome_card_id(session)
        to_write = outcomes_to_write(session)
        return cls(
            **session.model_dump(),
            next_action_card_id=next_action,
            next_outcome_card_id=next_outcome,
            outcomes_to_write=to_write,
            all_prompts_handled=next_action is None and next_outcome is None and not to_write,
        )


class TraceStatement(CamelModel):
    card_id: str
    field: Literal["action", "outcome"]
    text: str
    origin: CardOrigin
    edited: bool
    confidence: Confidence
    verification: str = "not_verified"
    evidence: list[EvidenceItem]


class CarePlanTrace(CamelModel):
    """Saved with the care plan so "View" works on every saved plan (slide 20)."""

    care_plan_id: int
    session_id: str
    assessment_template_user_id: int
    pathway: str | None = None
    created_at: datetime
    # When the evidence was read (the draft's start): the backend rebuilds the
    # evidence as of then to show the quotes (agents/care_plan/stored.py).
    evidence_as_of: datetime | None = None
    model_id: str
    statements: list[TraceStatement]
    not_used: list[NotUsedItem]
    not_used_evidence: list[EvidenceItem]
    example_ids: list[str] = Field(default_factory=list)
    omitted_count: int = 0
    # Records a reader agent judged not relevant to this pathway.
    not_relevant_count: int = 0

"""Care plan agent session operations - the deck's flow, one function per step.

start      evidence pack in -> drafted, checked suggestion cards out
decide     staff Add / Dismiss (or Undo) one action or outcome
redraft    reword ONE card from the same evidence; nothing else changes
finalize   after the backend saves the care plan: store the per-statement trace
"""

import uuid

from careplan_ai.agents.care_plan import llm
from careplan_ai.agents.care_plan.evidence import build_evidence, unanswered_questions
from careplan_ai.agents.care_plan.graph import run_draft
from careplan_ai.agents.care_plan.schemas import (
    ActionCard,
    CardOrigin,
    CardStatus,
    CarePlanSession,
    CarePlanTrace,
    EvidenceItem,
    GenerationInfo,
    SessionStatus,
    SessionView,
    SourceType,
    Suggestion,
    TraceStatement,
    VerdictOut,
    outcomes_to_write,
)
from careplan_ai.agents.care_plan.validation import (
    UNSUPPORTED_NAME_WARNING,
    UNSUPPORTED_NUMBER_WARNING,
    borrowed_terms,
    confidence_for,
    detail_warnings,
    stems,
)
from careplan_ai.agents.care_plan.verify import (
    NOT_VERIFIED_WARNING,
    REJECT,
    overstated_warning,
)
from careplan_ai.api.schemas.agent import DecisionRequest, StartSessionRequest
from careplan_ai.core.config import get_settings
from careplan_ai.core.exceptions import (
    InvalidRequestError,
    SessionNotFoundError,
    SessionStateError,
)
from careplan_ai.core.logging import get_logger
from careplan_ai.library.service import mark_ai_assisted
from careplan_ai.memory.store import get_session_store, utcnow

logger = get_logger(__name__)

REDRAFT_REJECTED_WARNING = (
    "Redraft discarded: the new wording wasn't confirmed by its sources."
)


def _normalise(text: str) -> str:
    return " ".join(text.split()).casefold()


MAX_UNANSWERED_LISTED = 15


def _notices(
    request: StartSessionRequest, evidence: list[EvidenceItem], *, drafted: bool
) -> list[str]:
    """What staff need to know about the records behind the suggestions."""
    pathway = request.pathway or "this assessment"
    notices: list[str] = []
    answered = any(e.source_type == SourceType.ASSESSMENT for e in evidence)
    if not drafted:
        notices.append(
            "There are no records to suggest care plan actions from yet."
        )
    if not answered:
        notices.append(
            f"{pathway} has no answers yet - complete it first: suggestions are built "
            "from its answers."
        )
    unanswered = unanswered_questions(request)
    if unanswered:
        listed = "; ".join(unanswered[:MAX_UNANSWERED_LISTED])
        more = len(unanswered) - MAX_UNANSWERED_LISTED
        notices.append(
            f"Not answered in {pathway}: {listed}" + (f"; and {more} more." if more > 0 else ".")
        )
    return notices


async def start_session(request: StartSessionRequest) -> SessionView:
    evidence = build_evidence(request)
    if not evidence and not unanswered_questions(request):
        raise InvalidRequestError(
            "No evidence to draft from - send the assessment, a previous care plan or a note"
        )
    now = utcnow()

    if not evidence:
        # Only unanswered questions: nothing to draft from, so tell staff which
        # questions to answer instead of calling a model.
        session = CarePlanSession(
            session_id=str(uuid.uuid4()),
            assessment_template_user_id=request.assessment_template_user_id,
            assessment_template_id=request.assessment_template_id,
            pathway=request.pathway,
            organisation_id=request.organisation_id,
            resident_name=request.resident_name,
            created_at=now,
            updated_at=now,
            evidence=[],
            notices=_notices(request, evidence, drafted=False),
            generation=GenerationInfo(model_id="none"),
        )
        await get_session_store().save_session(session)
        logger.info("care_plan_session_not_drafted", session_id=session.session_id)
        return SessionView.of(session)

    outcome = await run_draft(
        resident_name=request.resident_name,
        pathway=request.pathway,
        organisation_id=request.organisation_id,
        evidence=evidence,
    )
    result = outcome.result
    session = CarePlanSession(
        session_id=str(uuid.uuid4()),
        assessment_template_user_id=request.assessment_template_user_id,
        assessment_template_id=request.assessment_template_id,
        pathway=request.pathway,
        organisation_id=request.organisation_id,
        resident_name=request.resident_name,
        created_at=now,
        updated_at=now,
        evidence=evidence,
        omitted=outcome.omitted,
        notices=_notices(request, evidence, drafted=True),
        risks=result.risks,
        patterns=outcome.patterns,
        not_used=result.not_used,
        cards=result.cards,
        style_examples=outcome.examples,
        generation=GenerationInfo(
            model_id=outcome.model_id,
            fallback_used=outcome.fallback_used,
            invalid_reference_count=result.invalid_reference_count,
            backfilled_count=result.backfilled_count,
            copied_count=result.copied_count,
            example_ids=[e.id for e in outcome.examples],
            revisions=outcome.revisions,
            searches=outcome.searches,
            rejected_count=outcome.rejected,
            verifier_ran=outcome.verifier_ran,
        ),
    )
    await get_session_store().save_session(session)
    logger.info(
        "care_plan_session_started",
        session_id=session.session_id,
        assessment_template_user_id=session.assessment_template_user_id,
        evidence_items=len(evidence),
        omitted=len(session.omitted),
        revisions=outcome.revisions,
        rejected=outcome.rejected,
        verifier_ran=outcome.verifier_ran,
        examples=len(outcome.examples),
        cards=len(session.cards),
        model_id=outcome.model_id,
        fallback_used=outcome.fallback_used,
    )
    return SessionView.of(session)


async def _load_open(session_id: str) -> CarePlanSession:
    session = await get_session_store().get_session(session_id)
    if session is None:
        raise SessionNotFoundError("Session not found or expired")
    if session.status != SessionStatus.OPEN:
        raise SessionStateError("Session is already finalized")
    return session


def _card(session: CarePlanSession, card_id: str) -> ActionCard:
    card = next((c for c in session.cards if c.id == card_id), None)
    if card is None:
        raise SessionNotFoundError(f"Card {card_id} not found in this session")
    return card


async def get_session(session_id: str) -> SessionView:
    session = await get_session_store().get_session(session_id)
    if session is None:
        raise SessionNotFoundError("Session not found or expired")
    return SessionView.of(session)


def _apply_text(target: Suggestion, text: str | None) -> None:
    if text is None or not text.strip():
        return
    target.text = text.strip()
    target.edited = _normalise(target.text) != _normalise(target.original_text)


async def decide(session_id: str, card_id: str, request: DecisionRequest) -> SessionView:
    session = await _load_open(session_id)
    card = _card(session, card_id)

    if request.target == "action":
        if request.decision == "add":
            _apply_text(card, request.text)
            card.status = CardStatus.ADDED
            if card.outcome is not None and card.outcome.status == CardStatus.NOT_APPLICABLE:
                card.outcome.status = CardStatus.PENDING
        elif request.decision == "dismiss":
            card.status = CardStatus.DISMISSED
            # Deck slide 13: a dismissed action gets no outcome.
            if card.outcome is not None:
                card.outcome.status = CardStatus.NOT_APPLICABLE
        else:  # reset (Undo)
            card.status = CardStatus.PENDING
            if card.outcome is not None:
                card.outcome.status = CardStatus.PENDING
    else:
        outcome = card.outcome
        if outcome is None:
            raise SessionStateError(f"Card {card_id} has no suggested outcome")
        # Deck slide 13: outcomes only appear once their action is added.
        if card.status != CardStatus.ADDED:
            raise SessionStateError("Add the action before deciding its outcome")
        if request.decision == "add":
            _apply_text(outcome, request.text)
            outcome.status = CardStatus.ADDED
        elif request.decision == "dismiss":
            outcome.status = CardStatus.DISMISSED
        else:
            outcome.status = CardStatus.PENDING

    session.updated_at = utcnow()
    await get_session_store().save_session(session)
    return SessionView.of(session)


async def write_outcomes(session_id: str) -> SessionView:
    """The outcome step (deck slide 13): outcomes are written only for actions
    staff have added, when staff open the Outcome field - one call for all of
    them, then the same code checks and the verifier agent as everything else.
    A rejected outcome is simply not offered."""
    session = await _load_open(session_id)
    cards = [c for c in session.cards if c.id in set(outcomes_to_write(session))]
    if not cards:
        return SessionView.of(session)

    by_id = {e.id: e for e in session.evidence}
    # Previous outcomes for this pathway are useful wording for the new goal.
    previous_outcomes = [e for e in session.evidence if e.id.startswith("O")]
    quotes: dict[str, list[EvidenceItem]] = {}
    for card in cards:
        own = [by_id[i] for i in card.evidence_ids if i in by_id]
        quotes[card.id] = own + [e for e in previous_outcomes if e not in own]
    context = [s for s in (session.resident_name, session.pathway) if s]
    resident_stems = stems([e.text for e in session.evidence] + context)
    example_stems = stems([e.text for e in session.style_examples])

    async def attempt(
        todo: list[ActionCard], feedback: dict[str, str]
    ) -> tuple[dict[str, Suggestion], dict[str, VerdictOut] | None]:
        """Write -> code checks -> verifier, for these cards."""
        out = await llm.write_outcomes(
            resident_name=session.resident_name,
            actions=[(c.id, c.text, quotes[c.id]) for c in todo],
            examples=session.style_examples,
            feedback=feedback,
        )
        written = {o.ref.strip(): o for o in out.outcomes}
        drafts: dict[str, Suggestion] = {}
        for card in todo:
            o = written.get(card.id)
            text = " ".join(o.text.split()) if o else ""
            allowed = {e.id for e in quotes[card.id]}
            ids = [i for i in dict.fromkeys(o.evidence_ids) if i in allowed] if o else []
            ids = ids or card.evidence_ids
            sources = [by_id[i].text for i in allowed] + [by_id[i].label for i in allowed]
            sources += [card.text] + context
            if not text or borrowed_terms(text, source_stems=resident_stems,
                                          example_stems=example_stems):
                continue
            warnings = detail_warnings(text, sources)
            drafts[card.id] = Suggestion(text=text, original_text=text, evidence_ids=ids,
                                         warnings=warnings,
                                         confidence=confidence_for(ids, by_id, warnings))
        if not drafts or not get_settings().agent_verify:
            return drafts, None
        try:
            verified = await llm.verify_statements(
                resident_name=session.resident_name,
                statements=[(f"{cid}.outcome", s.text, quotes[cid]) for cid, s in drafts.items()],
            )
        except Exception as exc:  # noqa: BLE001 - mark not verified rather than fail
            logger.warning("care_plan_outcome_verifier_failed", error=type(exc).__name__)
            return drafts, None
        return drafts, {v.ref.strip(): v for v in verified.verdicts}

    drafts, verdicts = await attempt(cards, {})
    # An outcome the verifier rejects is rewritten once, with its reason.
    retry = {
        c.id: f"{drafts[c.id].text} - {verdicts[f'{c.id}.outcome'].reason}"
        for c in cards
        if verdicts and c.id in drafts and f"{c.id}.outcome" in verdicts
        and verdicts[f"{c.id}.outcome"].verdict in REJECT
    }
    if retry:
        again, again_verdicts = await attempt([c for c in cards if c.id in retry], retry)
        drafts.update(again)
        if verdicts is not None:
            if again_verdicts is None:  # rewrites couldn't be verified
                for cid in retry:
                    verdicts.pop(f"{cid}.outcome", None)
            else:
                verdicts.update(again_verdicts)

    for card in cards:
        card.outcome_attempted = True
        outcome = drafts.get(card.id)
        if outcome is None:
            continue
        verdict = None if verdicts is None else verdicts.get(f"{card.id}.outcome")
        if verdict is not None and verdict.verdict in REJECT:
            continue  # still unsupported after a rewrite: not offered
        if verdict is None:
            outcome.verification = "not_verified"
            outcome.warnings.append(NOT_VERIFIED_WARNING)
        elif verdict.verdict == "supported":
            outcome.verification = "supported"
        else:
            outcome.verification = "overstated"
            outcome.warnings.append(overstated_warning(verdict.reason.strip()))
        if outcome.warnings:
            outcome.confidence = confidence_for(outcome.evidence_ids, by_id, outcome.warnings)
        card.outcome = outcome

    session.updated_at = utcnow()
    await get_session_store().save_session(session)
    logger.info("care_plan_outcomes_written", session_id=session_id, actions=len(cards),
                offered=sum(1 for c in cards if c.outcome is not None))
    return SessionView.of(session)


async def _verify_one(resident_name: str | None, text: str, quotes: list[EvidenceItem]) -> str:
    try:
        out = await llm.verify_statements(
            resident_name=resident_name, statements=[("S1", text, quotes)]
        )
    except Exception as exc:  # noqa: BLE001 - unverifiable wording is simply not accepted
        logger.warning("care_plan_redraft_verifier_failed", error=type(exc).__name__)
        return "not_verified"
    verdict = next((v for v in out.verdicts if v.ref.strip() == "S1"), None)
    return verdict.verdict if verdict is not None else "not_verified"


async def redraft(session_id: str, card_id: str, target: str) -> SessionView:
    """New wording for one action or outcome, from exactly the same evidence.

    Every other card is untouched. If the new wording introduces a number that
    isn't in the sources, it is discarded and the current wording kept."""
    session = await _load_open(session_id)
    card = _card(session, card_id)
    suggestion: Suggestion
    if target == "outcome":
        if card.outcome is None:
            raise SessionStateError(f"Card {card_id} has no suggested outcome")
        suggestion = card.outcome
    else:
        suggestion = card
    if suggestion.status in (CardStatus.DISMISSED, CardStatus.NOT_APPLICABLE):
        raise SessionStateError("Dismissed suggestions can't be redrafted")

    by_id = {e.id: e for e in session.evidence}
    evidence: list[EvidenceItem] = [by_id[i] for i in suggestion.evidence_ids if i in by_id]
    if target == "outcome":
        # An outcome may restate its action, so the action's evidence counts too.
        evidence += [
            by_id[i] for i in card.evidence_ids if i in by_id and i not in suggestion.evidence_ids
        ]

    new_text = await llm.generate_redraft(
        resident_name=session.resident_name,
        target=target,
        current_text=suggestion.text,
        earlier_wordings=suggestion.wording_history,
        evidence=evidence,
        action_text=card.text if target == "outcome" else None,
        examples=session.style_examples,
    )

    sources = [e.text for e in evidence] + [e.label for e in evidence]
    sources += [suggestion.text] + [s for s in (session.resident_name, session.pathway) if s]
    if target == "outcome":
        sources.append(card.text)
    borrowed = bool(new_text) and borrowed_terms(
        new_text,
        source_stems=stems(sources + [e.text for e in session.evidence]),
        example_stems=stems([e.text for e in session.style_examples]),
    )
    rejected = not new_text or bool(borrowed) or bool(detail_warnings(new_text, sources))
    verification = "not_verified"
    if not rejected and get_settings().agent_verify:
        # The verifier agent checks the new wording against the same quotes;
        # only a supported wording replaces the current one.
        verification = await _verify_one(session.resident_name, new_text, evidence)
        rejected = verification != "supported"
    if rejected:
        if REDRAFT_REJECTED_WARNING not in suggestion.warnings:
            suggestion.warnings.append(REDRAFT_REJECTED_WARNING)
        logger.warning("care_plan_redraft_rejected", session_id=session_id, card_id=card_id,
                       verification=verification)
    else:
        suggestion.wording_history.append(suggestion.text)
        suggestion.text = new_text
        suggestion.original_text = new_text
        suggestion.edited = False
        suggestion.warnings = [
            w for w in suggestion.warnings
            if w not in (REDRAFT_REJECTED_WARNING, UNSUPPORTED_NUMBER_WARNING,
                         UNSUPPORTED_NAME_WARNING, NOT_VERIFIED_WARNING)
            and not w.startswith("May say more than its sources do")
        ]
        suggestion.verification = verification
        if verification != "supported":
            suggestion.warnings.append(NOT_VERIFIED_WARNING)
        suggestion.confidence = confidence_for(suggestion.evidence_ids, by_id, suggestion.warnings)
        if isinstance(suggestion, ActionCard) and suggestion.origin == CardOrigin.SOURCE_TEXT:
            suggestion.origin = CardOrigin.AI

    session.updated_at = utcnow()
    await get_session_store().save_session(session)
    return SessionView.of(session)


async def finalize(session_id: str, care_plan_id: int) -> CarePlanTrace:
    """Called after the backend has saved the care plan. Stores, per added
    statement, its final text and the evidence quotes behind it."""
    session = await _load_open(session_id)
    by_id = {e.id: e for e in session.evidence}

    statements: list[TraceStatement] = []
    for card in session.cards:
        if card.status != CardStatus.ADDED:
            continue
        statements.append(
            TraceStatement(
                card_id=card.id,
                field="action",
                text=card.text,
                origin=card.origin,
                edited=card.edited,
                confidence=card.confidence,
                verification=card.verification,
                evidence=[by_id[i] for i in card.evidence_ids if i in by_id],
            )
        )
        if card.outcome is not None and card.outcome.status == CardStatus.ADDED:
            statements.append(
                TraceStatement(
                    card_id=card.id,
                    field="outcome",
                    text=card.outcome.text,
                    origin=card.origin,
                    edited=card.outcome.edited,
                    confidence=card.outcome.confidence,
                    verification=card.outcome.verification,
                    evidence=[by_id[i] for i in card.outcome.evidence_ids if i in by_id],
                )
            )

    not_used_ids = {n.evidence_id for n in session.not_used}
    not_used_ids |= {n.superseded_by for n in session.not_used}
    trace = CarePlanTrace(
        care_plan_id=care_plan_id,
        session_id=session.session_id,
        assessment_template_user_id=session.assessment_template_user_id,
        pathway=session.pathway,
        created_at=utcnow(),
        model_id=session.generation.model_id,
        statements=statements,
        not_used=session.not_used,
        not_used_evidence=[e for e in session.evidence if e.id in not_used_ids],
        example_ids=session.generation.example_ids,
        omitted_count=len(session.omitted),
    )
    store = get_session_store()
    await store.save_trace(trace)
    await mark_ai_assisted(
        session.organisation_id, care_plan_id, edited=any(s.edited for s in statements)
    )
    session.status = SessionStatus.FINALIZED
    session.care_plan_id = care_plan_id
    session.updated_at = utcnow()
    await store.save_session(session)
    logger.info(
        "care_plan_session_finalized",
        session_id=session_id,
        care_plan_id=care_plan_id,
        statements=len(statements),
    )
    return trace


async def get_trace(care_plan_id: int) -> CarePlanTrace:
    trace = await get_session_store().get_trace(care_plan_id)
    if trace is None:
        raise SessionNotFoundError("No AI trace for this care plan")
    return trace

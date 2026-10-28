"""The verifier agent: every AI-written statement is checked by a second model
call against ONLY the quotes it cites.

    supported               -> shown, marked verified
    overstated / unsupported / contradicts
                            -> sent back to the care plan agent with the reason
                               (while revisions are left); if still not
                               supported after that: overstated is shown with a
                               warning and low confidence, the others are
                               dropped - and the completeness check then shows
                               the sources they covered in their own words.

Statements that are the source's own words (backfilled) need no verifying. If
the verifier can't run (switched off, out of quota) nothing is dropped: every
AI statement is marked "not verified" with a warning, so staff know.
"""

from careplan_ai.agents.care_plan import llm
from careplan_ai.agents.care_plan.schemas import (
    ActionOut,
    CardOrigin,
    Confidence,
    DraftPlanOut,
    EvidenceItem,
    Suggestion,
    VerdictOut,
)
from careplan_ai.agents.care_plan.validation import ValidationResult
from careplan_ai.core.logging import get_logger

logger = get_logger(__name__)

NOT_VERIFIED_WARNING = "Not checked against its sources by the verifier - check before adding."
REJECT = ("unsupported", "contradicts")


def overstated_warning(reason: str) -> str:
    return f"May say more than its sources do: {reason}".rstrip(": ")


def statements_for(
    result: ValidationResult, by_id: dict[str, EvidenceItem]
) -> list[tuple[str, str, list[EvidenceItem]]]:
    """(ref, text, cited quotes) for every AI-written action and outcome."""
    statements: list[tuple[str, str, list[EvidenceItem]]] = []
    for card in result.cards:
        if card.origin != CardOrigin.AI:
            continue
        statements.append((card.id, card.text, [by_id[i] for i in card.evidence_ids]))
        if card.outcome is not None:
            # An outcome is the intended result of its action, so the action's
            # quotes back it too.
            ids = list(dict.fromkeys(card.outcome.evidence_ids + card.evidence_ids))
            statements.append(
                (f"{card.id}.outcome", card.outcome.text, [by_id[i] for i in ids if i in by_id])
            )
    return statements


async def run_verifier(
    *, resident_name: str | None, result: ValidationResult, evidence: list[EvidenceItem]
) -> dict[str, VerdictOut] | None:
    """Verdict per statement ref; None if the verifier couldn't run. A
    statement the verifier skipped counts as not verified, never as supported."""
    statements = statements_for(result, {e.id: e for e in evidence})
    if not statements:
        return {}
    try:
        out = await llm.verify_statements(resident_name=resident_name, statements=statements)
    except Exception as exc:  # noqa: BLE001 - mark not verified rather than fail the session
        logger.warning("care_plan_verifier_failed", error=type(exc).__name__)
        return None
    refs = {ref for ref, _, _ in statements}
    verdicts = {v.ref.strip(): v for v in out.verdicts if v.ref.strip() in refs}
    logger.info(
        "care_plan_verified",
        statements=len(statements),
        supported=sum(v.verdict == "supported" for v in verdicts.values()),
        not_supported=sum(v.verdict != "supported" for v in verdicts.values()),
        missing=len(refs - verdicts.keys()),
    )
    return verdicts


def feedback_from(verdicts: dict[str, VerdictOut], result: ValidationResult) -> list[str]:
    """The verifier's objections, in words the care plan agent can act on."""
    texts = {c.id: c.text for c in result.cards}
    texts |= {f"{c.id}.outcome": c.outcome.text for c in result.cards if c.outcome}
    return [
        f"{'Outcome' if ref.endswith('.outcome') else 'Action'} \"{texts.get(ref, ref)[:80]}\" "
        f"is {v.verdict} by its quotes: {v.reason} - reword it to say only what its quotes "
        "say, or cite the evidence that does."
        for ref, v in verdicts.items()
        if v.verdict != "supported"
    ]


def rejected_draft(
    draft: DraftPlanOut, result: ValidationResult, verdicts: dict[str, VerdictOut]
) -> tuple[DraftPlanOut, int, dict[int, int]]:
    """The draft without the actions / outcomes the verifier rejected, how many
    were rejected, and old -> new action index for the actions kept."""
    drop_actions: set[int] = set()
    drop_outcomes: set[int] = set()
    for card in result.cards:
        if card.draft_index is None:
            continue
        verdict = verdicts.get(card.id)
        if verdict is not None and verdict.verdict in REJECT:
            drop_actions.add(card.draft_index)
        outcome_verdict = verdicts.get(f"{card.id}.outcome")
        if outcome_verdict is not None and outcome_verdict.verdict in REJECT:
            drop_outcomes.add(card.draft_index)
    actions: list[ActionOut] = []
    index_map: dict[int, int] = {}
    for index, action in enumerate(draft.actions):
        if index in drop_actions:
            continue
        if index in drop_outcomes:
            action = action.model_copy(update={"outcome_text": "", "outcome_evidence_ids": []})
        index_map[index] = len(actions)
        actions.append(action)
    rejected = len(drop_actions) + len(drop_outcomes - drop_actions)
    return draft.model_copy(update={"actions": actions}), rejected, index_map


def _mark(suggestion: Suggestion, verdict: VerdictOut | None) -> None:
    if verdict is None:
        suggestion.verification = "not_verified"
        suggestion.warnings.append(NOT_VERIFIED_WARNING)
        suggestion.confidence = Confidence.LOW
    elif verdict.verdict == "supported":
        suggestion.verification = "supported"
    else:  # overstated (rejected ones were already removed)
        suggestion.verification = "overstated"
        suggestion.warnings.append(overstated_warning(verdict.reason.strip()))
        suggestion.confidence = Confidence.LOW


def mark_cards(
    result: ValidationResult, verdicts_by_index: dict[tuple[int, str], VerdictOut] | None
) -> None:
    """Record each AI statement's verdict on its card. Keys are (draft action
    index, "action" | "outcome") so they survive the cards being renumbered."""
    for card in result.cards:
        if card.origin != CardOrigin.AI or card.draft_index is None:
            continue
        lookup = verdicts_by_index or {}
        if verdicts_by_index is None:
            _mark(card, None)
        else:
            _mark(card, lookup.get((card.draft_index, "action")))
        if card.outcome is not None:
            _mark(
                card.outcome,
                None if verdicts_by_index is None
                else lookup.get((card.draft_index, "outcome")),
            )


def by_draft_index(
    result: ValidationResult, verdicts: dict[str, VerdictOut], index_map: dict[int, int]
) -> dict[tuple[int, str], VerdictOut]:
    """Verdicts keyed by the action's index in the draft after rejections."""
    keyed: dict[tuple[int, str], VerdictOut] = {}
    for card in result.cards:
        if card.draft_index is None or card.draft_index not in index_map:
            continue
        new_index = index_map[card.draft_index]
        if card.id in verdicts:
            keyed[(new_index, "action")] = verdicts[card.id]
        if f"{card.id}.outcome" in verdicts:
            keyed[(new_index, "outcome")] = verdicts[f"{card.id}.outcome"]
    return keyed

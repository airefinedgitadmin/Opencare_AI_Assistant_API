from careplan_ai.agents.care_plan.schemas import (
    ActionOut,
    DraftPlanOut,
    EvidenceItem,
    PatternOut,
    RiskFlagOut,
    SourceType,
    VerdictOut,
)
from careplan_ai.agents.care_plan.validation import validate_draft, validate_patterns
from careplan_ai.agents.care_plan.verify import (
    by_draft_index,
    feedback_from,
    mark_cards,
    rejected_draft,
    statements_for,
)

EVIDENCE = [
    EvidenceItem(id="A1", source_type=SourceType.ASSESSMENT, label="Dentures", text="Yes"),
    EvidenceItem(id="A2", source_type=SourceType.ASSESSMENT, label="Gums", text="Sore"),
    EvidenceItem(id="P1", source_type=SourceType.PREVIOUS_CARE_PLAN, label="Previous",
                 text="Prompt denture cleaning."),
    EvidenceItem(id="R1", source_type=SourceType.REVIEW_NOTE, label="Review note",
                 text="Denture cleaning now needs full help."),
]
BY_ID = {e.id: e for e in EVIDENCE}


def _action(text: str, ids: list[str], outcome: str = "") -> ActionOut:
    return ActionOut(text=text, evidence_ids=ids, outcome_text=outcome,
                     outcome_evidence_ids=ids if outcome else [])


DRAFT = DraftPlanOut(risks=[RiskFlagOut(evidence_id="A2", reason="Sore gums")],
                     note_classes=[], not_used=[], actions=[
    _action("Staff should give full help with denture cleaning.", ["P1", "R1"],
            "Cathy's dentures will be clean."),
    _action("Staff should check Cathy's sore gums.", ["A2"]),
])


def _verdict(ref: str, verdict: str, reason: str = "") -> VerdictOut:
    return VerdictOut(ref=ref, verdict=verdict, reason=reason)  # type: ignore[arg-type]


def test_each_statement_is_verified_against_only_its_own_quotes() -> None:
    result = validate_draft(EVIDENCE, DRAFT, resident_name="Cathy")

    statements = statements_for(result, BY_ID)

    assert [(ref, [q.id for q in quotes]) for ref, _, quotes in statements] == [
        ("S1", ["P1", "R1"]), ("S1.outcome", ["P1", "R1"]), ("S2", ["A2"]),
    ]


def test_rejected_statements_are_removed_and_their_sources_come_back_word_for_word() -> None:
    result = validate_draft(EVIDENCE, DRAFT, resident_name="Cathy")
    verdicts = {"S1": _verdict("S1", "supported"),
                "S1.outcome": _verdict("S1.outcome", "unsupported", "'clean'"),
                "S2": _verdict("S2", "contradicts", "'sore gums'")}

    draft, rejected, index_map = rejected_draft(DRAFT, result, verdicts)
    final = validate_draft(EVIDENCE, draft, resident_name="Cathy")
    mark_cards(final, by_draft_index(result, verdicts, index_map))

    assert rejected == 2
    assert [c.text for c in final.cards if c.origin == "ai"] == [DRAFT.actions[0].text]
    assert final.cards[0].outcome is None  # the unsupported outcome is gone
    assert final.cards[0].verification == "supported"
    # The flagged risk S2 covered comes back in the source's own words.
    backfilled = [c for c in final.cards if c.origin == "source_text"]
    assert [(c.evidence_ids, c.text, c.verification) for c in backfilled] == [
        (["A2"], "Gums: Sore", "source_text")
    ]


def test_feedback_names_the_statement_and_the_words_at_fault() -> None:
    result = validate_draft(EVIDENCE, DRAFT, resident_name="Cathy")

    feedback = feedback_from({"S2": _verdict("S2", "overstated", "'check' - A2 only says sore"),
                              "S1": _verdict("S1", "supported")}, result)

    assert feedback == [
        "Action \"Staff should check Cathy's sore gums.\" is overstated by its quotes: "
        "'check' - A2 only says sore - reword it to say only what its quotes say, or cite the "
        "evidence that does."
    ]


def test_a_statement_the_verifier_skipped_is_not_treated_as_supported() -> None:
    result = validate_draft(EVIDENCE, DRAFT, resident_name="Cathy")

    mark_cards(result, by_draft_index(result, {"S1": _verdict("S1", "supported")}, {0: 0, 1: 1}))

    assert result.cards[0].verification == "supported"
    assert result.cards[0].outcome is not None
    assert result.cards[0].outcome.verification == "not_verified"
    assert result.cards[1].verification == "not_verified"


def test_patterns_need_two_real_items_and_no_invented_details() -> None:
    proposed = [
        PatternOut(summary="Denture cleaning has needed more help over time.",
                   evidence_ids=["P1", "R1"]),
        PatternOut(summary="Sore gums.", evidence_ids=["A2"]),  # one item only
        PatternOut(summary="Help needed 3 times since March.", evidence_ids=["P1", "R1"]),
        PatternOut(summary="Made up.", evidence_ids=["X1", "X2"]),
    ]

    kept = validate_patterns(proposed, EVIDENCE, resident_name="Cathy")

    assert [(p.summary, p.evidence_ids) for p in kept] == [
        ("Denture cleaning has needed more help over time.", ["P1", "R1"])
    ]

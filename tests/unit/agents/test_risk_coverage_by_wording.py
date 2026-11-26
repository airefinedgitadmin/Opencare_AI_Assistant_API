"""'Not covered' retest 26/11/2026 (Nell, 81366, Falls): a flagged question an action deals with in
its own words is linked to that action, so staff don't see "Not covered" for a risk the plan covers.
Risks the plan doesn't deal with still show."""

from careplan_ai.agents.care_plan.schemas import (
    ActionCard,
    ActionOut,
    DraftPlanOut,
    EvidenceItem,
    RiskFlag,
    SourceType,
)
from careplan_ai.agents.care_plan.validation import link_risks_by_wording, validate_draft

CONFUSED = EvidenceItem(
    id="A1", source_type=SourceType.ASSESSMENT, risk_flagged=True, text="Yes",
    label="Is the Resident confused, disorientated, restless or highly irritable or agitated?")
INSIGHT = EvidenceItem(
    id="A2", source_type=SourceType.ASSESSMENT, risk_flagged=True, text="Yes",
    label="Does the Resident have reduced insight and/or judgment and / or are they uncooperative "
          "with staff?")
DIZZY = EvidenceItem(
    id="A3", source_type=SourceType.ASSESSMENT, risk_flagged=True, text="Dizziness on standing",
    label="Does the resident experience:")
REVIEW = EvidenceItem(
    id="R1", source_type=SourceType.REVIEW_NOTE, label="Review note",
    text="Falls review: Nell needs a staff member with her from 15:00 to 18:00.")
BY_ID = {e.id: e for e in (CONFUSED, INSIGHT, DIZZY, REVIEW)}
RISKS = [RiskFlag(evidence_id=i, reason="flagged") for i in ("A1", "A2", "A3")]


def _card(text: str, ids: list[str], risk_ids: list[str] | None = None) -> ActionCard:
    return ActionCard(id="", text=text, original_text=text, evidence_ids=list(ids),
                      risk_evidence_ids=list(risk_ids or []))


def test_an_action_that_deals_with_the_risk_in_words_is_linked_to_it() -> None:
    cards = [
        _card("Staff should stay with Nell from 15:00 to 18:00.", ["R1"]),
        _card("Staff should walk with Nell and offer music or singing when she is restless or "
              "agitated.", ["R1"]),
    ]
    links = link_risks_by_wording(cards, RISKS, BY_ID, "Nell")
    assert ("A1", 1) in links
    assert "A1" in cards[1].risk_evidence_ids
    assert "A1" in cards[1].evidence_ids  # View lists the question as a source
    # Nothing in the plan says how staff manage her reduced insight: still not covered.
    assert not any("A2" in c.risk_evidence_ids for c in cards)


def test_one_shared_word_is_not_enough() -> None:
    cards = [_card("Staff should offer Nell music when restless.", ["R1"])]
    assert link_risks_by_wording(cards, RISKS, BY_ID, "Nell") == []
    assert cards[0].risk_evidence_ids == []


def test_a_chosen_answer_counts_with_its_question() -> None:
    cards = [_card("Staff should help Nell sit before standing and report any dizziness.", ["R1"])]
    assert link_risks_by_wording(cards, RISKS, BY_ID, "Nell") == [("A3", 0)]


def test_the_action_sharing_the_most_words_gets_the_link() -> None:
    cards = [
        _card("Staff should check on Nell when she is restless or agitated at night.", ["R1"]),
        _card("Staff should calm Nell when she is confused, restless, irritable or agitated.",
              ["R1"]),
    ]
    link_risks_by_wording(cards, RISKS, BY_ID, "Nell")
    assert "A1" in cards[1].risk_evidence_ids
    assert "A1" not in cards[0].risk_evidence_ids


def test_a_risk_already_cited_is_left_alone() -> None:
    cards = [_card("Staff should reassure Nell when she is restless or agitated.", ["A1"], ["A1"])]
    assert link_risks_by_wording(cards, RISKS, BY_ID, "Nell") == []
    assert cards[0].risk_evidence_ids == ["A1"]


def test_the_residents_name_is_not_a_key_word() -> None:
    named = EvidenceItem(id="A4", source_type=SourceType.ASSESSMENT, risk_flagged=True,
                         text="Yes", label="Does Helen Strachan wander?")
    cards = [_card("Staff should walk with Helen Strachan to lunch.", ["R1"])]
    assert link_risks_by_wording(cards, [RiskFlag(evidence_id="A4", reason="x")],
                                 {"A4": named}, "Helen Strachan") == []


def test_through_the_draft_check_the_risk_counts_as_covered() -> None:
    draft = DraftPlanOut(
        actions=[ActionOut(text="Staff should walk with Nell and offer music when she is restless "
                                "or agitated.", evidence_ids=["R1"])],
        risks=[], note_classes=[], not_used=[])
    result = validate_draft([CONFUSED, INSIGHT, REVIEW], draft, resident_name="Nell",
                            pathway="Falls Assessment", with_outcomes=False)
    card = result.cards[0]
    assert "A1" in card.risk_evidence_ids
    problems = "\n".join(result.problems)
    assert "[A1] must be covered" not in problems
    assert "[A2] must be covered" in problems  # the real gap is still reported

"""Falls retest 26/11/2026 (Agnes, 81362): the weak lines of a real draft are sent back to the
agent - the same instruction for several risks, an assessment question's wording, and the answer
tacked on as a reason. Every action below is the exact text of that draft.
"""

import pytest

from careplan_ai.agents.care_plan.schemas import (
    ActionOut,
    DraftPlanOut,
    EvidenceItem,
    SourceType,
)
from careplan_ai.agents.care_plan.validation import (
    copies_question,
    repeated_instructions,
    strip_reason,
    validate_draft,
)


def _q(i: int, label: str, text: str = "Yes") -> EvidenceItem:
    return EvidenceItem(id=f"A{i}", source_type=SourceType.ASSESSMENT, label=label, text=text,
                        risk_flagged=True)


QUESTIONS = [
    _q(1, "Does the Resident often get out of bed overnight?"),
    _q(2, "Balance and mobility:"),
    _q(3, "Has the Resident had one or more falls in the last 12 months? Any recent falls?"),
    _q(4, "Do continence issues contribute to the Resident’s falls risk?"),
    _q(5, "Does the Resident wear ill-fitting shoes, high- heel shoes, or shoes without grip?"),
    _q(6, "Does the Resident have suitable clothing and footwear?", "No"),
    _q(7, "Is the Resident confused, disorientated, restless or highly irritable or agitated?"),
    _q(8, "Does the Resident have reduced insight and/or judgment and / or are they uncooperative "
          "with staff?"),
    _q(9, "Does the Resident have poor vision?"),
    _q(10, "Does the resident experience:", "Dizziness on standing"),
    _q(11, "Has the Resident lost weight recently unintentionally or reduced appetite?"),
    _q(12, "Is the Resident taking 4 or more medicines?"),
]

DRAFT = [
    "Staff should keep Agnes’s four‑wheeled walker within reach at all times and place a "
    "sensor mat beside her bed overnight, and prompt her to use her walker when she often stands "
    "up from her chair to go to the toilet and when she often gets out of bed overnight; offer "
    "balance and mobility support during these times, with awareness of her recent falls in the "
    "last 12 months.",
    "Staff should walk with Agnes to the bathroom overnight and do not leave her to walk on her "
    "own at night, responding promptly when she asks to go, as per the falls review and with "
    "awareness that continence issues contribute to her falls risk and that she does not like "
    "waiting for staff.",
    "Staff should check that Agnes does not wear ill‑fitting shoes, high‑heel shoes, or shoes "
    "without grip, and ensure suitable clothing and footwear before mobilising.",
    "Staff should provide close supervision during mobility and use clear, simple prompts when she "
    "is confused, disorientated, restless, highly irritable or agitated, or when she has reduced "
    "insight or judgment or is uncooperative.",
    "Staff should provide close supervision during mobility when Agnes has poor vision.",
    "Staff should help Agnes sit for a moment before standing and check her lying and standing "
    "blood pressure when dizziness on standing is reported.",
    "Staff should check how Agnes feels before standing and during walking and avoid rushing if "
    "she reports dizziness on standing or has lost weight recently unintentionally or reduced "
    "appetite.",
    "Staff should provide close supervision during mobility when she is taking 4 or more "
    "medicines.",
    "Staff to keep her room clear of clutter and her call bell within reach.",
]

# How a senior carer would write the same plan: none of these is sent back.
GOOD = [
    "Keep Agnes's four-wheeled walker beside her chair and bed, in the same place every time, and "
    "remind her to use it whenever she stands.",
    "Place a sensor mat beside her bed at night and go to her straight away when it alarms.",
    "Offer the toilet before bed, at night checks and after her morning medicines, without "
    "waiting for her to ask.",
    "Walk with Agnes to the bathroom at night; she must not walk alone at night.",
    "Help her sit on the edge of the bed for a moment before standing.",
    "Check her lying and standing blood pressure weekly and tell the nurse about a drop.",
    "Make sure she wears closed, well-fitting footwear with grip.",
    "Make sure she wears clean glasses and leave a night light on.",
    "Keep her room clear of clutter and her call bell within reach.",
]


def test_question_wording_is_found_in_the_draft() -> None:
    copied = {i: copies_question(t, QUESTIONS) for i, t in enumerate(DRAFT)}
    assert copied[2] is not None and copied[2].id == "A5"  # ill-fitting shoes, high-heel shoes
    assert copied[3] is not None and copied[3].id == "A7"  # confused, disorientated, restless
    assert copied[6] is not None and copied[6].id == "A11"  # lost weight ... reduced appetite


@pytest.mark.parametrize("text", GOOD)
def test_plain_actions_do_not_copy_a_question(text: str) -> None:
    assert copies_question(text, QUESTIONS) is None


def test_close_supervision_three_times_is_one_instruction_repeated() -> None:
    pairs = repeated_instructions(DRAFT, "Agnes")
    assert {(3, 4), (3, 7), (4, 7)} <= set(pairs)


def test_a_plain_plan_repeats_no_instruction() -> None:
    assert repeated_instructions(GOOD, "Agnes") == []


def test_the_name_alone_is_not_a_repeat() -> None:
    assert repeated_instructions([
        "Offer Agnes McPhail her walker every time she stands.",
        "Weigh Agnes McPhail weekly on the same scales.",
    ], "Agnes McPhail") == []


def test_the_answer_tacked_on_as_a_reason_is_taken_out() -> None:
    assert strip_reason(DRAFT[0]).endswith(
        "offer balance and mobility support during these times."
    )
    assert strip_reason(DRAFT[1]) == (
        "Staff should walk with Agnes to the bathroom overnight and do not leave her to walk on "
        "her own at night, responding promptly when she asks to go, as per the falls review."
    )


def test_being_aware_is_still_care() -> None:
    text = "Be aware of her dizziness and walk beside her."
    assert strip_reason(text) == text


def test_the_draft_is_sent_back_with_each_problem() -> None:
    draft = DraftPlanOut(actions=[
        ActionOut(text=t, evidence_ids=[q.id for q in QUESTIONS]) for t in DRAFT
    ], risks=[], note_classes=[], not_used=[])
    result = validate_draft(QUESTIONS, draft, resident_name="Agnes", pathway="Falls Assessment",
                            with_outcomes=False)
    problems = "\n".join(result.problems)
    assert "repeats the wording of the assessment question [A5]" in problems
    assert "repeats the wording of the assessment question [A11]" in problems
    assert "give the same instruction" in problems
    # Nothing is dropped: the care stays and the agent rewrites it.
    assert len(result.cards) == len(DRAFT)
    assert "with awareness" not in result.cards[0].text


# MUST retest 26/11/2026 (Agnes): exact actions of the draft.
MUST_DRAFT = [
    "Staff should weigh Agnes weekly on the same scales.",
    "Staff should record everything Agnes eats and drinks on the Food and Fluid charts.",
    "Staff should rescreen MUST monthly, or sooner if Agnes's weight drops.",
    "Staff should tell the nurse or GP, and follow up any dietitian referral, if Agnes's weight "
    "keeps falling or Agnes eats less than half of each meal for 3 days.",
    "Staff should offer Agnes a drink every 1-2 hours while awake to reach 1500 ml a day, and "
    "record each drink on the Fluid Chart.",
    "Staff should offer small portions and sweet foods that Agnes prefers.",
    "Staff should continue the fortified diet that has been started and follow up the dietitian "
    "referral that has been requested.",
    "Staff should follow up with the GP who was told about Agnes's poor appetite.",
]


def test_the_dietitian_follow_up_written_twice_is_found() -> None:
    assert repeated_instructions(MUST_DRAFT, "Agnes") == [(3, 6)]


def test_what_has_already_happened_is_taken_out_of_the_action() -> None:
    draft = DraftPlanOut(
        actions=[ActionOut(text=t, evidence_ids=["A1"]) for t in MUST_DRAFT[6:]],
        risks=[], note_classes=[], not_used=[],
    )
    evidence = [_q(1, "MUST Score:", "1")]
    result = validate_draft(evidence, draft, resident_name="Agnes", pathway="MUST",
                            with_outcomes=False)
    assert [c.text for c in result.cards] == [
        "Staff should continue the fortified diet and follow up the dietitian referral.",
        "Staff should follow up with the GP.",
    ]
    assert sum("already happened" in p for p in result.problems) == 2


@pytest.mark.parametrize("text", [
    "Give the laxative that was prescribed each morning.",
    "Check whether the referral has been made.",
])
def test_which_thing_is_meant_is_kept(text: str) -> None:
    draft = DraftPlanOut(actions=[ActionOut(text=text, evidence_ids=["A1"])],
                         risks=[], note_classes=[], not_used=[])
    result = validate_draft([_q(1, "Takes a laxative")], draft, resident_name="Agnes",
                            pathway="Bladder and Bowel", with_outcomes=False)
    assert result.cards[0].text == text


# Oral Health retest 26/11/2026 (Agnes): review note 857 changes the old plan. It reached the AI as
# a current note linked to Oral Health, was classed "update", and so was never required - the old
# line was not used, the new care was never written, and staff were not told.
OLD_ORAL = EvidenceItem(id="P1", source_type=SourceType.PREVIOUS_CARE_PLAN,
                        label="Previous care plan",
                        text="Agnes cleans her own teeth and upper denture with a prompt.")
ORAL_REVIEW = EvidenceItem(
    id="R1", source_type=SourceType.REVIEW_NOTE, label="Review note",
    text="Oral care review: Agnes can no longer clean her denture herself; staff clean it morning "
         "and night; dentist referral for loose denture.",
)
GUM = _q(1, "Gums/surrounding tissues:", "Dry, shiny, swollen, one ulcer or sore spot")


def _oral_draft(actions: list[ActionOut], kind: str) -> DraftPlanOut:
    from careplan_ai.agents.care_plan.schemas import NoteClassOut, NotUsedOut

    return DraftPlanOut(
        actions=actions, risks=[], note_classes=[NoteClassOut(evidence_id="R1", kind=kind)],
        not_used=[NotUsedOut(evidence_id="P1", superseded_by="R1", reason="Changed")],
    )


def test_a_note_that_changes_the_care_must_be_covered() -> None:
    gum_only = [ActionOut(text="Staff should check the sore spot on Agnes's gum every day.",
                          evidence_ids=["A1"])]
    result = validate_draft([GUM, OLD_ORAL, ORAL_REVIEW], _oral_draft(gum_only, "update"),
                            resident_name="Agnes", pathway="Oral Health Assessment",
                            with_outcomes=False)
    assert "[R1] must be covered by an action" in "\n".join(result.problems)
    assert any("can no longer clean her denture herself" in n for n in result.not_covered)
    assert [n.evidence_id for n in result.not_used] == ["P1"]


def test_the_update_written_as_an_action_is_accepted() -> None:
    actions = [
        ActionOut(text="Staff should check the sore spot on Agnes's gum every day.",
                  evidence_ids=["A1"]),
        ActionOut(text="Staff should clean Agnes's upper denture morning and night.",
                  evidence_ids=["R1"]),
    ]
    result = validate_draft([GUM, OLD_ORAL, ORAL_REVIEW], _oral_draft(actions, "update"),
                            resident_name="Agnes", pathway="Oral Health Assessment",
                            with_outcomes=False)
    assert result.not_covered == []
    assert not any("[R1] must be covered" in p for p in result.problems)


def test_a_status_note_is_still_not_required() -> None:
    gum_only = [ActionOut(text="Staff should check the sore spot on Agnes's gum every day.",
                          evidence_ids=["A1"])]
    result = validate_draft([GUM, OLD_ORAL, ORAL_REVIEW], _oral_draft(gum_only, "status"),
                            resident_name="Agnes", pathway="Oral Health Assessment",
                            with_outcomes=False)
    assert result.not_covered == []

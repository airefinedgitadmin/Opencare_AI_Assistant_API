"""OPENCARE-98 (Falls retest for Agnes McPhail, 19/11): flagged assessment answers are always
covered, actions carry no reasons and don't wait for the problem, one action per need, and
outcomes are distinct results - not records, not the same goal twice."""

from careplan_ai.agents.care_plan.evidence import build_evidence, observation_text, readable_date
from careplan_ai.agents.care_plan.prompts import format_evidence_line
from careplan_ai.agents.care_plan.schemas import (
    ActionOut,
    DraftPlanOut,
    EvidenceItem,
    SourceType,
)
from careplan_ai.agents.care_plan.validation import (
    _REACTIVE,
    _REASON,
    UNADDRESSED_RISK_WARNING,
    describes_recording,
    overlapping_actions,
    repeats_outcome,
    strip_reason,
    validate_draft,
)
from careplan_ai.api.schemas.agent import ObservationIn, StartSessionRequest


def _answer(i: int, question: str, answer: str, flagged: bool) -> EvidenceItem:
    return EvidenceItem(id=f"A{i}", source_type=SourceType.ASSESSMENT, label=question,
                        date="18/11/2026", text=answer, risk_flagged=flagged)


def _draft(*actions: tuple[str, list[str]]) -> DraftPlanOut:
    return DraftPlanOut(risks=[], note_classes=[], not_used=[],
                        actions=[ActionOut(text=t, evidence_ids=ids) for t, ids in actions])


# --- 1. the assessment, and every flagged answer, is used ---

def test_the_risk_flag_reaches_the_evidence_and_the_model_sees_it() -> None:
    request = StartSessionRequest.model_validate({
        "assessmentTemplateUserId": 66778,
        "pathway": "Falls Assessment",
        "assessment": {"completedDate": "18/11/2026", "answers": [
            {"questionId": "footwear", "question": "Footwear", "answer": "Unsuitable slippers",
             "riskFlagged": True},
            {"questionId": "aid", "question": "Walking aid", "answer": "Walker"},
        ]},
    })
    items = build_evidence(request)
    assert [i.risk_flagged for i in items[:2]] == [True, False]
    assert "[FLAGGED RISK]" in format_evidence_line(items[0])
    assert "[FLAGGED RISK]" not in format_evidence_line(items[1])


def test_a_flagged_answer_the_model_skipped_is_a_risk_and_shown_as_not_covered() -> None:
    evidence = [
        _answer(1, "Falls since September", "2 falls", True),
        _answer(2, "Medicines", "6 medicines incl. a diuretic", True),
        _answer(3, "Walking aid", "Walker", False),
    ]
    result = validate_draft(
        evidence,
        _draft(("Staff should keep Agnes's walker within reach at all times.", ["A1", "A3"])),
        resident_name="Agnes", pathway="Falls Assessment",
    )
    assert {r.evidence_id for r in result.risks} == {"A1", "A2"}
    uncovered = [c for c in result.cards if c.evidence_ids == ["A2"]]
    assert len(uncovered) == 1 and uncovered[0].warnings == [UNADDRESSED_RISK_WARNING]
    assert any("[A2]" in p for p in result.problems)


def test_an_unflagged_answer_is_not_forced_to_be_a_risk() -> None:
    evidence = [_answer(1, "Walking aid", "Walker", False)]
    result = validate_draft(evidence, _draft(), resident_name="Agnes")
    assert result.risks == []


# --- 3. no reasons in actions ---

def test_the_tickets_reason_wording_is_caught_and_taken_out() -> None:
    for text, expected in [
        ("Staff should walk with Agnes to the bathroom at night as she is no longer safe to "
         "walk on her own at night.", "Staff should walk with Agnes to the bathroom at night."),
        ("Staff should help Agnes find the bathroom at night, noting she has needed help twice "
         "during a night shift.", "Staff should help Agnes find the bathroom at night."),
        ("Staff should keep the walker beside her bed since she fell.",
         "Staff should keep the walker beside her bed."),
    ]:
        assert _REASON.search(text), text
        assert strip_reason(text) == expected


def test_time_and_instructions_are_not_reasons() -> None:
    for text in [
        "Staff should respond as soon as she presses her call bell.",
        "Staff should give her medicines as advised by the GP.",
        "Staff should place the sensor mat as requested.",
        "Staff should offer a drink as needed.",
    ]:
        assert not _REASON.search(text), text


# --- 2. prevent, don't wait ---

def test_an_action_that_waits_for_the_problem_is_caught() -> None:
    assert _REACTIVE.search("If Agnes says she feels light-headed when standing, staff should "
                            "offer support.")
    assert _REACTIVE.search("Staff should help her when she complains of dizziness.")


def test_routine_care_at_a_time_is_not_reactive() -> None:
    for text in [
        "When Agnes gets up at night, staff should walk with her to the toilet.",
        "Staff should ask Agnes to sit on the edge of the bed before standing.",
    ]:
        assert not _REACTIVE.search(text), text


def test_a_reactive_action_is_sent_back_to_the_agent() -> None:
    evidence = [_answer(1, "Dizziness on standing", "Yes", True)]
    result = validate_draft(
        evidence,
        _draft(("If Agnes says she feels light-headed when standing, staff should offer "
                "support.", ["A1"])),
        resident_name="Agnes",
    )
    assert any("waits for the problem" in p for p in result.problems)


# --- 4. one action per need ---

def test_two_actions_for_the_same_need_are_caught() -> None:
    texts = [
        "Staff should walk with Agnes to the bathroom at night.",
        "Staff should take Agnes to the bathroom at night and walk with her.",
        "Staff should keep the walker within reach at all times.",
    ]
    assert overlapping_actions(texts) == [(0, 1)]


def test_different_needs_are_not_merged() -> None:
    texts = [
        "Staff should keep Agnes's walker within reach at all times.",
        "Staff should place the sensor mat beside her bed at night.",
        "Staff should make sure she wears closed shoes or slippers with grip.",
        "Staff should ask the GP or pharmacist to review her 6 medicines.",
    ]
    assert overlapping_actions(texts) == []


def test_the_same_need_twice_is_sent_back_to_the_agent() -> None:
    evidence = [_answer(1, "Needs help to the toilet at night", "Yes", True)]
    result = validate_draft(
        evidence,
        _draft(("Staff should walk with Agnes to the bathroom at night.", ["A1"]),
               ("Staff should take Agnes to the bathroom at night and walk with her.", ["A1"])),
        resident_name="Agnes",
    )
    assert any("cover the same need" in p for p in result.problems)


# --- 5. distinct outcomes ---

def test_no_falls_is_the_goal_of_one_outcome_only() -> None:
    kept = ["Agnes has no further falls by the next review."]
    assert repeats_outcome("Agnes walks to the toilet at night with no falls.", kept)
    assert repeats_outcome("Agnes has no more falls when she uses her walker.", kept)


def test_the_tickets_expected_outcomes_are_all_distinct() -> None:
    expected = [
        "Agnes has no further falls by the next review.",
        "Agnes uses her walker every time she walks.",
        "Agnes gets to the toilet safely at night with staff support.",
        "Agnes has less light-headedness on standing.",
        "Agnes wears safe footwear every day.",
    ]
    for i, text in enumerate(expected):
        assert not repeats_outcome(text, expected[:i]), text


def test_an_outcome_about_the_record_is_not_a_result() -> None:
    assert describes_recording("Agnes is recorded as using her walker.")
    # Results the outcome rules themselves ask for are kept.
    assert not describes_recording("Cathy's pain is controlled and each PRN dose is recorded "
                                   "with its effect.")
    assert not describes_recording("Mr X's preferences for daily routine are recorded and "
                                   "followed in his first week.")


# --- 6. chart entries read as staff read them ---

def test_iso_dates_become_dd_mm_yyyy() -> None:
    assert readable_date("2026-11-09T21:40") == "09/11/2026 21:40"
    assert readable_date("2026-11-16 09:00:00") == "16/11/2026 09:00"
    assert readable_date("2026-11-09") == "09/11/2026"
    assert readable_date("2026-11-09T21:40:00.000Z") == "09/11/2026 21:40"
    for not_a_date in ["Bathroom", "104/62", "1500", "2026"]:
        assert readable_date(not_a_date) == not_a_date


def test_a_falls_diary_entry_leads_with_its_date_and_has_no_raw_iso_date() -> None:
    entry = ObservationIn.model_validate({
        "entryId": "fd1", "observationName": "Falls Diary", "recordedDate": "09/11/2026",
        "values": [
            {"field": "Date & Time of Fall", "value": "2026-11-09T21:40"},
            {"field": "Location of Fall", "value": "Bathroom"},
            {"field": "What Happened?",
             "value": "Slipped on the bathroom floor while walking without her walker."},
            {"field": "Injury", "value": ""},
        ],
    })
    text = observation_text(entry)
    assert text == ("09/11/2026 21:40 · Location of Fall: Bathroom · What Happened?: Slipped on "
                    "the bathroom floor while walking without her walker.")
    assert "2026-11-09" not in text and "T21" not in text and "Injury" not in text

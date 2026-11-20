"""OPENCARE-99 regression case: Agnes McPhail (81362), MUST, retest 19/11/2026.

Her records are fixed here, as the ticket lists them. The draft the ticket reports as wrong must
be caught on every point, and the ticket's expected draft must pass the same checks clean: the
MUST band sets the care standard (K items), the Food and Fluid charts both need an action, the
chart summary is dated by the entries' own Date & Time, chart quotes show only their own fields
with a running total, no action carries a reason or a status, outcomes are distinct and none
describes a record."""

from careplan_ai.agents.care_plan.evidence import build_evidence
from careplan_ai.agents.care_plan.schemas import ActionOut, DraftPlanOut, SourceType
from careplan_ai.agents.care_plan.standards import must_band
from careplan_ai.agents.care_plan.validation import (
    describes_recording,
    repeats_outcome,
    validate_draft,
)
from careplan_ai.api.schemas.agent import ObservationIn, StartSessionRequest


def _drink(time: str, drink: str, ml: str) -> dict:  # type: ignore[type-arg]
    return {"entryId": f"flc-{time}", "observationType": "OF_FLC", "observationName": "Fluid Chart",
            "recordedDate": "16/11/2026", "pathwayLinked": True, "role": "primary", "values": [
                {"field": "Date & Time", "value": f"2026-11-16T{time}:00"},
                {"field": "Record type", "value": "Intake"},
                {"field": "Drink", "value": drink},
                {"field": "Fluid target per day (ml)", "value": "1500"},
                {"field": "Current intake (ml)", "value": ml}]}


def _meal(time: str, meal: str, main: str, pudding: str | None) -> dict:  # type: ignore[type-arg]
    values = [{"field": "Date & Time", "value": f"2026-11-16T{time}"},
              {"field": "Meal", "value": meal},
              {"field": "Main course eaten", "value": main}]
    if pudding:
        values.append({"field": "Pudding eaten", "value": pudding})
    return {"entryId": f"fc-{time}", "observationType": "OF_FC", "observationName": "Food Chart",
            "recordedDate": "16/11/2026", "pathwayLinked": True, "role": "primary",
            "values": values}


AGNES = StartSessionRequest.model_validate({
    "assessmentTemplateUserId": 66801,
    "pathway": "MUST",
    "residentName": "Agnes",
    "assessment": {"completedDate": "18/11/2026", "answers": [
        {"questionId": "weight", "question": "Weight (kg)", "answer": "52.4"},
        {"questionId": "previousWeight", "question": "Previous weight (kg) - August",
         "answer": "55.8"},
        {"questionId": "loss", "question": "Weight loss (%)", "answer": "6.1"},
        {"questionId": "step2", "question": "Step 2 - weight loss score", "answer": "1"},
        {"questionId": "score", "question": "MUST score", "answer": "1"},
        {"questionId": "risk", "question": "Risk category", "answer": "Medium risk"},
        {"questionId": "meals", "question": "Eating", "answer": "Leaves most of her main meals"},
    ]},
    "notes": [
        {"noteId": 901, "noteType": "nursing", "createdDate": "18/11/2026", "pathwayLinked": True,
         "note": "GP told about poor appetite. Fortified diet started. Dietitian referral "
                 "requested."},
        {"noteId": 902, "noteType": "carer", "createdDate": "18/11/2026", "pathwayLinked": True,
         "note": "Prefers small portions and sweet foods. Likes rice pudding and milky tea with "
                 "one sugar."},
    ],
    "observations": [
        _drink("08:00", "Tea", "200"), _drink("11:00", "Water", "150"),
        _drink("15:00", "Tea", "200"), _drink("19:00", "Milk", "150"),
        _meal("12:30", "Lunch", "1/4", "All"), _meal("17:00", "Tea", "1/2", None),
    ],
})

EVIDENCE = build_evidence(AGNES)
BY_LABEL = {e.label: e.id for e in EVIDENCE if e.source_type == SourceType.ASSESSMENT}


def of(source_type: SourceType, label: str | None = None) -> list[str]:
    return [e.id for e in EVIDENCE
            if e.source_type == source_type and (label is None or e.label == label)]


def text_of(evidence_id: str) -> str:
    return next(e.text for e in EVIDENCE if e.id == evidence_id)


K = {e.text.split(" ")[0] + ":" + e.text.split(" ")[1]: e.id
     for e in EVIDENCE if e.source_type == SourceType.CARE_STANDARD}
SCORE = [BY_LABEL["MUST score"], BY_LABEL["Risk category"]]
FOOD = of(SourceType.OBSERVATION, "Food Chart") + of(SourceType.OBSERVATION_SUMMARY, "Food Chart")
FLUID = (of(SourceType.OBSERVATION, "Fluid Chart")
         + of(SourceType.OBSERVATION_SUMMARY, "Fluid Chart"))
NURSING = of(SourceType.NURSING_NOTE)
CARER = of(SourceType.CARER_NOTE)


def _check(*actions: tuple[str, list[str]]):
    draft = DraftPlanOut(risks=[], note_classes=[], not_used=[],
                         actions=[ActionOut(text=t, evidence_ids=i) for t, i in actions])
    return validate_draft(EVIDENCE, draft, resident_name="Agnes", pathway="MUST")


def _kinds(problems: list[str]) -> set[str]:
    found = set()
    for p in problems:
        if "explains why" in p:
            found.add("reason")
        if "how things stand" in p:
            found.add("status")
        if "cover the same need" in p:
            found.add("same need")
        if "must be covered by an action but no action cites it" in p:
            found.add("uncovered")
        if "raises a need for this pathway but no action cites it" in p:
            found.add("chart not used")
    return found


def test_the_must_band_comes_from_the_total_score_not_a_step_score() -> None:
    band = must_band(EVIDENCE, {})
    assert band is not None and band[:2] == ("Medium risk", 1)
    assert band[2] == BY_LABEL["MUST score"]


def test_medium_risk_gives_the_medium_care_standard_and_the_drink_step() -> None:
    standards = [e for e in EVIDENCE if e.source_type == SourceType.CARE_STANDARD]
    assert [e.text for e in standards] == [
        "Weigh Agnes weekly on the same scales.",
        "Record everything Agnes eats and drinks on the Food and Fluid charts.",
        "Rescreen MUST monthly, or sooner if Agnes's weight drops.",
        "Tell the nurse or GP, and follow up any dietitian referral, if Agnes's weight keeps "
        "falling or Agnes eats less than half of each meal for 3 days.",
        "Offer Agnes a drink every 1-2 hours while awake to reach 1500 ml a day, and record "
        "each drink on the Fluid Chart.",
    ]
    assert {e.label for e in standards} == {"MUST care standard - Medium risk",
                                            "Fluid care standard"}


def test_the_fluid_summary_is_dated_by_the_entries_and_totals_the_day_against_the_target() -> None:
    summary = text_of(of(SourceType.OBSERVATION_SUMMARY, "Fluid Chart")[0])
    assert summary.startswith("4 entries on 16/11/2026 · 700 ml against a 1500 ml target")
    assert "18/11/2026" not in summary
    # Said once here, not on every entry.
    assert "Fluid target per day (ml): 1500 every time" in summary
    assert "Record type: Intake every time" in summary


def test_chart_quotes_show_the_time_the_drink_and_the_running_total() -> None:
    entries = [text_of(i) for i in of(SourceType.OBSERVATION, "Fluid Chart")]
    assert entries[0] == ("16/11/2026 08:00 · Drink: Tea · "
                          "Current intake (ml): 200 (running total 200 ml)")
    assert entries[3] == ("16/11/2026 19:00 · Drink: Milk · "
                          "Current intake (ml): 150 (running total 700 ml)")
    assert not any("target" in e or "Record type" in e or "2026-11" in e for e in entries)


def test_the_food_chart_is_read_with_each_meal() -> None:
    entries = [text_of(i) for i in of(SourceType.OBSERVATION, "Food Chart")]
    assert entries == ["16/11/2026 12:30 · Meal: Lunch · Main course eaten: 1/4 · "
                       "Pudding eaten: All",
                       "16/11/2026 17:00 · Meal: Tea · Main course eaten: 1/2"]


def test_the_reported_draft_is_caught_on_every_point() -> None:
    result = _check(
        ("Staff should offer Agnes small portions with sweet options as she prefers these; a "
         "dietitian referral has been requested and she has a poor appetite reported to the GP.",
         [*NURSING, *CARER]),
        ("Staff should offer Agnes small portions and sweet foods such as rice pudding.", CARER),
        ("Staff should encourage Agnes to drink regularly to work towards 1500 ml.", FLUID),
    )
    kinds = _kinds(result.problems)
    assert {"reason", "status", "same need", "uncovered", "chart not used"} <= kinds
    # Neither the MUST standard nor the Food chart was acted on: both come back to staff.
    uncovered = {c.evidence_ids[0] for c in result.cards if c.verification == "source_text"}
    assert set(K.values()) - {K["Offer:Agnes"]} <= uncovered
    assert any(i in uncovered for i in FOOD)
    # Staff are offered the care without the reason or the status.
    first = result.cards[0].text
    assert first == "Staff should offer Agnes small portions with sweet options."


def test_the_expected_draft_passes_clean() -> None:
    result = _check(
        ("Staff should weigh Agnes weekly on the same scales.", [K["Weigh:Agnes"], *SCORE]),
        ("Staff should offer Agnes a fortified diet in small portions, including the sweet foods "
         "she likes such as rice pudding and milky tea with one sugar.", [*NURSING, *CARER, *FOOD]),
        ("Staff should offer Agnes a drink every 1-2 hours while awake to reach 1500 ml a day.",
         [K["Offer:Agnes"], *FLUID]),
        ("Staff should record everything Agnes eats and drinks on the Food and Fluid charts.",
         [K["Record:everything"], *FOOD, *FLUID]),
        ("Staff should rescreen MUST monthly, or sooner if Agnes's weight drops.",
         [K["Rescreen:MUST"], *SCORE]),
        ("Staff should follow up the dietitian referral, and tell the nurse or GP if Agnes's "
         "weight keeps falling or she eats less than half of each meal for 3 days.",
         [K["Tell:the"], *NURSING, *SCORE]),
    )
    kinds = _kinds(result.problems)
    assert kinds & {"reason", "status", "same need", "uncovered", "chart not used"} == set(), \
        result.problems
    assert all(c.verification != "source_text" for c in result.cards)


def test_the_expected_outcomes_are_distinct_and_none_describes_a_record() -> None:
    outcomes = [
        "Agnes's weight stays stable or goes up by the next review.",
        "Agnes eats at least half of each main meal.",
        "Agnes drinks at least 1500 ml a day.",
        "Agnes's MUST score is the same or lower at the monthly rescreen.",
    ]
    for i, text in enumerate(outcomes):
        assert not repeats_outcome(text, outcomes[:i]), text
        assert not describes_recording(text), text
    # The reported outcomes: the weight goal twice, and a record standing in for the result.
    assert repeats_outcome(
        "Agnes's weight stays stable or goes up by the next review as her appetite is poor.",
        outcomes[:1])
    assert describes_recording("Agnes's recorded daily fluid intake reaches 1500 ml.")


def test_no_band_recorded_means_no_must_standard() -> None:
    request = AGNES.model_copy(deep=True)
    assert request.assessment is not None
    request.assessment.answers = [a for a in request.assessment.answers
                                  if a.question_id not in {"score", "risk", "step2"}]
    standards = [e for e in build_evidence(request) if e.source_type == SourceType.CARE_STANDARD]
    assert [e.label for e in standards] == ["Fluid care standard"]


def test_high_and_low_scores_pick_their_own_band() -> None:
    for score, band in (("0", "Low risk"), ("3", "High risk")):
        request = AGNES.model_copy(deep=True)
        assert request.assessment is not None
        for answer in request.assessment.answers:
            if answer.question_id == "score":
                answer.answer = score
            if answer.question_id == "risk":
                answer.answer = band
        labels = {e.label for e in build_evidence(request)
                  if e.source_type == SourceType.CARE_STANDARD}
        assert f"MUST care standard - {band}" in labels


def test_the_must_chart_gives_the_band_when_the_assessment_has_none() -> None:
    request = AGNES.model_copy(deep=True)
    assert request.assessment is not None
    request.assessment.answers = [a for a in request.assessment.answers
                                  if a.question_id not in {"score", "risk", "step2"}]
    request.observations.append(ObservationIn.model_validate({
        "entryId": "must1", "observationType": "OF_MUST", "observationName": "MUST",
        "recordedDate": "18/11/2026", "pathwayLinked": True, "role": "primary",
        "values": [{"field": "Date & Time", "value": "2026-11-18T10:00"},
                   {"field": "Total MUST score", "value": "2"}]}))
    labels = {e.label for e in build_evidence(request)
              if e.source_type == SourceType.CARE_STANDARD}
    assert "MUST care standard - High risk" in labels


def test_other_pathways_get_no_must_standard() -> None:
    request = AGNES.model_copy(deep=True, update={"pathway": "Falls Assessment"})
    labels = {e.label for e in build_evidence(request) if e.source_type == SourceType.CARE_STANDARD}
    assert not any(label.startswith("MUST") for label in labels)


# --- code review fixes ---------------------------------------------------------------------


def _answer(evidence_id: str, question: str, answer: str):  # type: ignore[no-untyped-def]
    from careplan_ai.agents.care_plan.schemas import EvidenceItem
    return EvidenceItem(id=evidence_id, source_type=SourceType.ASSESSMENT, label=question,
                        text=answer)


def test_a_question_that_is_not_the_must_risk_never_sets_the_band() -> None:
    assert must_band([_answer("A1", "Activity level", "Low"),
                      _answer("A2", "Step 2 - weight loss score", "High (more than 10%)")],
                     {}) is None
    assert must_band([_answer("A1", "Activity level", "Low"),
                      _answer("A2", "MUST risk", "Medium")], {})[:2] == ("Medium risk", 1)


def test_care_after_a_semicolon_is_not_taken_for_a_status() -> None:
    from careplan_ai.agents.care_plan.validation import _STATUS_TAIL
    assert not _STATUS_TAIL.search(
        "Staff should give her medicines as prescribed; if a new medicine has been started, "
        "check with the nurse.")
    assert not _STATUS_TAIL.search(
        "Staff should support Agnes at meals; check whether the referral has been made.")
    assert _STATUS_TAIL.search(
        "Staff should offer small portions; a dietitian referral has been requested.")


def test_keeping_a_record_of_a_result_is_not_a_record_standing_in_for_it() -> None:
    assert not describes_recording("Each drink is recorded with the amount Agnes drinks.")
    assert describes_recording("Her recorded weight stays stable.")


def test_an_output_entry_is_not_counted_as_intake() -> None:
    def entry(time: str, kind: str, ml: str) -> dict:  # type: ignore[type-arg]
        return {"observationName": "Fluid Chart", "recordedDate": "16/11/2026",
                "pathwayLinked": True, "role": "primary", "values": [
                    {"field": "Date & Time", "value": f"2026-11-16T{time}"},
                    {"field": "Record type", "value": kind},
                    {"field": "Fluid target per day (ml)", "value": "1500"},
                    {"field": "Amount (ml)", "value": ml}]}
    request = StartSessionRequest.model_validate({"assessmentTemplateUserId": 1, "observations": [
        entry("08:00", "Intake", "200"), entry("10:00", "Output", "300")]})
    evidence = build_evidence(request)
    summary = next(e.text for e in evidence if e.source_type == SourceType.OBSERVATION_SUMMARY)
    assert "200 ml against a 1500 ml target" in summary
    entries = [e.text for e in evidence if e.source_type == SourceType.OBSERVATION]
    assert "running total" not in entries[1]

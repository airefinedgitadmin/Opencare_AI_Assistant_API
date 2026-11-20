"""OPENCARE-98 regression case: Agnes McPhail (81362), Falls Assessment, retest 19/11/2026.

Her records are fixed here, as the ticket lists them. The draft the ticket reports as wrong must
be caught on every point, and the ticket's expected draft must pass the same checks clean:
the assessment is a source, all 9 risks are covered, no action carries a reason, actions don't
wait for the problem or repeat a need, outcomes are distinct, chart quotes use dd/mm/yyyy."""

import re

from careplan_ai.agents.care_plan.evidence import build_evidence
from careplan_ai.agents.care_plan.schemas import ActionOut, DraftPlanOut, NotUsedOut, SourceType
from careplan_ai.agents.care_plan.validation import (
    UNADDRESSED_RISK_WARNING,
    repeats_outcome,
    validate_draft,
)
from careplan_ai.api.schemas.agent import StartSessionRequest

RISKS = {
    "falls": ("Falls since September", "2 falls since September"),
    "walker": ("Uses walking aid", "Walker not used when she needs the toilet urgently"),
    "meds": ("Medicines", "6 medicines incl. a diuretic"),
    "dizzy": ("Dizziness on standing", "Yes"),
    "cognition": ("Cognition", "Confused, poor insight"),
    "continence": ("Continence", "Needs help with continence"),
    "vision": ("Vision", "Poor vision"),
    "night": ("Up at night", "Yes"),
    "footwear": ("Footwear", "Unsuitable footwear"),
}

AGNES = StartSessionRequest.model_validate({
    "assessmentTemplateUserId": 66778,
    "pathway": "Falls Assessment",
    "residentName": "Agnes",
    "assessment": {"completedDate": "18/11/2026", "answers": [
        {"questionId": key, "question": q, "answer": a, "riskFlagged": True}
        for key, (q, a) in RISKS.items()
    ]},
    "previousCarePlans": [{"carePlanId": 501, "carePlanDate": "12/05/2026",
                           "action": "Agnes walks independently with her walker.", "outcome": ""}],
    "reviewNotes": [{"noteId": 801, "createdDate": "18/11/2026", "title": "Falls review",
                     "note": "Not safe to walk alone at night. Walker within reach. Sensor mat.",
                     "pathwayLinked": True}],
    "notes": [{"noteId": 802, "noteType": "carer", "createdDate": "18/11/2026",
               "note": "Stands up without her walker. Doesn't like waiting. Needed help to "
                       "find the bathroom twice at night.", "pathwayLinked": False}],
    "observations": [
        {"entryId": "fd1", "observationName": "Falls Diary", "recordedDate": "09/11/2026",
         "pathwayLinked": True, "role": "primary", "values": [
             {"field": "Date & Time of Fall", "value": "2026-11-09T21:40"},
             {"field": "Location of Fall", "value": "Bathroom"},
             {"field": "What Happened?", "value": "Slipped on the bathroom floor while walking "
                                                  "without her walker. Bruised left elbow."}]},
        {"entryId": "tpr1", "observationName": "TPR", "recordedDate": "16/11/2026",
         "pathwayLinked": True, "role": "primary", "values": [
             {"field": "Date & Time", "value": "2026-11-16T09:00"},
             {"field": "Lying BP", "value": "122/74"},
             {"field": "Standing BP", "value": "104/62"},
             {"field": "Comment", "value": "Light-headed on standing"}]},
    ],
})

EVIDENCE = build_evidence(AGNES)
ANSWER = {item.label: item.id for item in EVIDENCE if item.source_type == SourceType.ASSESSMENT}


def a(key: str) -> str:
    return ANSWER[RISKS[key][0]]


def ids(*prefixes: str) -> list[str]:
    return [e.id for e in EVIDENCE if any(e.id.startswith(p) for p in prefixes)]


# The old "walks independently with her walker" line, superseded by the review note.
NOT_USED = [NotUsedOut(evidence_id="P1", superseded_by="R2",
                       reason="Not safe to walk alone at night")]


def _check(*actions: tuple[str, list[str]]):
    draft = DraftPlanOut(risks=[], note_classes=[], not_used=NOT_USED,
                         actions=[ActionOut(text=t, evidence_ids=i) for t, i in actions])
    return validate_draft(EVIDENCE, draft, resident_name="Agnes", pathway="Falls Assessment")


def _kinds(problems: list[str]) -> set[str]:
    found = set()
    for p in problems:
        if "explains why" in p:
            found.add("reason")
        if "waits for the problem" in p:
            found.add("reactive")
        if "cover the same need" in p:
            found.add("same need")
        if "must be covered by an action but no action cites it" in p:
            found.add("uncovered")
    return found


def test_the_assessment_is_a_source_with_all_nine_risks_flagged() -> None:
    assessment = [e for e in EVIDENCE if e.source_type == SourceType.ASSESSMENT]
    assert len(assessment) == 9 and all(e.risk_flagged for e in assessment)
    assert all(e.date == "18/11/2026" for e in assessment)


def test_chart_quotes_use_dd_mm_yyyy_and_only_their_fields() -> None:
    charts = {e.label: e.text for e in EVIDENCE if e.source_type == SourceType.OBSERVATION}
    assert charts["Falls Diary"].startswith("09/11/2026 21:40 · Location of Fall: Bathroom")
    assert charts["TPR"].startswith("16/11/2026 09:00 · Lying BP: 122/74 · Standing BP: 104/62")
    assert not any(re.search(r"\d{4}-\d{2}-\d{2}", text) for text in charts.values())


def test_the_reported_draft_is_caught_on_every_point() -> None:
    walker_and_mat = [a("falls"), a("walker"), *ids("R")]
    result = _check(
        ("If Agnes says she feels light-headed when standing, staff should offer support.",
         [a("dizzy"), *ids("V")]),
        ("Staff should keep her walker within reach, use the sensor mat and prompt her, as she is "
         "no longer safe to walk on her own at night.", walker_and_mat),
        ("Staff should walk with Agnes to the bathroom at night, noting she has needed help twice "
         "during a night shift.", [a("night"), *ids("C")]),
        ("Staff should take Agnes to the bathroom at night and walk with her.",
         [a("night"), *ids("C")]),
    )
    assert {"reason", "reactive", "same need", "uncovered"} <= _kinds(result.problems)
    # Medicines, vision, footwear, continence and cognition had no action: shown as not covered.
    not_covered = {
        c.evidence_ids[0] for c in result.cards if UNADDRESSED_RISK_WARNING in c.warnings
    }
    assert {a("meds"), a("vision"), a("footwear"), a("continence"), a("cognition")} <= not_covered
    # The reasons were taken out of what staff are offered.
    assert not any(re.search(r"\b(as she|noting|because|since)\b", c.text) for c in result.cards)


def test_the_expected_draft_passes_clean_and_covers_all_nine_risks() -> None:
    result = _check(
        ("Staff should keep Agnes's walker within reach at all times, including beside her bed at "
         "night.", [a("falls"), a("walker"), *ids("R")]),
        ("Staff should place the sensor mat beside her bed at night and respond straight away when "
         "it alerts.", [a("falls"), a("cognition"), *ids("R")]),
        ("Staff should prompt and walk with Agnes to the toilet before bed, during the night and "
         "after her morning diuretic.", [a("continence"), a("night"), a("meds"), *ids("C")]),
        ("Staff should ask Agnes to sit on the edge of the bed before standing and to stand "
         "slowly.", [a("dizzy"), *ids("V")]),
        ("Staff should make sure she wears closed shoes or slippers with grip.", [a("footwear")]),
        ("Staff should keep a night light on and the route to the bathroom clear.",
         [a("vision"), a("night")]),
        ("Staff should ask the GP or pharmacist to review her 6 medicines.", [a("meds")]),
    )
    assert _kinds(result.problems) & {"reason", "reactive", "same need", "uncovered"} == set()
    assert not any(UNADDRESSED_RISK_WARNING in c.warnings for c in result.cards)
    assert {r.evidence_id for r in result.risks} == set(ANSWER.values())
    # "Agnes walks independently with her walker" is listed as not used, with the reason.
    assert [(n.evidence_id, n.superseded_by) for n in result.not_used] == [("P1", "R2")]
    assert not any("walks independently" in c.text for c in result.cards)


def test_the_expected_outcomes_are_distinct_and_no_falls_appears_once() -> None:
    outcomes = [
        "Agnes has no further falls by the next review.",
        "Agnes uses her walker every time she walks.",
        "Agnes gets to the toilet safely at night with staff support.",
        "Agnes has less light-headedness on standing.",
        "Agnes wears safe footwear every day.",
    ]
    for i, text in enumerate(outcomes):
        assert not repeats_outcome(text, outcomes[:i]), text
    # The reported outcomes: every one repeated "no falls ... by the next review".
    reported = ["Agnes has no falls by the next review.",
                "Agnes has no falls at night by the next review when using her walker."]
    assert repeats_outcome(reported[1], reported[:1])

"""OPENCARE-113: earlier care plan text the new score or diet level makes wrong stays out of
Actions.

The residents and wording are the 24/11/2026 retest's (OpenCare_Buddy_Test_Residents_18-11-2026).
The rules are code, so every run gives the same result (the ticket's "3 runs, 0 failures").
"""

import re

import pytest

from careplan_ai.agents.care_plan.schemas import (
    ActionOut,
    DraftPlanOut,
    EvidenceItem,
    NoteClassOut,
    NotUsedOut,
    SourceType,
)
from careplan_ai.agents.care_plan.superseded import describes_state
from careplan_ai.agents.care_plan.validation import validate_draft


def _p(i: int, text: str) -> EvidenceItem:
    return EvidenceItem(id=f"P{i}", source_type=SourceType.PREVIOUS_CARE_PLAN,
                        label="Previous care plan - action", date="20/03/2026", text=text)


def _draft(actions: list[tuple[str, list[str]]], **kw: object) -> DraftPlanOut:
    base: dict[str, object] = {"risks": [], "note_classes": [], "not_used": []}
    base.update(kw)
    return DraftPlanOut(actions=[ActionOut(text=t, evidence_ids=ids) for t, ids in actions],
                        **base)  # type: ignore[arg-type]


# --- Rab Lennox (81363) · Choking Risk Assessment ---

RAB = [
    _p(1, "Rab eats a normal diet and drinks thin fluids."),
    _p(2, "Staff to cut up his meat at mealtimes."),
    EvidenceItem(id="N1", source_type=SourceType.NURSING_NOTE, label="Nursing note",
                 date="18/11/2026",
                 text="SLT visit: Rab is now on IDDSI Level 5 minced and moist, with Level 2 "
                      "mildly thick fluids. Sit him upright at 90 degrees for meals."),
]


@pytest.mark.parametrize("run", [1, 2, 3])
def test_rab_choking_no_action_mentions_cutting_food_normal_diet_or_thin_fluids(run: int) -> None:
    draft = _draft(
        [("Staff should give assistance with food cutting or preparing prior to consuming at "
          "mealtimes, including cutting up his meat.", ["P2", "N1"]),
         ("Staff should give Rab IDDSI Level 5 minced and moist food and Level 2 mildly thick "
          "fluids.", ["N1"])],
        note_classes=[NoteClassOut(evidence_id="N1", kind="new_requirement")],
        # As on the retest: the model put only the first line under Not used.
        not_used=[NotUsedOut(evidence_id="P1", superseded_by="N1",
                             reason="SLT changed his diet.")],
    )

    result = validate_draft(RAB, draft, resident_name="Rab", pathway="Choking Risk Assessment")

    texts = " ".join(c.text for c in result.cards).casefold()
    assert not re.search(r"\bcut|normal diet|thin fluid", texts)
    # Both old lines are under Not used, each with a reason.
    reasons = {n.evidence_id: n.reason for n in result.not_used}
    assert set(reasons) == {"P1", "P2"}
    assert reasons["P2"] == "Now on IDDSI Level 5 minced and moist diet."
    assert all(reasons.values())
    assert any("repeats [P" in p for p in result.problems)


# --- Willie Baxter (81364) · MUST ---

WILLIE_MUST = [
    _p(1, "Weigh Willie monthly."),
    _p(2, "Offer snacks between meals."),
    EvidenceItem(id="A1", source_type=SourceType.ASSESSMENT, label="MUST Score", text="1"),
    EvidenceItem(id="K1", source_type=SourceType.CARE_STANDARD,
                 label="MUST care standard - Medium risk",
                 text="Weigh Willie weekly on the same scales."),
]


@pytest.mark.parametrize("run", [1, 2, 3])
def test_willie_must_only_weigh_weekly_appears(run: int) -> None:
    draft = _draft([
        ("Weigh Willie monthly.", ["P1"]),
        ("Weigh William weekly on the same scales.", ["K1", "A1"]),
        ("Offer Willie snacks between meals.", ["P2"]),
    ])

    result = validate_draft(WILLIE_MUST, draft, resident_name="Willie", pathway="MUST")

    texts = [c.text for c in result.cards]
    assert not any("monthly" in t.casefold() for t in texts)
    assert "Weigh William weekly on the same scales." in texts
    reasons = {n.evidence_id: n.reason for n in result.not_used}
    assert reasons == {"P1": "MUST is now Medium risk, weigh weekly."}


def test_a_low_must_score_keeps_weigh_monthly() -> None:
    evidence = [*WILLIE_MUST[:2],
                EvidenceItem(id="K1", source_type=SourceType.CARE_STANDARD,
                             label="MUST care standard - Low risk",
                             text="Weigh Willie monthly on the same scales.")]
    result = validate_draft(evidence, _draft([("Weigh Willie monthly.", ["P1", "K1"])]),
                            resident_name="Willie", pathway="MUST")

    assert result.not_used == []
    assert [c.text for c in result.cards][0] == "Weigh Willie monthly."


# --- Willie Baxter (81364) · Waterlow ---

WILLIE_WATERLOW = [
    _p(1, "Willie's skin is intact."),
    _p(2, "Encourage him to change position every 4 hours."),
    EvidenceItem(id="R1", source_type=SourceType.REVIEW_NOTE, label="Review note",
                 text="Reposition Willie every 2 hours."),
]


def test_willie_waterlow_skin_is_intact_is_never_an_action() -> None:
    draft = _draft(
        [("Willie's skin is intact.", ["P1"]),
         ("Staff should help Willie change position every 2 hours.", ["R1"])],
        note_classes=[NoteClassOut(evidence_id="R1", kind="new_requirement")],
        not_used=[NotUsedOut(evidence_id="P2", superseded_by="R1", reason="Now every 2 hours.")],
    )

    result = validate_draft(WILLIE_WATERLOW, draft, resident_name="Willie", pathway="Waterlow")

    assert [c.text for c in result.cards] == [
        "Staff should help Willie change position every 2 hours."]
    assert any("describes the resident" in p for p in result.problems)
    # Not pasted back either: listed for staff.
    assert result.not_actions == ["Willie's skin is intact."]


# --- Agnes McPhail (81362) · Falls Assessment ---

AGNES = [
    _p(1, "Agnes walks independently with her walker."),
    EvidenceItem(id="R1", source_type=SourceType.REVIEW_NOTE, label="Review note",
                 text="Falls review: Agnes is no longer safe to walk on her own at night."),
]


def test_agnes_falls_walks_independently_never_appears_in_an_action() -> None:
    draft = _draft(
        [("Staff should walk with Agnes to the bathroom at night; she walks independently with "
          "her walker during the day.", ["R1"]),
         ("Staff should walk with Agnes to the bathroom at night.", ["R1"])],
        note_classes=[NoteClassOut(evidence_id="R1", kind="new_requirement")],
        not_used=[NotUsedOut(evidence_id="P1", superseded_by="R1",
                             reason="No longer safe to walk alone at night.")],
    )

    result = validate_draft(AGNES, draft, resident_name="Agnes", pathway="Falls Assessment")

    assert not any("walks independently" in c.text for c in result.cards)
    assert [c.text for c in result.cards] == [
        "Staff should walk with Agnes to the bathroom at night."]


def test_any_line_under_not_used_never_appears_in_actions() -> None:
    draft = _draft(
        [("Staff should walk with Agnes to the bathroom at night.", ["R1"])],
        note_classes=[NoteClassOut(evidence_id="R1", kind="new_requirement")],
        not_used=[NotUsedOut(evidence_id="P1", superseded_by="R1", reason="Not safe alone.")],
    )
    result = validate_draft(AGNES, draft, resident_name="Agnes", pathway="Falls Assessment")

    not_used = [next(e.text for e in AGNES if e.id == n.evidence_id) for n in result.not_used]
    for line in not_used:
        assert all(line.casefold() not in c.text.casefold() for c in result.cards)


@pytest.mark.parametrize("text,state", [
    ("Willie's skin is intact.", True),
    ("Nell walks independently and safely around the unit.", True),
    ("She has upper and lower dentures.", True),
    ("Encourage her to join activities.", False),
    ("Weigh Willie monthly.", False),
    ("Agnes should be offered a drink every hour.", False),
    ("Staff to cut up his meat at mealtimes.", False),
])
def test_what_counts_as_describing_the_resident(text: str, state: bool) -> None:
    assert describes_state(text, "Willie Nell Agnes") is state

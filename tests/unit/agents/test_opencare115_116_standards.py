"""OPENCARE-115 (Waterlow and Choking care standards) and OPENCARE-116 (TPR & BP readings, the fluid
step only where it belongs) - built from the 24/11/2026 retest residents' records."""

from careplan_ai.agents.care_plan.evidence import build_evidence
from careplan_ai.agents.care_plan.schemas import (
    ActionOut,
    DraftPlanOut,
    EvidenceItem,
    SourceType,
)
from careplan_ai.agents.care_plan.standards import choking_not_recorded
from careplan_ai.agents.care_plan.validation import validate_draft
from careplan_ai.api.schemas.agent import StartSessionRequest


def _request(pathway: str, name: str, answers: list[tuple[str, str]],
             notes: list[str] = (), observations: list[dict] = ()) -> StartSessionRequest:  # type: ignore[assignment,type-arg]
    return StartSessionRequest.model_validate({
        "assessmentTemplateUserId": 1, "pathway": pathway, "residentName": name,
        "assessment": {"completedDate": "18/11/2026", "answers": [
            {"questionId": f"q{i}", "question": q, "answer": a}
            for i, (q, a) in enumerate(answers)]},
        "notes": [{"noteId": 900 + i, "noteType": "nursing", "createdDate": "18/11/2026",
                   "pathwayLinked": True, "note": n} for i, n in enumerate(notes)],
        "observations": list(observations),
    })


def _standards(evidence: list[EvidenceItem]) -> list[EvidenceItem]:
    return [e for e in evidence if e.source_type == SourceType.CARE_STANDARD]


def _tpr(day: str, role: str = "supporting", **fields: str) -> dict:  # type: ignore[type-arg]
    return {"entryId": f"tpr-{day}", "observationType": "OF_TPR", "observationName": "TPR & BP",
            "recordedDate": day, "pathwayLinked": role == "primary", "role": role,
            "values": [{"field": "Date & Time", "value": f"2026-11-{day[:2]}T10:00"}]
            + [{"field": k.replace("_", " "), "value": v} for k, v in fields.items()]}


def _drinks(day: str, ml: str) -> dict:  # type: ignore[type-arg]
    return {"entryId": f"flc-{day}", "observationType": "OF_FLC", "observationName": "Fluid Chart",
            "recordedDate": day, "pathwayLinked": True, "role": "primary", "values": [
                {"field": "Date & Time", "value": f"2026-11-{day[:2]}T10:00"},
                {"field": "Fluid target per day (ml)", "value": "1500"},
                {"field": "Current intake (ml)", "value": ml}]}


# --- OPENCARE-115: Waterlow ---

def test_rab_waterlow_18_gets_the_high_risk_standard() -> None:
    evidence = build_evidence(_request(
        "Waterlow", "Rab",
        [("Restricted mobility", "Yes"), ("Total Score", "18")],
        notes=["Rab prefers to sit in his wheelchair in the lounge for most of the afternoon."]))

    steps = _standards(evidence)
    assert {s.label for s in steps} == {"Waterlow care standard - High risk"}
    texts = " ".join(s.text for s in steps)
    assert "every 2 to 3 hours" in texts and "including when sitting in a chair" in texts
    assert "pressure-relieving mattress" in texts and "cushion" in texts
    assert "Check Rab's skin every day" in texts
    # Each step brings the score that calls for it, so Sources show it.
    score_id = next(e.id for e in evidence if e.label == "Total Score")
    assert all(s.backed_by == score_id for s in steps)


def test_waterlow_bands_and_below_ten_is_no_standard() -> None:
    def band(score: str) -> set[str]:
        return {s.label for s in _standards(build_evidence(
            _request("Waterlow", "Kate", [("Total Score", score)])))}

    assert band("9") == set()
    assert band("12") == {"Waterlow care standard - At risk"}
    assert band("20") == {"Waterlow care standard - Very high risk"}


# --- OPENCARE-115: Choking ---

RAB_SLT = ("SLT visit: Rab is now on IDDSI Level 5 minced and moist, Level 2 mildly thick "
           "fluids. Sit upright at 90 degrees. One-to-one supervision at meals.")


def test_rab_choking_gets_his_levels_and_every_safe_swallowing_step() -> None:
    evidence = build_evidence(_request("Choking Risk Assessment", "Rab",
                                       [("Coughs when eating", "Yes")], notes=[RAB_SLT]))

    texts = [s.text for s in _standards(evidence)]
    assert "Give Rab food only at IDDSI Level 5 minced and moist." in texts
    assert "Give Rab drinks only at Level 2 mildly thick fluids." in texts
    assert any("Stop if Rab coughs" in t and "tell the nurse" in t for t in texts)
    assert any("fully upright" in t for t in texts)
    # No plain drinks step in a Choking plan (OPENCARE-116).
    assert not any("1500" in t for t in texts)


def test_a_drink_action_without_the_thickened_level_is_dropped() -> None:
    evidence = build_evidence(_request("Choking Risk Assessment", "Rab",
                                       [("Coughs when eating", "Yes")], notes=[RAB_SLT]))
    note = next(e.id for e in evidence if e.is_note)
    draft = DraftPlanOut(risks=[], note_classes=[], not_used=[], actions=[
        ActionOut(text="Staff should offer Rab a drink every 1-2 hours while awake.",
                  evidence_ids=[note]),
        ActionOut(text="Staff should offer Rab Level 2 mildly thick drinks every 1-2 hours.",
                  evidence_ids=[note]),
    ])

    result = validate_draft(evidence, draft, resident_name="Rab",
                            pathway="Choking Risk Assessment")

    assert [c.text for c in result.cards if c.origin == "ai"] == [
        "Staff should offer Rab Level 2 mildly thick drinks every 1-2 hours."]
    assert any("does not say they are mildly thick" in p for p in result.problems)


def test_kate_choking_level_6_and_normal_fluids() -> None:
    evidence = build_evidence(_request(
        "Choking Risk Assessment", "Kate", [("Parkinson's disease", "Yes")],
        notes=["SLT: Kate is on IDDSI Level 6 soft and bite-sized, normal fluids."]))

    texts = [s.text for s in _standards(evidence)]
    assert "Give Kate food only at IDDSI Level 6 soft and bite-sized." in texts
    assert "Give Kate drinks only at normal fluids." in texts
    assert any("Supervise Kate" in t for t in texts)
    assert any("Stop if Kate coughs" in t for t in texts)


def test_nell_low_risk_choking_light_steps_and_level_not_recorded() -> None:
    request = _request("Choking Risk Assessment", "Nell",
                       [("Confusion", "Yes"), ("Eats rapidly", "Yes")],
                       observations=[_drinks("17/11/2026", "900")])
    evidence = build_evidence(request)

    steps = _standards(evidence)
    assert {s.label for s in steps} == {"Choking care standard - Low risk"}
    assert not any("1500" in s.text for s in steps)  # no general fluids step
    notice = choking_not_recorded(evidence, pathway=request.pathway, resident_name="Nell")
    assert notice is not None and notice.startswith("Diet level not recorded")


# --- OPENCARE-116: the fluid step only for MUST / Bladder and Bowel, below target ---

def test_fluid_step_only_for_must_and_bladder_and_bowel_below_target() -> None:
    def fluid(pathway: str, ml: str) -> bool:
        evidence = build_evidence(_request(pathway, "Kate", [("Comment", "x")],
                                           observations=[_drinks("17/11/2026", ml)]))
        return any(s.label == "Fluid care standard" for s in _standards(evidence))

    assert fluid("MUST", "900")
    assert fluid("Bladder and Bowel Assessment", "900")
    assert not fluid("MUST", "1600")  # target met
    assert not fluid("Choking Risk Assessment", "900")
    assert not fluid("Falls Assessment", "900")


# --- OPENCARE-116: TPR & BP ---

def test_kate_falls_standing_drop_gives_the_blood_pressure_step_with_its_reading() -> None:
    evidence = build_evidence(_request(
        "Falls Assessment", "Kate", [("Dizziness", "Yes")],
        observations=[_tpr("17/11/2026", Lying_BP="136/78", Standing_BP="108/64",
                           Comments="Dizzy on standing")]))

    steps = [s for s in _standards(evidence) if s.label.startswith("TPR")]
    assert [s.label for s in steps] == ["TPR & BP care standard - lower blood pressure on standing"]
    assert "sit for a moment before standing" in steps[0].text
    tpr = next(e for e in evidence if e.is_observation and e.label == "TPR & BP")
    assert steps[0].backed_by == tpr.id


def test_nell_falls_temperature_and_confusion_gives_the_report_step() -> None:
    evidence = build_evidence(_request(
        "Falls Assessment", "Nell", [("Confusion", "Yes")],
        observations=[_tpr("17/11/2026", Temperature="37.8", Pulse="102",
                           Comments="More confused, urine sample sent")]))

    steps = [s for s in _standards(evidence) if s.label.startswith("TPR")]
    assert [s.label for s in steps] == ["TPR & BP care standard - temperature 37.8 °C"]
    assert "new or sudden confusion" in steps[0].text and "tell the nurse" in steps[0].text


def test_rab_choking_temperature_gives_the_watch_chest_step() -> None:
    evidence = build_evidence(_request(
        "Choking Risk Assessment", "Rab", [("Coughs when eating", "Yes")], notes=[RAB_SLT],
        observations=[_tpr("16/11/2026", Temperature="37.9", SpO2="93", Comments="Chesty cough"),
                      _tpr("17/11/2026", Temperature="37.2", Comments="Cough easing")]))

    steps = [s for s in _standards(evidence) if s.label.startswith("TPR")]
    assert [s.label for s in steps] == ["TPR & BP care standard - temperature 37.9 °C"]
    assert "Watch Rab's temperature and chest" in steps[0].text


def test_a_normal_reading_or_another_pathway_adds_nothing() -> None:
    normal = _tpr("17/11/2026", Temperature="36.8", Lying_BP="130/80", Standing_BP="126/78")
    assert not [s for s in _standards(build_evidence(_request(
        "Falls Assessment", "Kate", [("x", "y")], observations=[normal])))
        if s.label.startswith("TPR")]
    hot = _tpr("17/11/2026", Temperature="38.2")
    assert not [s for s in _standards(build_evidence(_request(
        "MUST", "Kate", [("x", "y")], observations=[hot])))
        if s.label.startswith("TPR")]


def test_a_step_cites_its_reading_so_sources_show_it() -> None:
    evidence = build_evidence(_request(
        "Falls Assessment", "Kate", [("Dizziness", "Yes")],
        observations=[_tpr("17/11/2026", Lying_BP="136/78", Standing_BP="108/64")]))
    step = next(s for s in _standards(evidence) if s.label.startswith("TPR"))
    answer = next(e.id for e in evidence if e.source_type == SourceType.ASSESSMENT)
    draft = DraftPlanOut(risks=[], note_classes=[], not_used=[], actions=[
        ActionOut(text="Staff should help Kate sit for a moment before standing, and check her "
                       "lying and standing blood pressure.", evidence_ids=[step.id, answer])])

    result = validate_draft(evidence, draft, resident_name="Kate", pathway="Falls Assessment")

    assert step.backed_by in result.cards[0].evidence_ids

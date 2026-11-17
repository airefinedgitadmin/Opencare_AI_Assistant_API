"""End-to-end test findings on ockw-test (MUST): outcomes in the rules' own wording are kept,
and actions giving a reason or only doing paperwork are caught."""

from careplan_ai.agents.care_plan.prompts import (
    AGENT_SYSTEM_PROMPT,
    OUTCOME_SYSTEM_PROMPT,
    outcome_wording,
)
from careplan_ai.agents.care_plan.schemas import (
    ActionOut,
    DraftPlanOut,
    EvidenceItem,
    RiskFlagOut,
    SourceType,
)
from careplan_ai.agents.care_plan.validation import (
    _NOT_CARE,
    _REASON,
    borrowed_terms,
    paperwork_only,
    stems,
    strip_reason,
    validate_draft,
)

EVIDENCE = ["Weighed 46.8 kg (50.6 kg in July).", "Food and fluid chart started."]
EXAMPLES = ["Cathy's weight stays stable by her next review.",
            "Cathy uses her walking aid safely and her sensor mat is in place."]


def _borrowed(text: str, pathway: str) -> list[str]:
    source = stems(EVIDENCE + ["Elspeth", pathway, outcome_wording(pathway)])
    return borrowed_terms(text, source_stems=source, example_stems=stems(EXAMPLES))


def test_outcome_in_the_rules_own_wording_is_not_dropped_as_copied() -> None:
    assert _borrowed("Elspeth's weight stays stable or goes up by the next review.", "MUST") == []


def test_another_pathways_measures_are_still_caught_as_copied() -> None:
    # "walking aid" is a Falls measure: in a MUST outcome it can only have come from an example.
    assert "walking" in _borrowed("Elspeth has safe use of the walking aid.", "MUST")
    assert _borrowed("Elspeth has safe use of the walking aid.", "Falls Assessment") == []
    assert "sensor" in _borrowed("Elspeth has her sensor mat in place.", "Falls Assessment")


def test_outcome_wording_is_read_from_the_outcome_rules_for_the_pathway() -> None:
    must = outcome_wording("MUST")
    assert "weight trend" in must and "walking aid" not in must
    assert "IDDSI textures" in outcome_wording("Choking Risk Assessment")
    assert "evacuation plan" in outcome_wording("Bedroom Fire Risk Assessment (BFRA)")
    assert "evacuation plan" in outcome_wording("BFRA")
    other = outcome_wording("A pathway added later")
    assert "completed referral" in other and "weight trend" not in other
    assert "completed referral" in outcome_wording(None)
    for wording in (must, other):
        assert wording.startswith(OUTCOME_SYSTEM_PROMPT.split("WHAT A GOOD")[0])


def test_paperwork_from_an_earlier_plan_is_caught() -> None:
    for text in [
        "Staff should document that Elspeth was weighed 46.8 kg and file this with the chart.",
        "Staff should note that a dietitian referral was made.",
        "Staff should note that she has been referred to the dietitian.",
        "Staff should keep the chart and file it with her notes.",
    ]:
        assert _NOT_CARE.search(text), text
    for text in [
        "Staff should record every drink on the Fluid chart.",
        "Staff should record Elspeth's food and fluid intake on the Food and Fluid charts.",
        "Staff should weigh Elspeth weekly and record her weight on the MUST chart.",
        "Staff should offer Elspeth the IDDSI Level 6 soft and bite-sized diet.",
        # Facts about the resident are not paperwork (from Elspeth's Oral Health plan on test).
        "Staff should prompt Elspeth to put in her upper denture before eating, and note that "
        "she has an upper denture and natural lower teeth.",
        "Staff should note that Cathy has a hearing aid in her left ear and check it is in.",
        # A reason is not paperwork: it is handled by _REASON, which keeps the action.
        "Staff should offer the IDDSI Level 6 diet as advised because she was coughing.",
    ]:
        assert not _NOT_CARE.search(text), text


def test_reason_clauses_are_found_but_times_are_not() -> None:
    for text in ["Staff should offer the IDDSI Level 6 diet as advised because she was coughing.",
                 "Staff should offer thickened fluids due to her swallowing difficulty.",
                 "Staff should sit Elspeth upright as a result of her coughing.",
                 "Staff should offer small meals owing to her poor appetite."]:
        assert _REASON.search(text), text
    time = "Staff should give Elspeth her medication when it is due to be given."
    assert not _REASON.search(time)


def _previous(pid: str, text: str) -> EvidenceItem:
    return EvidenceItem(id=pid, source_type=SourceType.PREVIOUS_CARE_PLAN,
                        label="Previous care plan", text=text)


def _validate(evidence: list[EvidenceItem], actions: list[ActionOut]):  # type: ignore[no-untyped-def]
    draft = DraftPlanOut(risks=[], note_classes=[], not_used=[], actions=actions)
    return validate_draft(evidence, draft, resident_name="Elspeth", with_outcomes=False)


def test_an_action_giving_a_reason_keeps_the_care_without_the_reason() -> None:
    evidence = [_previous("P1", "Offer the IDDSI Level 6 diet; coughing with thin drinks.")]
    text = ("Staff should offer the IDDSI Level 6 diet as advised because she was coughing "
            "with thin drinks, and ensure she sits fully upright for meals.")

    result = _validate(evidence, [ActionOut(text=text, evidence_ids=["P1"])])

    assert [c.text for c in result.cards] == [  # the care is not lost; the reason is
        "Staff should offer the IDDSI Level 6 diet as advised, and ensure she sits fully "
        "upright for meals."
    ]
    assert result.cards[0].original_text == result.cards[0].text
    assert any("explains why" in p for p in result.problems)


def test_strip_reason_keeps_the_sentence_readable() -> None:
    assert strip_reason("Staff should offer thickened fluids due to her swallowing difficulty.") \
        == "Staff should offer thickened fluids."
    assert strip_reason("Staff should, because she is at risk, sit Cathy upright for meals.") \
        == "Staff should sit Cathy upright for meals."
    assert strip_reason("Because she coughs with thin drinks, staff should offer Level 2 fluids.") \
        == "Staff should offer Level 2 fluids."
    time = "Staff should give Cathy her medication when it is due to be given."
    assert strip_reason(time) == time
    # A dot or comma inside a number does not end the reason.
    assert strip_reason("Staff should offer fortified snacks because her weight fell from 50.6 kg "
                        "to 46.8 kg, and weigh her weekly.") \
        == "Staff should offer fortified snacks, and weigh her weekly."
    assert strip_reason("Staff should encourage drinks due to intake of 1,200 ml of a 1,500 ml "
                        "target.") == "Staff should encourage drinks."


def test_only_lines_that_are_all_paperwork_count_as_paperwork() -> None:
    assert paperwork_only("Staff should document that Elspeth was weighed 46.8 kg (50.6 kg in "
                          "July) and file this with the food and fluid chart.")
    assert paperwork_only("Staff should note that a dietitian referral has been sent.")
    for care in ["Staff should help Cathy with meals and document that she was weighed.",
                 "Staff should note that a referral has been sent. Staff should offer snacks.",
                 "Staff should note that she has an upper denture and natural lower teeth.",
                 "Staff should record every drink on the Fluid chart."]:
        assert not paperwork_only(care), care


def test_paperwork_from_earlier_plans_is_not_carried_and_staff_are_told_once() -> None:
    paperwork = ("Staff should document that Elspeth was weighed 46.8 kg (50.6 kg in July) and "
                 "file this with the food and fluid chart that has been started.")
    evidence = [_previous("P1", "Offer fortified snacks."), _previous("P4", paperwork),
                _previous("P8", paperwork)]

    result = _validate(evidence, [ActionOut(text="Staff should offer Elspeth fortified snacks.",
                                            evidence_ids=["P1"])])

    assert [c.text for c in result.cards] == ["Staff should offer Elspeth fortified snacks."]
    assert result.not_carried == [paperwork]
    assert not any("must be covered" in p for p in result.problems)


def test_the_same_line_from_two_earlier_plans_is_shown_once() -> None:
    line = "Staff should offer Elspeth a small plate at meals."
    evidence = [_previous("P1", "Offer fortified snacks."), _previous("P3", line),
                _previous("P7", line + " ")]

    result = _validate(evidence, [ActionOut(text="Staff should offer Elspeth fortified snacks.",
                                            evidence_ids=["P1"])])

    backfilled = [c for c in result.cards if c.text.strip() == line]
    assert len(backfilled) == 1
    assert backfilled[0].evidence_ids == ["P3", "P7"]


def test_different_answers_with_the_same_words_stay_separate() -> None:
    evidence = [
        EvidenceItem(id="A1", source_type=SourceType.ASSESSMENT, label="Dentures", text="Yes"),
        EvidenceItem(id="A2", source_type=SourceType.ASSESSMENT, label="Glasses", text="Yes"),
        _previous("P1", "Offer fortified snacks."),
    ]
    draft = DraftPlanOut(
        risks=[RiskFlagOut(evidence_id="A1", reason="dentures"),
               RiskFlagOut(evidence_id="A2", reason="glasses")],
        note_classes=[], not_used=[],
        actions=[ActionOut(text="Staff should offer Elspeth fortified snacks.",
                           evidence_ids=["P1"])],
    )

    result = validate_draft(evidence, draft, resident_name="Elspeth", with_outcomes=False)

    assert [(c.text, c.evidence_ids) for c in result.cards[1:]] == [
        ("Dentures: Yes", ["A1"]), ("Glasses: Yes", ["A2"])]


def test_the_draft_rules_say_earlier_plans_are_not_wording_to_copy() -> None:
    assert "not wording to copy" in AGENT_SYSTEM_PROMPT
    assert "because she was coughing" in AGENT_SYSTEM_PROMPT

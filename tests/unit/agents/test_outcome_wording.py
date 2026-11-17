"""End-to-end test findings on ockw-test (MUST): outcomes in the rules' own wording are kept,
and actions giving a reason or only doing paperwork are caught."""

from careplan_ai.agents.care_plan.prompts import (
    AGENT_SYSTEM_PROMPT,
    OUTCOME_SYSTEM_PROMPT,
    outcome_wording,
)
from careplan_ai.agents.care_plan.schemas import ActionOut, DraftPlanOut, EvidenceItem, SourceType
from careplan_ai.agents.care_plan.validation import (
    _NOT_CARE,
    _REASON,
    borrowed_terms,
    stems,
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


def test_an_action_giving_a_reason_is_kept_and_sent_back_to_take_the_reason_out() -> None:
    evidence = [EvidenceItem(id="P1", source_type=SourceType.PREVIOUS_CARE_PLAN,
                             label="Previous care plan",
                             text="Offer the IDDSI Level 6 diet; coughing with thin drinks.")]
    text = ("Staff should offer the IDDSI Level 6 diet as advised because she was coughing "
            "with thin drinks.")
    draft = DraftPlanOut(risks=[], note_classes=[], not_used=[],
                         actions=[ActionOut(text=text, evidence_ids=["P1"])])

    result = validate_draft(evidence, draft, resident_name="Elspeth", with_outcomes=False)

    assert [c.text for c in result.cards] == [text]  # the care is not lost
    assert any("explains why" in p for p in result.problems)


def test_the_draft_rules_say_earlier_plans_are_not_wording_to_copy() -> None:
    assert "not wording to copy" in AGENT_SYSTEM_PROMPT
    assert "because she was coughing" in AGENT_SYSTEM_PROMPT

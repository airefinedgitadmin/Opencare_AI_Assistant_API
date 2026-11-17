"""OPENCARE-80: placeholder text is never evidence; drafts stay in the pathway; outcome rules."""

from careplan_ai.agents.care_plan.evidence import (
    build_evidence,
    is_placeholder,
    to_sentences,
)
from careplan_ai.agents.care_plan.prompts import (
    AGENT_SYSTEM_PROMPT,
    OUTCOME_SYSTEM_PROMPT,
    VERIFY_SYSTEM_PROMPT,
    build_outcome_messages,
)
from careplan_ai.agents.care_plan.schemas import EvidenceItem, SourceType
from careplan_ai.api.schemas.agent import StartSessionRequest


def test_placeholder_text_is_recognised() -> None:
    junk_texts = ["test", "Test.", "TEST", "testing", "test 2", "tbc", "TBA", "xxx", "asdf"]
    for junk in junk_texts:
        assert is_placeholder(junk), junk
    for real in ["Test results awaited from GP.", "Weighed 61.2 kg.", "Eats puddings.", "No pain.",
                 "ABC", "ABC chart started.", "Lubi herbatę.", "喜欢喝茶",
                 "✓", "+", "-"]:  # a tick or +/- can be a real answer
        assert not is_placeholder(real), real


def test_placeholder_sentences_are_dropped_from_evidence_only() -> None:
    assert to_sentences("<p>test</p>", drop_placeholders=True) == []
    assert to_sentences("Test. Offer a fortified diet.", drop_placeholders=True) == [
        "Offer a fortified diet."]
    assert to_sentences("<ul><li>tbc</li><li>Weigh weekly</li></ul>", drop_placeholders=True) == [
        "Weigh weekly"]
    # The style library (and anything else) still gets the text unchanged.
    assert to_sentences("<p>test</p>") == ["test"]


def test_a_previous_care_plan_saying_test_gives_no_evidence() -> None:
    request = StartSessionRequest.model_validate(
        {
            "assessmentTemplateUserId": 1,
            "previousCarePlans": [
                {"carePlanId": 9, "carePlanDate": "06/08/2026", "action": "test",
                 "outcome": "Test."}
            ],
            "assessment": {"answers": [
                {"questionId": "q1", "question": "Weight", "answer": "tbc"},
                {"questionId": "q2", "question": "MUST score", "answer": "1 - Medium risk"},
            ]},
            "notes": [
                {"noteId": 5, "noteType": "nursing", "createdDate": "01/11/2026",
                 "note": "Weight down from 66.0 kg, fortified diet started."},
                {"noteId": 6, "noteType": "nursing", "title": "Test", "note": "test"},
            ],
        }
    )

    dropped: set[str] = set()
    evidence = build_evidence(request, dropped)

    assert all(e.source_type != SourceType.PREVIOUS_CARE_PLAN for e in evidence)
    assert all("test" not in e.text.casefold() for e in evidence)  # a note's title too
    answers = [e.text for e in evidence if e.source_type == SourceType.ASSESSMENT]
    assert answers == ["1 - Medium risk"]
    # Counted per record: the plan (action and outcome), the "tbc" answer, the "Test" note
    assert len(dropped) == 3


def test_draft_rules_keep_the_pathway_and_use_its_charts_and_score() -> None:
    assert "STAY IN THE PATHWAY" in AGENT_SYSTEM_PROMPT
    assert "check \\" not in AGENT_SYSTEM_PROMPT  # line continuations resolved
    assert "MUST score 1" in AGENT_SYSTEM_PROMPT
    assert "Fluid chart showing intake below its" in AGENT_SYSTEM_PROMPT
    assert "ONE thing staff do" in AGENT_SYSTEM_PROMPT


def test_outcome_rules_cover_every_pathway() -> None:
    assert "RESULT for the resident, not a staff task" in OUTCOME_SYSTEM_PROMPT
    assert "checkable at the next review" in OUTCOME_SYSTEM_PROMPT
    assert "never invent a" in OUTCOME_SYSTEM_PROMPT
    for pathway in ["MUST", "Diet and Fluid Notification", "Choking Risk", "Falls", "Oral Health",
                    "Waterlow", "Skin Integrity", "Wound Care", "Bladder and Bowel", "Stoma Care",
                    "Stress and Distress", "Assessment of Dependency", "PRN Medication",
                    "End of Life Wishes", "Bedroom Fire Risk Assessment (BFRA)",
                    "Admission Care Plan", "Any other pathway"]:
        assert f"\n- {pathway}: " in OUTCOME_SYSTEM_PROMPT, pathway
    # No made-up target or other resident's name in the examples the model sees.
    assert "1500" not in OUTCOME_SYSTEM_PROMPT and "Yoda" not in OUTCOME_SYSTEM_PROMPT
    assert "Yoda" not in AGENT_SYSTEM_PROMPT
    assert "a target no quote gives is" in VERIFY_SYSTEM_PROMPT


def test_outcome_request_names_the_pathway() -> None:
    quote = EvidenceItem(id="T1", source_type=SourceType.OBSERVATION_SUMMARY, label="Fluid Chart",
                         text="5 entries; Fluid target per day (ml): 1500 every time",
                         date="02/11/2026")
    messages = build_outcome_messages(
        resident_name="Yoda",
        actions=[("S1", "Staff should offer Yoda drinks every 1-2 hours.", [quote])],
        pathway="MUST",
    )

    human = messages[1].content
    assert "Resident: Yoda\nPathway: MUST" in human
    assert "[S1] ACTION:" in human
    unnamed = build_outcome_messages(resident_name="Yoda", actions=[])
    assert "Pathway: Not specified" in unnamed[1].content

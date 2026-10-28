from careplan_ai.agents.care_plan.evidence import build_evidence, to_sentences
from careplan_ai.agents.care_plan.schemas import SourceType
from careplan_ai.api.schemas.agent import StartSessionRequest


def test_to_sentences_splits_quill_html_into_items() -> None:
    html = (
        "<p>Cathy is independent with her oral care needs. She needs prompting&nbsp;from staff.</p>"
        "<ul><li>Uses upper dentures</li><li>Natural bottom teeth</li></ul>"
    )

    assert to_sentences(html) == [
        "Cathy is independent with her oral care needs.",
        "She needs prompting from staff.",
        "Uses upper dentures",
        "Natural bottom teeth",
    ]


def test_to_sentences_handles_plain_text_and_empty() -> None:
    assert to_sentences("One. Two.") == ["One.", "Two."]
    assert to_sentences(None) == []
    assert to_sentences("   ") == []


def test_build_evidence_numbers_each_source_and_skips_unanswered() -> None:
    request = StartSessionRequest.model_validate(
        {
            "assessmentTemplateUserId": 1,
            "assessment": {
                "completedDate": "14/10/2026",
                "answers": [
                    {"questionId": "q1", "question": "Dentures", "answer": "Yes"},
                    {"questionId": "q2", "question": "Dental pain", "answer": None},
                    {"questionId": "q3", "question": "Oral cleanliness", "answer": "Some plaque"},
                ],
            },
            "previousCarePlans": [
                {
                    "carePlanId": 9,
                    "carePlanDate": "20/05/2026",
                    "action": "<p>Needs prompting. Has upper dentures.</p>",
                    "outcome": "No oral discomfort.",
                }
            ],
            "reviewNotes": [
                {"noteId": 5, "createdDate": "08/10/2026", "note": "Still uses dentures."}
            ],
        }
    )

    evidence = build_evidence(request)

    assert [e.id for e in evidence] == ["A1", "A2", "P1", "P2", "O1", "R1"]
    assert evidence[1].label == "Oral cleanliness"  # unanswered q2 skipped
    assert evidence[2].source_type == SourceType.PREVIOUS_CARE_PLAN
    assert evidence[2].source_ref == "9"
    assert evidence[5].text == "Still uses dentures."


def test_build_evidence_never_caps_or_clips() -> None:
    # Deck slide 20: no information removed or skipped.
    request = StartSessionRequest.model_validate(
        {
            "assessmentTemplateUserId": 1,
            "assessment": {
                "answers": [{"question": f"Q{i}", "answer": "x" * 5000} for i in range(300)]
            },
        }
    )

    evidence = build_evidence(request)

    assert len(evidence) == 300
    assert all(len(e.text) == 5000 for e in evidence)


def test_build_evidence_numbers_every_note_type() -> None:
    request = StartSessionRequest.model_validate(
        {
            "assessmentTemplateUserId": 1,
            "reviewNotes": [
                {"noteId": 1, "note": "Needs help now.", "pathwayLinked": False},
            ],
            "notes": [
                {"noteId": 2, "noteType": "carer", "createdDate": "10/10/2026",
                 "note": "Refused to brush teeth. Ate well."},
                {"noteId": 3, "noteType": "nursing", "note": "Gums sore.", "pathwayLinked": True},
                {"noteId": 4, "noteType": "professional", "professionalType": "Dentist",
                 "createdDate": "01/10/2026", "visitDate": "30/09/2026",
                 "title": "Dental check", "note": "Soft brush advised."},
                {"noteId": 5, "noteType": "relative", "relation": "Daughter",
                 "note": "Mum prefers mint toothpaste."},
            ],
        }
    )

    evidence = build_evidence(request)
    by_id = {e.id: e for e in evidence}

    assert [e.id for e in evidence] == ["R1", "C1", "C2", "N1", "G1", "G2", "F1"]
    assert by_id["R1"].linked_to_pathway is False
    assert by_id["C1"].source_type == SourceType.CARER_NOTE
    assert by_id["C1"].linked_to_pathway is False  # other notes default to resident-level
    assert by_id["N1"].linked_to_pathway is True
    # The title is kept as its own quotable sentence; the visit date wins.
    assert (by_id["G1"].text, by_id["G1"].label, by_id["G1"].date) == (
        "Dental check", "Dentist note", "30/09/2026"
    )
    assert by_id["F1"].label == "Relative (Daughter)"
    assert all(e.is_note for e in evidence)

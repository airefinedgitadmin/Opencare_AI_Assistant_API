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


def test_build_evidence_makes_one_item_per_observation_entry() -> None:
    request = StartSessionRequest.model_validate(
        {
            "assessmentTemplateUserId": 1,
            "observations": [
                {"entryId": "m1", "observationType": "OF_FLC", "observationName": "Fluid Chart",
                 "recordedDate": "12/10/2026", "pathwayLinked": True,
                 "values": [{"field": "Drink", "value": "Tea"},
                            {"field": "Amount (ml)", "value": " 200 "},
                            {"field": "Comments", "value": ""}]},
                {"entryId": "m2", "observationName": "Fluid Chart",
                 "values": [{"field": "Drink", "value": None}]},
                {"entryId": "m3", "observationName": "Falls Diary", "historical": True,
                 "values": [{"field": "Location", "value": "Lounge"}]},
            ],
        }
    )

    evidence = build_evidence(request)

    # m2 recorded nothing, so it is skipped; blank fields are left out.
    assert [e.id for e in evidence] == ["V1", "V2"]
    v1, v2 = evidence
    assert (v1.text, v1.label, v1.date, v1.source_ref) == (
        "Drink: Tea; Amount (ml): 200", "Fluid Chart", "12/10/2026", "m1"
    )
    assert v1.source_type == SourceType.OBSERVATION
    assert v1.is_observation and not v1.is_note
    assert v1.linked_to_pathway is True and v1.historical is False
    assert v2.linked_to_pathway is False and v2.historical is True


def test_each_reading_keeps_its_role_from_the_pathway_rules() -> None:
    def reading(**extra: object) -> dict:  # type: ignore[type-arg]
        return {"observationName": "Toileting Chart",
                "values": [{"field": "Output", "value": "Wet"}], **extra}

    request = StartSessionRequest.model_validate({
        "assessmentTemplateUserId": 1,
        "observations": [reading(role="primary", pathwayLinked=True),
                         reading(role="supporting"),
                         reading(pathwayLinked=True),  # older backend: no role sent
                         reading()],
    })

    readings = [e for e in build_evidence(request) if e.is_observation]

    assert [e.supporting for e in readings] == [False, True, False, True]


def _fluid(day: str, amount: str, target: str = "3000", historical: bool = False) -> dict:  # type: ignore[type-arg]
    return {"observationName": "Fluid Chart", "recordedDate": day, "pathwayLinked": True,
            "historical": historical,
            "values": [{"field": "Drink", "value": "Tea"},
                       {"field": "Amount (ml)", "value": amount},
                       {"field": "Target (ml)", "value": target}]}


def test_each_chart_gets_one_exact_summary_line_computed_in_code() -> None:
    request = StartSessionRequest.model_validate(
        {
            "assessmentTemplateUserId": 1,
            "observations": [
                _fluid("21/10/2026", "200"), _fluid("21/10/2026", "60"),
                _fluid("23/10/2026", "150"), _fluid("22/10/2026", "90"),
                _fluid("01/05/2026", "100", historical=True),  # one old entry: no summary
                {"observationName": "Falls Diary", "recordedDate": "20/10/2026",
                 "values": [{"field": "Location", "value": "Lounge"}]},
            ],
        }
    )

    summaries = [e for e in build_evidence(request) if e.is_chart_summary]

    assert [(s.id, s.label, s.date) for s in summaries] == [("T1", "Fluid Chart", "23/10/2026")]
    # Figures exactly as recorded: range and the latest by date - no totals or averages.
    assert summaries[0].text == (
        "4 entries on 3 days (21/10/2026 to 23/10/2026); "
        "Amount (ml): 60 to 200, latest 150; Target (ml): 3000 every time"
    )
    assert summaries[0].linked_to_pathway is True and summaries[0].historical is False

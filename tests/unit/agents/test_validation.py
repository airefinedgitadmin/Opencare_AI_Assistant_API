from careplan_ai.agents.care_plan.schemas import (
    ActionOut,
    CardOrigin,
    Confidence,
    DraftPlanOut,
    EvidenceItem,
    NoteClassOut,
    NotUsedOut,
    RiskFlagOut,
    SourceType,
)
from careplan_ai.agents.care_plan.validation import (
    UNADDRESSED_RISK_WARNING,
    UNSUPPORTED_NAME_WARNING,
    UNSUPPORTED_NUMBER_WARNING,
    validate_draft,
)

EVIDENCE = [
    EvidenceItem(id="A1", source_type=SourceType.ASSESSMENT, label="Dentures", text="Yes"),
    EvidenceItem(id="A2", source_type=SourceType.ASSESSMENT, label="Oral cleanliness",
                 text="Some plaque"),
    EvidenceItem(id="P1", source_type=SourceType.PREVIOUS_CARE_PLAN, label="Previous care plan",
                 text="Cathy is independent with her oral care."),
    EvidenceItem(id="P2", source_type=SourceType.PREVIOUS_CARE_PLAN, label="Previous care plan",
                 text="She needs prompting from staff."),
    EvidenceItem(id="R1", source_type=SourceType.REVIEW_NOTE, label="Review note",
                 text="Staff may need to assist with oral hygiene."),
]


def _draft(**overrides: object) -> DraftPlanOut:
    base: dict[str, object] = {
        "risks": [RiskFlagOut(evidence_id="A1", reason="Has dentures"),
                  RiskFlagOut(evidence_id="A2", reason="Some plaque")],
        "note_classes": [NoteClassOut(evidence_id="R1", kind="new_requirement")],
        "not_used": [NotUsedOut(evidence_id="P1", superseded_by="R1", reason="Needs help now")],
        "actions": [
            ActionOut(text="Staff should support Cathy with her dentures.",
                      evidence_ids=["A1"], outcome_text="Cathy keeps using her dentures.",
                      outcome_evidence_ids=["A1"]),
            ActionOut(text="Staff should prompt Cathy with oral hygiene and assist when needed.",
                      evidence_ids=["A2", "P2", "R1"],
                      outcome_text="Cathy's mouth is kept clean.", outcome_evidence_ids=["A2"]),
        ],
    }
    base.update(overrides)
    return DraftPlanOut(**base)  # type: ignore[arg-type]


def test_clean_draft_passes_with_no_backfill() -> None:
    result = validate_draft(EVIDENCE, _draft(), resident_name="Cathy")

    assert [c.id for c in result.cards] == ["S1", "S2"]
    assert result.invalid_reference_count == 0
    assert result.backfilled_count == 0
    # The only problem - in words the care plan agent can fix on revision.
    assert result.problems == [
        'Action "Staff should prompt Cathy with oral hygiene and assist when needed." '
        "leaves out details of [A2]: plaque"
    ]
    # Assessment + previous plan + review note behind S2 -> high confidence.
    assert result.cards[1].confidence == Confidence.HIGH
    assert result.cards[0].confidence == Confidence.MEDIUM
    assert result.cards[1].risk_evidence_ids == ["A2"]
    assert [n.evidence_id for n in result.not_used] == ["P1"]


ORAL_READING = EvidenceItem(id="V1", source_type=SourceType.OBSERVATION, label="Oral Health",
                     text="Oral care: Prompted; Gums: Sore", date="13/10/2026")


def test_observations_support_actions_but_are_never_required_or_superseding() -> None:
    draft = _draft(
        note_classes=[NoteClassOut(evidence_id="R1", kind="new_requirement"),
                      NoteClassOut(evidence_id="V1", kind="new_requirement")],
        not_used=[NotUsedOut(evidence_id="P1", superseded_by="V1", reason="Gums sore")],
        carry_over=["V1"],
        risks=[RiskFlagOut(evidence_id="A1", reason="Has dentures"),
               RiskFlagOut(evidence_id="A2", reason="Some plaque"),
               RiskFlagOut(evidence_id="V1", reason="Sore gums")],
        actions=[
            ActionOut(text="Staff should support Cathy with her dentures and her sore gums.",
                      evidence_ids=["A1", "V1"], outcome_text="", outcome_evidence_ids=[]),
            ActionOut(text="Staff should prompt Cathy with oral hygiene to reduce plaque and "
                           "assist when needed.",
                      evidence_ids=["A2", "P2", "R1"], outcome_text="", outcome_evidence_ids=[]),
        ],
    )

    result = validate_draft([*EVIDENCE, ORAL_READING], draft, resident_name="Cathy")

    # A reading backs an action (and adds a second source type: high confidence) ...
    assert result.cards[0].evidence_ids == ["A1", "V1"]
    assert result.cards[0].confidence == Confidence.HIGH
    # ... but is not a note, a risk, a reason to drop P1, or a carry-over.
    for use in ("Risk 'V1'", "Note class 'V1'", "not_used 'P1 superseded by V1'",
                "carry_over 'V1'"):
        assert any(p.startswith(use) for p in result.problems), use
    assert result.invalid_reference_count == 4
    assert result.not_used == []
    # P1 is still current (its "superseding" reading doesn't count), so it is
    # added back word for word; the reading itself never is.
    backfilled = [c.evidence_ids for c in result.cards if c.origin == CardOrigin.SOURCE_TEXT]
    assert backfilled == [["P1"]]


def test_a_supporting_reading_never_drives_an_action_on_its_own() -> None:
    supporting = EvidenceItem(id="V2", source_type=SourceType.OBSERVATION, label="Fluid Chart",
                              text="Drink: Water; Amount (ml): 50", supporting=True)
    draft = _draft(actions=[
        ActionOut(text="Staff should encourage Cathy to drink more water.",
                  evidence_ids=["V2"], outcome_text="", outcome_evidence_ids=[]),
        ActionOut(text="Staff should support Cathy with her dentures.",
                  evidence_ids=["A1", "V2"], outcome_text="", outcome_evidence_ids=[]),
        ActionOut(text="Staff should prompt Cathy with oral hygiene to reduce plaque and "
                       "assist when needed.",
                  evidence_ids=["A2", "P2", "R1"], outcome_text="", outcome_evidence_ids=[]),
    ])

    result = validate_draft([*EVIDENCE, supporting], draft, resident_name="Cathy")

    texts = [c.text for c in result.cards]
    assert "Staff should encourage Cathy to drink more water." not in texts
    assert any("cites only supporting readings (V2)" in p for p in result.problems)
    # Alongside other evidence, a supporting reading is fine.
    assert result.cards[0].evidence_ids == ["A1", "V2"]


def test_invented_ids_are_discarded_and_unsupported_actions_dropped() -> None:
    draft = _draft(
        actions=[
            ActionOut(text="Staff should support Cathy with her dentures.",
                      evidence_ids=["A1", "X9"], outcome_text="", outcome_evidence_ids=[]),
            ActionOut(text="Staff should use a hoist for all transfers.",
                      evidence_ids=["Z1"], outcome_text="", outcome_evidence_ids=[]),
            ActionOut(text="Staff should prompt oral hygiene and assist.",
                      evidence_ids=["A2", "P2", "R1"], outcome_text="", outcome_evidence_ids=[]),
        ]
    )

    result = validate_draft(EVIDENCE, draft, resident_name="Cathy")

    texts = [c.text for c in result.cards]
    assert "Staff should use a hoist for all transfers." not in texts  # no real source
    assert result.cards[0].evidence_ids == ["A1"]  # invented X9 removed
    assert result.dropped_action_count == 1
    assert result.invalid_reference_count == 2
    assert any("'X9' is not a valid evidence id" in p for p in result.problems)
    assert any("has no valid evidence and was dropped" in p for p in result.problems)


def test_missed_items_are_added_back_word_for_word() -> None:
    # The model forgot the oral-cleanliness risk, P2 and the review note.
    draft = _draft(
        actions=[
            ActionOut(text="Staff should support Cathy with her dentures.",
                      evidence_ids=["A1"], outcome_text="", outcome_evidence_ids=[]),
        ]
    )

    result = validate_draft(EVIDENCE, draft, resident_name="Cathy")

    backfilled = [c for c in result.cards if c.origin == CardOrigin.SOURCE_TEXT]
    assert [c.evidence_ids for c in backfilled] == [["A2"], ["P2"], ["R1"]]
    assert backfilled[0].text == "Oral cleanliness: Some plaque"
    assert backfilled[0].warnings == [UNADDRESSED_RISK_WARNING]
    assert backfilled[1].text == "She needs prompting from staff."
    assert result.backfilled_count == 3


def test_superseded_previous_action_is_not_required_or_citable() -> None:
    draft = _draft(
        actions=[
            ActionOut(text="Staff should support Cathy with her dentures.",
                      evidence_ids=["A1", "P1"], outcome_text="", outcome_evidence_ids=[]),
            ActionOut(text="Staff should prompt oral hygiene and assist.",
                      evidence_ids=["A2", "P2", "R1"], outcome_text="", outcome_evidence_ids=[]),
        ]
    )

    result = validate_draft(EVIDENCE, draft, resident_name="Cathy")

    assert result.cards[0].evidence_ids == ["A1"]  # superseded P1 stripped
    assert result.backfilled_count == 0  # P1 isn't required once superseded


def test_supersession_must_point_at_a_real_review_note() -> None:
    draft = _draft(not_used=[NotUsedOut(evidence_id="P1", superseded_by="A1", reason="?")])

    result = validate_draft(EVIDENCE, draft, resident_name="Cathy")

    assert result.not_used == []
    # P1 is therefore still required, and nothing cites it -> added back.
    assert any(c.evidence_ids == ["P1"] for c in result.cards)


def test_numbers_not_in_sources_are_flagged() -> None:
    draft = _draft(
        actions=[
            ActionOut(text="Staff should check Cathy's dentures 3 times a day.",
                      evidence_ids=["A1"], outcome_text="", outcome_evidence_ids=[]),
            ActionOut(text="Staff should prompt oral hygiene and assist.",
                      evidence_ids=["A2", "P2", "R1"], outcome_text="", outcome_evidence_ids=[]),
        ]
    )

    result = validate_draft(EVIDENCE, draft, resident_name="Cathy")

    assert result.cards[0].warnings == [UNSUPPORTED_NUMBER_WARNING]
    assert result.cards[0].confidence == Confidence.LOW
    assert any(UNSUPPORTED_NUMBER_WARNING in p for p in result.problems)


def test_a_digit_inside_a_longer_number_does_not_support_a_number() -> None:
    from careplan_ai.agents.care_plan.validation import unsupported_numbers

    sources = ["Seen on 02/03/2025.", "Offer 250ml fluids at 08:30."]

    assert unsupported_numbers("Check 5 times.", sources) == ["5"]  # not "5" from 2025/250
    assert unsupported_numbers("Offer 250ml at 8:30 on 2/3.", sources) == []


def test_unclassified_review_note_counts_as_a_requirement() -> None:
    draft = _draft(
        note_classes=[],
        actions=[
            ActionOut(text="Staff should support Cathy with her dentures.",
                      evidence_ids=["A1"], outcome_text="", outcome_evidence_ids=[]),
            ActionOut(text="Staff should prompt Cathy with oral hygiene.",
                      evidence_ids=["A2", "P2"], outcome_text="", outcome_evidence_ids=[]),
        ],
    )

    result = validate_draft(EVIDENCE, draft, resident_name="Cathy")

    assert any(c.evidence_ids == ["R1"] for c in result.cards)


# --- other notes (carer / nursing / professional / relative) ---

CARER_GUMS = EvidenceItem(id="C1", source_type=SourceType.CARER_NOTE, label="Carer note",
                          text="Cathy said her gums were sore.", linked_to_pathway=False)
CARER_MUSIC = EvidenceItem(id="C2", source_type=SourceType.CARER_NOTE, label="Carer note",
                           text="Cathy enjoyed the music session.", linked_to_pathway=False)
GP_LINKED = EvidenceItem(id="G1", source_type=SourceType.PROFESSIONAL_NOTE, label="GP note",
                         text="Use a soft toothbrush.", linked_to_pathway=True)


def test_note_requirements_must_be_covered() -> None:
    evidence = EVIDENCE + [CARER_GUMS, CARER_MUSIC, GP_LINKED]
    draft = _draft(note_classes=[NoteClassOut(evidence_id="R1", kind="new_requirement"),
                                 NoteClassOut(evidence_id="C1", kind="new_requirement"),
                                 NoteClassOut(evidence_id="C2", kind="status")])

    result = validate_draft(evidence, draft, resident_name="Cathy")

    backfilled = [c.evidence_ids for c in result.cards if c.origin == CardOrigin.SOURCE_TEXT]
    # C1: the agent found a requirement. G1: linked to the pathway and unclassified
    # -> when unsure, show it. C2: resident-level, not a requirement -> optional.
    assert backfilled == [["C1"], ["G1"]]


def test_a_previous_action_can_be_superseded_by_any_note_type() -> None:
    nurse = EvidenceItem(id="N1", source_type=SourceType.NURSING_NOTE, label="Nursing note",
                         text="Cathy now needs full assistance.")
    draft = _draft(not_used=[NotUsedOut(evidence_id="P1", superseded_by="N1", reason="Changed")],
                   note_classes=[NoteClassOut(evidence_id="R1", kind="new_requirement"),
                                 NoteClassOut(evidence_id="N1", kind="update")])

    result = validate_draft(EVIDENCE + [nurse], draft, resident_name="Cathy")

    assert [(n.evidence_id, n.superseded_by) for n in result.not_used] == [("P1", "N1")]


# --- nothing removed when combining (deck slide 20) ---


def test_details_dropped_while_combining_are_listed_on_the_card() -> None:
    evidence = [
        EvidenceItem(id="P1", source_type=SourceType.PREVIOUS_CARE_PLAN, label="Previous",
                     text="Cathy has upper dentures."),
        EvidenceItem(id="R1", source_type=SourceType.REVIEW_NOTE, label="Review note",
                     text="Prompt oral hygiene after meals."),
    ]
    draft = DraftPlanOut(
        risks=[], note_classes=[NoteClassOut(evidence_id="R1", kind="new_requirement")],
        not_used=[],
        actions=[ActionOut(text="Staff should prompt Cathy with her dentures and oral hygiene.",
                           evidence_ids=["P1", "R1"], outcome_text="",
                           outcome_evidence_ids=[])],
    )

    result = validate_draft(evidence, draft, resident_name="Cathy")

    missing = {m.evidence_id: m.words for m in result.cards[0].possibly_missing}
    assert missing == {"P1": ["upper"], "R1": ["meals"]}
    assert result.missing_detail_count == 2


# --- unsupported names ---


def test_names_not_in_sources_are_flagged() -> None:
    draft = _draft(actions=[
        ActionOut(text="Staff should support Cathy and tell Dr Patel about her dentures.",
                  evidence_ids=["A1"], outcome_text="", outcome_evidence_ids=[]),
        ActionOut(text="Staff should prompt oral hygiene and assist.",
                  evidence_ids=["A2", "P2", "R1"], outcome_text="", outcome_evidence_ids=[]),
    ])

    result = validate_draft(EVIDENCE, draft, resident_name="Cathy")

    assert result.cards[0].warnings == [UNSUPPORTED_NAME_WARNING]
    assert result.cards[1].warnings == []


def test_possessive_names_with_either_apostrophe_are_recognised() -> None:
    draft = _draft(actions=[
        ActionOut(text="Staff should look after Cathyâ€™s dentures and Cathy's mouth.",
                  evidence_ids=["A1"], outcome_text="", outcome_evidence_ids=[]),
        ActionOut(text="Staff should prompt oral hygiene and assist.",
                  evidence_ids=["A2", "P2", "R1"], outcome_text="", outcome_evidence_ids=[]),
    ])

    result = validate_draft(EVIDENCE, draft, resident_name="Cathy")

    assert result.cards[0].warnings == []


# --- style examples (OPENCARE-51) ---

HOIST_EXAMPLE = (
    "Action:\n[Resident] uses a hoist for all transfers. Staff should encourage [Resident] "
    "to remain independent and promote dignity at all times."
)


def test_story_scenario_detail_borrowed_from_an_example_is_dropped() -> None:
    # "Given an example from another resident mentions 'uses a hoist' and this
    # resident's sources do not mention a hoist, the draft does not mention a hoist."
    draft = _draft(actions=[
        ActionOut(text="Staff should support Cathy with her dentures, using a hoist.",
                  evidence_ids=["A1"], outcome_text="", outcome_evidence_ids=[]),
        ActionOut(text="Staff should encourage Cathy to promote her dignity and independence "
                       "with oral hygiene, assisting when needed.",
                  evidence_ids=["A2", "P2", "R1"], outcome_text="", outcome_evidence_ids=[]),
    ])

    result = validate_draft(EVIDENCE, draft, resident_name="Cathy", examples=[HOIST_EXAMPLE])

    texts = " ".join(c.text for c in result.cards)
    assert "hoist" not in texts.lower()
    assert result.copied_count == 1
    # Style words lent by the example are fine; the dropped card's source comes back.
    assert result.cards[0].text.startswith("Staff should encourage Cathy")
    assert any(c.evidence_ids == ["A1"] and c.origin == CardOrigin.SOURCE_TEXT
               for c in result.cards)
    assert any("only a STYLE EXAMPLE contains (hoist)" in p for p in result.problems)


def test_a_detail_the_resident_also_has_is_not_treated_as_borrowed() -> None:
    hoist_note = EvidenceItem(id="C1", source_type=SourceType.CARER_NOTE, label="Carer note",
                              text="Cathy was moved with the hoist today.",
                              linked_to_pathway=False)
    draft = _draft(actions=[
        ActionOut(text="Staff should support Cathy with her dentures, using a hoist.",
                  evidence_ids=["A1"], outcome_text="", outcome_evidence_ids=[]),
        ActionOut(text="Staff should prompt oral hygiene and assist.",
                  evidence_ids=["A2", "P2", "R1"], outcome_text="", outcome_evidence_ids=[]),
    ])

    result = validate_draft(EVIDENCE, draft, resident_name="Cathy", examples=[HOIST_EXAMPLE],
                            resident_sources=[e.text for e in EVIDENCE] + [hoist_note.text])

    assert result.copied_count == 0


def test_placeholders_from_examples_never_reach_a_suggestion() -> None:
    draft = _draft(actions=[
        ActionOut(text="Staff should support [Resident] with her dentures.",
                  evidence_ids=["A1"], outcome_text="", outcome_evidence_ids=[]),
        ActionOut(text="Staff should prompt oral hygiene and assist.",
                  evidence_ids=["A2", "P2", "R1"], outcome_text="", outcome_evidence_ids=[]),
    ])

    result = validate_draft(EVIDENCE, draft, resident_name="Cathy")

    assert all("[Resident]" not in c.text for c in result.cards)
    assert result.copied_count == 1


def test_action_pointing_at_the_records_is_dropped() -> None:
    draft = _draft(
        actions=[
            ActionOut(text="Staff should support Cathy with her dentures.", evidence_ids=["A1"]),
            ActionOut(text="Staff should prompt Cathy with oral hygiene, plaque and assist.",
                      evidence_ids=["A2", "R1"]),
            ActionOut(text="Staff should follow the previous care plan item: She needs prompting.",
                      evidence_ids=["P2"]),
        ]
    )

    result = validate_draft(EVIDENCE, draft, resident_name="Cathy")

    assert result.dropped_action_count == 1
    assert any("refers to the records instead of describing the care" in p for p in result.problems)
    assert all("previous care plan" not in c.text for c in result.cards)


def test_carried_over_item_is_shown_as_written_without_a_problem() -> None:
    draft = _draft(
        actions=[
            ActionOut(text="Staff should support Cathy with her dentures.", evidence_ids=["A1"]),
            ActionOut(text="Staff should assist Cathy with oral hygiene and plaque.",
                      evidence_ids=["A2", "R1"]),
        ],
        carry_over=["P2", "A1"],  # A1 isn't a previous care plan item
    )

    result = validate_draft(EVIDENCE, draft, resident_name="Cathy")

    carried = [c for c in result.cards if c.evidence_ids == ["P2"]]
    assert len(carried) == 1
    assert carried[0].origin == CardOrigin.SOURCE_TEXT
    assert carried[0].text == "She needs prompting from staff."
    assert carried[0].warnings == []
    assert result.backfilled_count == 0
    assert not any("must be covered" in p for p in result.problems)
    assert any("carry_over 'A1'" in p for p in result.problems)

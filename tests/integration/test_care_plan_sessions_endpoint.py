"""The care plan agents through the real app - only the model calls are faked
(the agent's tool loop, code checks and graph run for real)."""

import pytest
from fastapi.testclient import TestClient

from agent_fakes import fake_agent, search
from careplan_ai.agents.care_plan.schemas import (
    ActionOut,
    DraftPlanOut,
    EvidenceItem,
    NoteClassOut,
    NotUsedOut,
    OutcomeOut,
    OutcomesOut,
    PatternOut,
    ReadOut,
    RelevantRecordOut,
    RiskFlagOut,
    VerdictOut,
    VerifyOut,
)
from careplan_ai.api.app import create_app

pytestmark = pytest.mark.integration

HEADERS = {"X-Api-Key": "test-api-key", "X-CarePlan-Version": "2"}

START_PAYLOAD = {
    "assessmentTemplateUserId": 42,
    "assessmentTemplateId": 7,
    "pathway": "Oral Health Assessment",
    "residentName": "Cathy",
    "assessment": {
        "completedDate": "14/10/2026",
        "answers": [
            {"questionId": "dentures", "question": "Dentures", "answer": "Yes"},
            {"questionId": "clean", "question": "Oral cleanliness", "answer": "Some plaque"},
            {"questionId": "pain", "question": "Dental pain", "answer": "No"},
        ],
    },
    "previousCarePlans": [
        {
            "carePlanId": 900,
            "carePlanDate": "20/05/2026",
            "action": (
                "<p>Cathy is independent with her oral care. "
                "She needs prompting from staff.</p>"
            ),
            "outcome": "<p>Cathy will not experience any oral discomfort.</p>",
        }
    ],
    "reviewNotes": [
        {"noteId": 5, "createdDate": "08/10/2026",
         "note": "Staff may be required to assist with providing oral hygiene."}
    ],
}

# Evidence ids for START_PAYLOAD: A1 dentures, A2 plaque, A3 no pain,
# P1 independent, P2 prompting, O1 no discomfort, R1 assist.
ACTION_1 = ActionOut(text="Staff should help Cathy look after her dentures.",
                     evidence_ids=["A1"], outcome_text="Cathy will keep using her dentures.",
                     outcome_evidence_ids=["A1"])
ACTION_2 = ActionOut(text="Staff should prompt Cathy with oral hygiene to reduce plaque, and "
                          "may need to assist with providing it.",
                     evidence_ids=["A2", "P2", "R1"],
                     outcome_text="Cathy will not experience any oral discomfort.",
                     outcome_evidence_ids=["A2", "O1"])
OUTCOME_FOR = {ACTION_1.text: ACTION_1.outcome_text, ACTION_2.text: ACTION_2.outcome_text}
OUTCOME_IDS_FOR = {ACTION_1.text: ["A1"], ACTION_2.text: ["A2", "O1"]}
# Passes every code check, so no revision is needed.
FAKE_DRAFT = DraftPlanOut(
    risks=[RiskFlagOut(evidence_id="A1", reason="Has dentures"),
           RiskFlagOut(evidence_id="A2", reason="Some plaque")],
    note_classes=[NoteClassOut(evidence_id="R1", kind="new_requirement")],
    not_used=[NotUsedOut(evidence_id="P1", superseded_by="R1", reason="Now needs assistance")],
    actions=[ACTION_1, ACTION_2],
)


def _all_supported(statements: list[tuple[str, str, list[EvidenceItem]]]) -> VerifyOut:
    return VerifyOut(verdicts=[VerdictOut(ref=ref, verdict="supported", reason="")
                               for ref, _, _ in statements])


@pytest.fixture
def fake_models(monkeypatch: pytest.MonkeyPatch) -> dict[str, list]:  # type: ignore[type-arg]
    """Care plan agent submits FAKE_DRAFT; verifier agent supports everything."""
    calls: dict[str, list] = {"redraft": [], "verify": []}  # type: ignore[type-arg]
    calls["draft"] = fake_agent(monkeypatch, lambda role, messages: FAKE_DRAFT)
    redraft_texts = iter(["Staff should support Cathy to care for her dentures.",
                          "Staff should help with dentures 4 times daily."])

    async def _redraft(**kwargs: object) -> str:
        calls["redraft"].append(kwargs)
        return next(redraft_texts)

    async def _outcomes(*, resident_name: str | None,
                        actions: list[tuple[str, str, list[EvidenceItem]]],
                        examples: object = None, feedback: object = None) -> OutcomesOut:
        calls["outcomes"].append([(ref, [q.id for q in quotes]) for ref, _, quotes in actions])
        calls["outcome_feedback"].append(feedback)
        # Staff may have edited the action before adding it - the outcome is
        # written for their wording.
        return OutcomesOut(outcomes=[
            OutcomeOut(ref=ref, text=OUTCOME_FOR.get(text, ACTION_1.outcome_text),
                       evidence_ids=OUTCOME_IDS_FOR.get(text, ["A1"]))
            for ref, text, _ in actions
        ])

    calls["outcomes"] = []
    calls["outcome_feedback"] = []
    monkeypatch.setattr("careplan_ai.agents.care_plan.llm.write_outcomes", _outcomes)

    async def _verify(*, resident_name: str | None,
                      statements: list[tuple[str, str, list[EvidenceItem]]]) -> VerifyOut:
        calls["verify"].append([(ref, text, [q.id for q in quotes])
                                for ref, text, quotes in statements])
        return _all_supported(statements)

    async def _read(*, resident_name: str | None, pathway: str | None,
                    context: list[EvidenceItem], records: list[EvidenceItem]) -> ReadOut:
        # Reader agents: nothing found by default (tests script their own).
        calls["read"].append([e.id for e in records])
        return ReadOut(relevant=[], not_relevant=[e.id for e in records])

    calls["read"] = []
    monkeypatch.setattr("careplan_ai.agents.care_plan.llm.generate_redraft", _redraft)
    monkeypatch.setattr("careplan_ai.agents.care_plan.llm.verify_statements", _verify)
    monkeypatch.setattr("careplan_ai.agents.care_plan.llm.read_records", _read)
    return calls


def _start(client: TestClient, payload: dict | None = None) -> dict:  # type: ignore[type-arg]
    response = client.post("/care-plans/sessions", json=payload or START_PAYLOAD, headers=HEADERS)
    assert response.status_code == 201, response.text
    return response.json()


def test_start_session_returns_checked_and_verified_cards_with_traceable_evidence(
    base_env: None, fake_models: dict[str, list]  # type: ignore[type-arg]
) -> None:
    with TestClient(create_app()) as client:
        session = _start(client)

    assert fake_models["draft"][0]["evidence"] == ["A1", "A2", "A3", "P1", "P2", "O1", "R1"]
    assert len(fake_models["draft"]) == 1  # clean draft -> no revision
    assert [c["id"] for c in session["cards"]] == ["S1", "S2"]
    assert session["nextActionCardId"] == "S1"
    assert session["nextOutcomeCardId"] is None
    assert session["allPromptsHandled"] is False
    assert session["notUsed"][0]["evidenceId"] == "P1"
    # "View" quotes come from the evidence the backend sent, not from the model.
    evidence = {e["id"]: e for e in session["evidence"]}
    assert evidence["R1"]["text"] == "Staff may be required to assist with providing oral hygiene."
    assert session["cards"][1]["confidence"] == "high"
    assert session["generation"]["modelId"] == "gpt-5"
    # Only actions are drafted and verified up front - each with ONLY its quotes.
    assert fake_models["verify"][0] == [
        ("S1", ACTION_1.text, ["A1"]),
        ("S2", ACTION_2.text, ["A2", "P2", "R1"]),
    ]
    assert [(c["verification"], c["outcome"]) for c in session["cards"]] == [
        ("supported", None), ("supported", None)
    ]
    assert fake_models["outcomes"] == []  # nothing asked for outcomes yet
    assert session["outcomesToWrite"] == []
    assert session["generation"]["verifierRan"] is True
    assert session["generation"]["revisions"] == 0


def test_outcomes_are_written_only_for_added_actions_when_asked(
    base_env: None, fake_models: dict[str, list]  # type: ignore[type-arg]
) -> None:
    with TestClient(create_app()) as client:
        session = _start(client)
        sid = session["sessionId"]
        url = f"/care-plans/sessions/{sid}"
        client.post(f"{url}/cards/S1/decision", json={"decision": "add"}, headers=HEADERS)
        s = client.post(f"{url}/cards/S2/decision", json={"decision": "dismiss"},
                        headers=HEADERS).json()
        assert s["outcomesToWrite"] == ["S1"]
        assert s["allPromptsHandled"] is False  # S1's outcome is still to come

        s = client.post(f"{url}/outcomes", headers=HEADERS).json()
        again = client.post(f"{url}/outcomes", headers=HEADERS).json()  # nothing left to write

    # One call, for the added action only - with its quotes plus previous outcomes.
    assert fake_models["outcomes"] == [[("S1", ["A1", "O1"])]]
    assert fake_models["verify"][-1] == [("S1.outcome", ACTION_1.outcome_text, ["A1", "O1"])]
    outcome = s["cards"][0]["outcome"]
    assert (outcome["text"], outcome["evidenceIds"], outcome["verification"]) == (
        ACTION_1.outcome_text, ["A1"], "supported")
    assert s["cards"][1]["outcome"] is None  # dismissed action: never gets one
    assert s["nextOutcomeCardId"] == "S1" and s["outcomesToWrite"] == []
    assert len(fake_models["outcomes"]) == 1 and again["cards"] == s["cards"]


def test_a_rejected_outcome_is_rewritten_once_with_the_verifiers_reason(
    base_env: None, fake_models: dict[str, list], monkeypatch: pytest.MonkeyPatch  # type: ignore[type-arg]
) -> None:
    verdicts = iter(["unsupported", "supported"])
    with TestClient(create_app()) as client:
        session = _start(client)
        url = f"/care-plans/sessions/{session['sessionId']}"
        client.post(f"{url}/cards/S1/decision", json={"decision": "add"}, headers=HEADERS)

        async def _verify(**_: object) -> VerifyOut:
            return VerifyOut(verdicts=[VerdictOut(ref="S1.outcome", verdict=next(verdicts),
                                                  reason="'keep using'")])

        monkeypatch.setattr("careplan_ai.agents.care_plan.llm.verify_statements", _verify)
        s = client.post(f"{url}/outcomes", headers=HEADERS).json()

    assert fake_models["outcome_feedback"] == [
        {}, {"S1": f"{ACTION_1.outcome_text} - 'keep using'"}
    ]
    assert s["cards"][0]["outcome"]["verification"] == "supported"


def test_an_outcome_the_verifier_rejects_is_not_offered(
    base_env: None, fake_models: dict[str, list], monkeypatch: pytest.MonkeyPatch  # type: ignore[type-arg]
) -> None:
    with TestClient(create_app()) as client:
        session = _start(client)
        url = f"/care-plans/sessions/{session['sessionId']}"
        client.post(f"{url}/cards/S1/decision", json={"decision": "add"}, headers=HEADERS)

        async def _verify(**_: object) -> VerifyOut:
            return VerifyOut(verdicts=[VerdictOut(ref="S1.outcome", verdict="unsupported",
                                                  reason="'keep using'")])

        monkeypatch.setattr("careplan_ai.agents.care_plan.llm.verify_statements", _verify)
        s = client.post(f"{url}/outcomes", headers=HEADERS).json()

    assert s["cards"][0]["outcome"] is None
    assert s["outcomesToWrite"] == [] and s["nextOutcomeCardId"] is None


def test_care_plan_agent_revises_from_the_exact_problems_the_checks_found(
    base_env: None, fake_models: dict[str, list], monkeypatch: pytest.MonkeyPatch  # type: ignore[type-arg]
) -> None:
    first = FAKE_DRAFT.model_copy(update={"actions": [ACTION_1]})  # forgets R1 / A2 / P2
    drafts = iter([first, FAKE_DRAFT])
    turns = fake_agent(monkeypatch, lambda role, messages: next(drafts))
    with TestClient(create_app()) as client:
        session = _start(client)

    # The code checks answered its submission with the exact problems.
    feedback = turns[1]["feedback"]
    assert feedback.startswith("Not accepted. Fix every one of these")
    assert "- [R1] must be covered" in feedback
    assert "- [A2] must be covered" in feedback
    assert session["generation"]["revisions"] == 1
    assert [c["origin"] for c in session["cards"]] == ["ai", "ai"]  # fixed, nothing backfilled


def test_verifier_rejection_goes_back_once_then_the_statement_is_replaced_by_its_sources(
    base_env: None, fake_models: dict[str, list], monkeypatch: pytest.MonkeyPatch  # type: ignore[type-arg]
) -> None:
    turns = fake_agent(monkeypatch, lambda role, messages: FAKE_DRAFT)  # never fixes it

    async def _verify(*, resident_name: str | None,
                      statements: list[tuple[str, str, list[EvidenceItem]]]) -> VerifyOut:
        verdicts = _all_supported(statements).verdicts
        verdicts[1] = VerdictOut(ref="S2", verdict="contradicts",
                                 reason="'prompt' - R1 says staff may need to assist")
        verdicts[0] = VerdictOut(ref="S1", verdict="overstated",
                                 reason="'look after' - A1 only says she has dentures")
        return VerifyOut(verdicts=verdicts)

    monkeypatch.setattr("careplan_ai.agents.care_plan.llm.verify_statements", _verify)
    with TestClient(create_app()) as client:
        session = _start(client)

    # The objections went back into the same agent conversation, in words it can act on.
    verifier_notes = [t["feedback"] for t in turns if t["feedback"].startswith("VERIFIER")]
    assert len(verifier_notes) == 2  # sent back up to agent_max_revisions (2)
    assert "is contradicts by its quotes: 'prompt'" in verifier_notes[0]
    assert session["generation"]["revisions"] == 2
    assert session["generation"]["rejectedCount"] == 1
    cards = session["cards"]
    # S2 is gone; the required sources it covered come back word for word.
    assert ACTION_2.text not in [c["text"] for c in cards]
    assert [c["evidenceIds"] for c in cards if c["origin"] == "source_text"] == [
        ["A2"], ["P2"], ["R1"]
    ]
    assert all(c["verification"] == "source_text" for c in cards if c["origin"] == "source_text")
    # The overstated action stays, flagged and low confidence.
    assert cards[0]["verification"] == "overstated"
    assert cards[0]["confidence"] == "low"
    assert "'look after'" in cards[0]["warnings"][0]


def test_verifier_unavailable_marks_statements_not_verified_instead_of_failing(
    base_env: None, fake_models: dict[str, list], monkeypatch: pytest.MonkeyPatch  # type: ignore[type-arg]
) -> None:
    async def _verify(**_: object) -> VerifyOut:
        raise RuntimeError("429")

    monkeypatch.setattr("careplan_ai.agents.care_plan.llm.verify_statements", _verify)
    with TestClient(create_app()) as client:
        session = _start(client)

    assert session["generation"]["verifierRan"] is False
    for card in session["cards"]:
        assert card["verification"] == "not_verified"
        assert card["confidence"] == "low"
        assert any("Not checked" in w for w in card["warnings"])


def test_redraft_is_kept_only_when_code_checks_and_the_verifier_confirm_it(
    base_env: None, fake_models: dict[str, list], monkeypatch: pytest.MonkeyPatch  # type: ignore[type-arg]
) -> None:
    with TestClient(create_app()) as client:
        session = _start(client)
        url = f"/care-plans/sessions/{session['sessionId']}/cards"
        first = client.post(f"{url}/S1/redraft", json={"target": "action"}, headers=HEADERS).json()
        second = client.post(f"{url}/S1/redraft", json={"target": "action"}, headers=HEADERS).json()

        async def _verify(*, resident_name: str | None,
                          statements: list[tuple[str, str, list[EvidenceItem]]]) -> VerifyOut:
            return VerifyOut(verdicts=[VerdictOut(ref="S1", verdict="unsupported", reason="x")])

        monkeypatch.setattr("careplan_ai.agents.care_plan.llm.verify_statements", _verify)
        monkeypatch.setattr("careplan_ai.agents.care_plan.llm.generate_redraft",
                            lambda **_: _async("Staff should support Cathy with her dentures."))
        third = client.post(f"{url}/S1/redraft", json={"target": "action"}, headers=HEADERS).json()

    s1 = first["cards"][0]
    assert s1["text"] == "Staff should support Cathy to care for her dentures."
    assert s1["verification"] == "supported"
    assert s1["wordingHistory"] == ["Staff should help Cathy look after her dentures."]
    assert s1["evidenceIds"] == ["A1"]  # same evidence
    assert first["cards"][1] == session["cards"][1]  # other cards untouched
    # Second redraft invented "4 times daily" -> rejected by code, wording kept.
    assert second["cards"][0]["text"] == s1["text"]
    assert "Redraft discarded" in second["cards"][0]["warnings"][0]
    # Third passed the code checks but the verifier rejected it -> wording kept.
    assert third["cards"][0]["text"] == s1["text"]


async def _async(value: str) -> str:
    return value


def test_one_by_one_flow_outcomes_follow_added_actions_and_finalize_saves_trace(
    base_env: None, fake_models: dict[str, list]  # type: ignore[type-arg]
) -> None:
    with TestClient(create_app()) as client:
        session = _start(client)
        sid = session["sessionId"]
        url = f"/care-plans/sessions/{sid}/cards"

        # Outcome can't be decided before its action is added (deck slide 13).
        early = client.post(f"{url}/S1/decision", json={"target": "outcome", "decision": "add"},
                            headers=HEADERS)
        assert early.status_code == 409

        edited = "Staff should help Cathy with her dentures."
        s = client.post(f"{url}/S1/decision", headers=HEADERS,
                        json={"decision": "add", "text": edited}).json()
        assert s["cards"][0]["edited"] is True
        assert s["nextActionCardId"] == "S2"
        assert s["outcomesToWrite"] == ["S1"]

        s = client.post(f"{url}/S2/decision", json={"decision": "dismiss"}, headers=HEADERS).json()
        # Dismissed action -> no outcome is ever written for it.
        assert s["cards"][1]["outcome"] is None

        s = client.post(f"/care-plans/sessions/{sid}/outcomes", headers=HEADERS).json()
        assert s["nextOutcomeCardId"] == "S1"
        s = client.post(f"{url}/S1/decision", json={"target": "outcome", "decision": "add"},
                        headers=HEADERS).json()
        assert s["allPromptsHandled"] is True

        trace = client.post(f"/care-plans/sessions/{sid}/finalize", json={"carePlanId": 1234},
                            headers=HEADERS)
        assert trace.status_code == 200, trace.text
        subject = client.get("/care-plans/traces/1234/subject", headers=HEADERS).json()
        # The backend rebuilds the evidence for that assessment and sends it.
        stored = client.post("/care-plans/traces/1234/view", headers=HEADERS,
                             json={"evidence": START_PAYLOAD}).json()
        blind = client.get("/care-plans/traces/1234", headers=HEADERS).json()
        after = client.post(f"{url}/S2/decision", json={"decision": "add"}, headers=HEADERS)

    assert subject["assessmentTemplateUserId"] == 42 and subject["evidenceAsOf"]
    statements = stored["statements"]
    assert [(s["cardId"], s["field"]) for s in statements] == [("S1", "action"), ("S1", "outcome")]
    assert statements[0]["text"] == "Staff should help Cathy with her dentures."
    assert statements[0]["edited"] is True
    assert statements[0]["verification"] == "supported"
    assert statements[0]["evidence"][0]["text"] == "Yes"
    assert stored["notUsed"][0]["supersededBy"] == "R1"
    # Without the evidence the trace still holds its statements, but no record text.
    assert blind["statements"][0]["text"] == "Staff should help Cathy with her dentures."
    assert blind["statements"][0]["evidence"][0]["changed"] is True
    assert after.status_code == 409  # finalized sessions are read-only


def test_agent_memory_holds_no_record_text_or_name(
    base_env: None, fake_models: dict[str, list]  # type: ignore[type-arg]
) -> None:
    import json

    from careplan_ai.memory.store import get_session_store

    with TestClient(create_app()) as client:
        session = _start(client)
        sid = session["sessionId"]
        stored = get_session_store()._sessions[sid]  # type: ignore[attr-defined]
        # Without the evidence pack, records come back as changed - never stale.
        bare = client.get(f"/care-plans/sessions/{sid}", headers=HEADERS).json()
        # With it, the same ids and text as at the start.
        viewed = client.post(f"/care-plans/sessions/{sid}/view", headers=HEADERS,
                             json={"evidence": START_PAYLOAD}).json()
        other = client.post(f"/care-plans/sessions/{sid}/view", headers=HEADERS,
                            json={"evidence": {**START_PAYLOAD, "assessmentTemplateUserId": 7}})

    # Patient content: the name, answers, care plan and note wording, AI wording.
    # (Record references - e.g. the form's question id "dentures" - are kept.)
    for secret in ("Cathy", "plaque", "independent with her oral care",
                   "assist with providing oral hygiene", "Staff should"):
        assert secret not in stored, secret
    assert set(json.loads(stored)) == {"session", "evidence", "sealed"}
    assert all(e["changed"] for e in bare["evidence"])
    assert bare["residentName"] is None
    assert [(e["id"], e["text"]) for e in viewed["evidence"]] == [
        (e["id"], e["text"]) for e in session["evidence"]
    ]
    assert viewed["residentName"] == "Cathy"
    assert viewed["cards"] == session["cards"]
    assert other.status_code == 422  # evidence for another assessment is refused


def test_a_record_changed_since_the_draft_is_flagged_not_quoted_stale(
    base_env: None, fake_models: dict[str, list]  # type: ignore[type-arg]
) -> None:
    changed = {**START_PAYLOAD, "reviewNotes": [
        {"noteId": 5, "createdDate": "08/10/2026", "note": "Staff now provide full oral care."}
    ]}
    with TestClient(create_app()) as client:
        sid = _start(client)["sessionId"]
        viewed = client.post(f"/care-plans/sessions/{sid}/view", headers=HEADERS,
                             json={"evidence": changed}).json()

    by_id = {e["id"]: e for e in viewed["evidence"]}
    assert by_id["R1"]["changed"] is True
    assert by_id["R1"]["text"] != "Staff now provide full oral care."
    assert by_id["A1"]["changed"] is False and by_id["A1"]["text"] == "Yes"


MUSIC_AND_TEETH_NOTES = [
    *({"noteId": 80 + i, "noteType": "carer", "createdDate": f"{10 + i:02d}/10/2026",
       "note": "Cathy enjoyed the music session."} for i in range(4)),
    {"noteId": 90, "noteType": "carer", "createdDate": "01/10/2026",
     "note": "Cathy said her dentures hurt."},
    {"noteId": 91, "noteType": "carer", "createdDate": "02/03/2025", "historical": True,
     "note": "Cathy refused to wear her dentures."},
]


def _small_pack(monkeypatch: pytest.MonkeyPatch) -> None:
    """A pack with room for the must-read records plus the two dentures notes."""
    from careplan_ai.agents.care_plan.context import count_tokens, is_must_read, total_tokens
    from careplan_ai.agents.care_plan.evidence import build_evidence
    from careplan_ai.api.schemas.agent import StartSessionRequest

    evidence = build_evidence(StartSessionRequest.model_validate(
        {**START_PAYLOAD, "notes": MUSIC_AND_TEETH_NOTES}))
    must = total_tokens([e for e in evidence if is_must_read(e)])
    dentures = sum(count_tokens(e) for e in evidence if e.id in ("C5", "C6"))
    monkeypatch.setenv("CAREPLAN_AI_AGENT_PACK_TOKENS", str(must + dentures))


def test_each_need_gets_its_best_matches_and_the_rest_stays_searchable(
    base_env: None, fake_models: dict[str, list], monkeypatch: pytest.MonkeyPatch  # type: ignore[type-arg]
) -> None:
    _small_pack(monkeypatch)
    with TestClient(create_app()) as client:
        session = _start(client, {**START_PAYLOAD, "notes": MUSIC_AND_TEETH_NOTES})

    shown = fake_models["draft"][0]["evidence"]
    # The dentures need (A1) pulled in its matching notes - current and history -
    # ahead of four unrelated notes about music.
    assert "C5" in shown and "C6" in shown
    assert not {"C1", "C2", "C3", "C4"} & set(shown)
    # The four music notes didn't fit: a reader agent read them (as one line).
    assert fake_models["read"] == [["C1", "C2", "C3", "C4"]]
    assert [(s["evidenceId"], s["kind"]) for s in session["screened"]] == [
        ("C1", "not_relevant"), ("C2", "not_relevant"), ("C3", "not_relevant"),
        ("C4", "not_relevant"),
    ]
    assert session["omitted"] == []
    assert session["notices"][-1] == (
        "The AI read all 13 records. 4 of them have nothing for this pathway."
    )


def test_a_need_found_by_a_reader_must_be_covered_or_is_shown_word_for_word(
    base_env: None, fake_models: dict[str, list], monkeypatch: pytest.MonkeyPatch  # type: ignore[type-arg]
) -> None:
    _small_pack(monkeypatch)

    async def _read(**kwargs: object) -> ReadOut:
        # Any id of a repeated record's line answers for the whole line.
        return ReadOut(relevant=[RelevantRecordOut(evidence_id="C2", kind="new_requirement",
                                                   reason="enjoyed the music session")],
                       not_relevant=[])

    monkeypatch.setattr("careplan_ai.agents.care_plan.llm.read_records", _read)
    with TestClient(create_app()) as client:
        session = _start(client, {**START_PAYLOAD, "notes": MUSIC_AND_TEETH_NOTES})

    # C2 is the same note as C1, C3 and C4 - the reader's answer covers the line.
    kinds = {s["evidenceId"]: s["kind"] for s in session["screened"]}
    assert kinds == {"C1": "new_requirement", "C2": "new_requirement",
                     "C3": "new_requirement", "C4": "new_requirement"}
    # The care plan agent got the record, and no action cited it: shown word for word.
    assert {"C1", "C2", "C3", "C4"} <= set(fake_models["draft"][0]["evidence"])
    backfilled = [c for c in session["cards"] if c["origin"] == "source_text"]
    assert [c["evidenceIds"] for c in backfilled] == [["C1"]]
    assert backfilled[0]["text"] == "Cathy enjoyed the music session."


def test_records_no_reader_could_read_are_reported_not_dropped(
    base_env: None, fake_models: dict[str, list], monkeypatch: pytest.MonkeyPatch  # type: ignore[type-arg]
) -> None:
    _small_pack(monkeypatch)

    async def _down(**kwargs: object) -> ReadOut:
        raise RuntimeError("Azure unavailable")

    monkeypatch.setattr("careplan_ai.agents.care_plan.llm.read_records", _down)
    with TestClient(create_app()) as client:
        session = _start(client, {**START_PAYLOAD, "notes": MUSIC_AND_TEETH_NOTES})

    assert session["omitted"] == ["C1", "C2", "C3", "C4"]
    assert session["notices"][-1] == (
        "The AI could not read 4 of 13 records - check them yourself before saving."
    )
    assert session["cards"]  # the draft still goes ahead


def test_the_agent_can_search_the_rest_of_the_record(
    base_env: None, fake_models: dict[str, list], monkeypatch: pytest.MonkeyPatch  # type: ignore[type-arg]
) -> None:
    _small_pack(monkeypatch)
    replies = iter([search("music session"), FAKE_DRAFT])
    turns = fake_agent(monkeypatch, lambda role, messages: next(replies))
    with TestClient(create_app()) as client:
        session = _start(client, {**START_PAYLOAD, "notes": MUSIC_AND_TEETH_NOTES})

    # The search result: the four identical notes shown once, with every id.
    assert turns[1]["feedback"].startswith("[C1, C2, C3, C4] Carer note (x4: 10/10/2026")
    assert session["omitted"] == []  # everything has now been seen
    assert session["generation"]["searches"] == ["music session"]


def test_history_and_patterns_inform_the_plan_but_never_become_requirements(
    base_env: None, fake_models: dict[str, list], monkeypatch: pytest.MonkeyPatch  # type: ignore[type-arg]
) -> None:
    payload = {
        **START_PAYLOAD,
        "notes": [{"noteType": "carer", "createdDate": "02/03/2025", "historical": True,
                   "note": "Cathy refused to brush her teeth again."}],
        "relatedCarePlans": [
            {"carePlanId": 800, "carePlanDate": "01/02/2025", "pathway": "Oral Health Assessment",
             "status": "archived", "action": "Staff should help Cathy clean her teeth daily."},
        ],
    }
    draft = FAKE_DRAFT.model_copy(update={
        "note_classes": FAKE_DRAFT.note_classes + [NoteClassOut(evidence_id="C1",
                                                                kind="new_requirement")],
        "patterns": [
            PatternOut(summary="Cathy has needed staff help with her teeth before.",
                       evidence_ids=["H1", "C1"]),
            PatternOut(summary="Invented 5 refusals.", evidence_ids=["H1", "C1"]),
        ],
    })

    fake_agent(monkeypatch, lambda role, messages: draft)
    with TestClient(create_app()) as client:
        session = _start(client, payload)

    # Only the pattern backed by real items, with no invented figure, is kept.
    assert [p["evidenceIds"] for p in session["patterns"]] == [["H1", "C1"]]
    # History is never a requirement on its own, even if the model says so.
    backfilled = [c["evidenceIds"] for c in session["cards"] if c["origin"] == "source_text"]
    assert ["C1"] not in backfilled and ["H1"] not in backfilled
    labels = {e["id"]: e["label"] for e in session["evidence"]}
    assert labels["H1"] == "Earlier care plan (Oral Health Assessment, archived) - action"


def test_observation_entries_are_numbered_shown_and_citable(
    base_env: None, fake_models: dict[str, list], monkeypatch: pytest.MonkeyPatch  # type: ignore[type-arg]
) -> None:
    payload = {
        **START_PAYLOAD,
        "observations": [
            {"entryId": "65f0a1", "observationType": "OF_OH", "observationName": "Oral Health",
             "recordedDate": "13/10/2026", "pathwayLinked": True,
             "values": [{"field": "Oral care", "value": "Prompted"},
                        {"field": "Gums", "value": "Sore"}]},
        ],
    }
    draft = FAKE_DRAFT.model_copy(update={"actions": [
        ACTION_1.model_copy(update={"evidence_ids": ["A1", "V1"]}), ACTION_2,
    ]})
    fake_agent(monkeypatch, lambda role, messages: draft)
    with TestClient(create_app()) as client:
        session = _start(client, payload)

    v1 = next(e for e in session["evidence"] if e["id"] == "V1")
    assert (v1["sourceType"], v1["label"], v1["sourceRef"], v1["text"]) == (
        "observation", "Oral Health", "65f0a1", "Oral care: Prompted; Gums: Sore"
    )
    assert session["cards"][0]["evidenceIds"] == ["A1", "V1"]
    # The verifier judges the action against the reading it cites.
    assert ("S1", ACTION_1.text, ["A1", "V1"]) in fake_models["verify"][0]
    assert session["omitted"] == []


def test_rate_limit_wait_follows_azure_retry_after(base_env: None) -> None:
    import httpx
    import openai
    import tenacity

    from careplan_ai.agents.base.llm import _wait

    def state_for(exc: BaseException) -> tenacity.RetryCallState:
        state = tenacity.RetryCallState(None, None, (), {})  # type: ignore[arg-type]
        state.attempt_number = 1
        future: tenacity.Future = tenacity.Future(1)
        future.set_exception(exc)
        state.outcome = future
        return state

    def rate_limit(headers: dict[str, str]) -> openai.RateLimitError:
        request = httpx.Request("POST", "https://example")
        return openai.RateLimitError("429", response=httpx.Response(429, headers=headers,
                                                                    request=request), body=None)

    assert _wait(state_for(rate_limit({"retry-after": "7"}))) == 7.0
    assert _wait(state_for(rate_limit({"retry-after": "120"}))) == 20.0  # capped
    assert _wait(state_for(rate_limit({}))) == 5.0
    assert _wait(state_for(ValueError("boom"))) == 0.5


def test_agent_endpoints_require_version_2_header(base_env: None) -> None:
    with TestClient(create_app()) as client:
        missing = client.post("/care-plans/sessions", json=START_PAYLOAD,
                              headers={"X-Api-Key": "test-api-key"})
        wrong = client.post("/care-plans/sessions", json=START_PAYLOAD,
                            headers={"X-Api-Key": "test-api-key", "X-CarePlan-Version": "1"})

    assert missing.status_code == 400
    assert missing.json()["error_code"] == "unsupported_version"
    assert wrong.status_code == 400


def test_agent_endpoints_require_api_key(base_env: None) -> None:
    with TestClient(create_app()) as client:
        response = client.post("/care-plans/sessions", json=START_PAYLOAD,
                               headers={"X-CarePlan-Version": "2"})

    assert response.status_code == 401


BFRA_QUESTIONS = [{"question": "Does the resident smoke?", "answer": None},
                  {"question": "Are there electrical heaters in the room?", "answer": None}]


def test_every_record_is_used_and_unanswered_questions_are_listed(
    base_env: None, fake_models: dict[str, list]  # type: ignore[type-arg]
) -> None:
    # The BFRA case: form not answered; previous plan and notes say "test".
    payload = {
        "assessmentTemplateUserId": 66093, "pathway": "Bedroom Fire Risk Assessment (BFRA)",
        "residentName": "Cathy", "assessment": {"answers": BFRA_QUESTIONS},
        "previousCarePlans": [{"carePlanId": 1, "action": "<p>test</p>", "outcome": "test"}],
        "notes": [{"noteType": "nursing", "pathwayLinked": True, "note": "test"}],
    }
    with TestClient(create_app()) as client:
        session = _start(client, payload)

    # "test" records are records like any other: the agent gets all of them.
    assert fake_models["draft"][0]["evidence"] == ["P1", "O1", "N1"]
    assert session["notices"] == [
        "Bedroom Fire Risk Assessment (BFRA) has no answers yet - complete it first: "
        "suggestions are built from its answers.",
        "Not answered in Bedroom Fire Risk Assessment (BFRA): Does the resident smoke?; "
        "Are there electrical heaters in the room?.",
        "The AI read all 3 records.",
    ]
    # Everything fitted the pack, so no reader agent was needed.
    assert fake_models["read"] == []


def test_only_unanswered_questions_gives_a_message_without_a_model_call(
    base_env: None, fake_models: dict[str, list]  # type: ignore[type-arg]
) -> None:
    with TestClient(create_app()) as client:
        session = _start(client, {"assessmentTemplateUserId": 1, "pathway": "BFRA",
                                  "assessment": {"answers": BFRA_QUESTIONS}})

    assert fake_models["draft"] == [] and session["cards"] == []
    assert session["notices"][0] == "There are no records to suggest care plan actions from yet."
    assert session["allPromptsHandled"] is True


def test_start_without_any_evidence_is_rejected(
    base_env: None, fake_models: dict[str, list]  # type: ignore[type-arg]
) -> None:
    with TestClient(create_app()) as client:
        response = client.post("/care-plans/sessions", headers=HEADERS,
                               json={"assessmentTemplateUserId": 1, "assessment": {"answers": []}})

    assert response.status_code == 422
    assert fake_models["draft"] == []  # no model call wasted


def test_unknown_session_returns_404(base_env: None) -> None:
    with TestClient(create_app()) as client:
        response = client.get("/care-plans/sessions/does-not-exist", headers=HEADERS)

    assert response.status_code == 404


def test_revisions_use_the_fallback_model_when_configured(
    base_env: None, fake_models: dict[str, list], monkeypatch: pytest.MonkeyPatch  # type: ignore[type-arg]
) -> None:
    monkeypatch.setenv("CAREPLAN_AI_AGENT_FALLBACK_DEPLOYMENT", "gpt-5-big")
    monkeypatch.setenv("CAREPLAN_AI_AGENT_FALLBACK_MODEL_ID", "gpt-5")
    roles: list[str] = []
    weak = DraftPlanOut(risks=[], note_classes=[], not_used=[],
                        actions=[ActionOut(text="Use a hoist.", evidence_ids=["Z9"],
                                           outcome_text="", outcome_evidence_ids=[])])

    def _reply(role: str, messages: object) -> DraftPlanOut:
        roles.append(role)
        return weak if role == "primary" else FAKE_DRAFT

    fake_agent(monkeypatch, _reply)

    with TestClient(create_app()) as client:
        session = _start(client)

    assert roles == ["primary", "fallback"]
    assert session["generation"]["fallbackUsed"] is True
    assert "Use a hoist." not in [c["text"] for c in session["cards"]]

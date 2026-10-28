"""OPENCARE-51 acceptance criteria through the real app, on the in-memory
vector store (same interface as Pinecone). Only the model calls are faked."""

import pytest
from fastapi.testclient import TestClient

from agent_fakes import fake_agent
from careplan_ai.agents.care_plan.schemas import (
    ActionOut,
    DraftPlanOut,
    NoteClassOut,
    RiskFlagOut,
)
from careplan_ai.api.app import create_app

pytestmark = pytest.mark.integration

KEY = {"X-Api-Key": "test-api-key"}
AGENT = {**KEY, "X-CarePlan-Version": "2"}
STOBHILL, CARDONALD = 4, 2


def _plan(care_plan_id: int, pathway: str, action: str, **extra: object) -> dict:  # type: ignore[type-arg]
    return {"carePlanId": care_plan_id, "pathway": pathway, "action": action, **extra}


def _index(client: TestClient, home: int, *plans: dict) -> dict:  # type: ignore[type-arg]
    response = client.post(f"/library/homes/{home}/care-plans", json={"items": list(plans)},
                           headers=KEY)
    assert response.status_code == 200, response.text
    return response.json()


@pytest.fixture
def library_env(base_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CAREPLAN_AI_LIBRARY_PROVIDER", "memory")


PLAN = DraftPlanOut(
    risks=[RiskFlagOut(evidence_id="A1", reason="Has dentures")],
    note_classes=[NoteClassOut(evidence_id="R1", kind="status")],
    not_used=[],
    actions=[ActionOut(text="Staff should help Cathy look after her dentures.",
                       evidence_ids=["A1"], outcome_text="", outcome_evidence_ids=[])],
)


@pytest.fixture
def drafts(monkeypatch: pytest.MonkeyPatch) -> list[dict]:  # type: ignore[type-arg]
    """The agent submits PLAN; each turn logs the style examples it was shown."""
    return fake_agent(monkeypatch, lambda role, messages: PLAN)


def _start(client: TestClient, home: int, pathway: str) -> dict:  # type: ignore[type-arg]
    response = client.post("/care-plans/sessions", headers=AGENT, json={
        "assessmentTemplateUserId": 1, "pathway": pathway, "organisationId": home,
        "residentName": "Cathy",
        "assessment": {"answers": [{"question": "Dentures", "answer": "Yes"}]},
        "reviewNotes": [{"note": "Cathy is settled."}],
    })
    assert response.status_code == 201, response.text
    return response.json()


def test_libraries_are_separate_per_home_and_per_pathway(
    library_env: None, drafts: list[dict]  # type: ignore[type-arg]
) -> None:
    with TestClient(create_app()) as client:
        _index(client, STOBHILL,
               _plan(1, "Oral Health Assessment", "Prompt [Resident] to clean dentures."),
               _plan(2, "MUST", "Weigh [Resident] weekly."))
        _index(client, CARDONALD,
               _plan(3, "Oral Health Assessment", "Cardonald oral care wording."))
        oral = _start(client, STOBHILL, "Oral Health Assessment")
        must = _start(client, STOBHILL, "must")  # pathway tags match case-insensitively

    assert oral["generation"]["exampleIds"] == ["cp-1"]  # never Cardonald's cp-3
    assert must["generation"]["exampleIds"] == ["cp-2"]
    assert drafts[0]["examples"] == ["Action:\nPrompt [Resident] to clean dentures."]


def test_a_care_plan_saved_now_is_an_example_for_the_next_draft(
    library_env: None, drafts: list[dict]  # type: ignore[type-arg]
) -> None:
    with TestClient(create_app()) as client:
        before = _start(client, STOBHILL, "Oral Health Assessment")
        _index(client, STOBHILL, _plan(10, "Oral Health Assessment", "New wording."))
        after = _start(client, STOBHILL, "Oral Health Assessment")

    assert before["generation"]["exampleIds"] == []  # empty library -> neutral style
    assert after["generation"]["exampleIds"] == ["cp-10"]


def test_identifiers_are_replaced_before_indexing(
    library_env: None, drafts: list[dict]  # type: ignore[type-arg]
) -> None:
    with TestClient(create_app()) as client:
        _index(client, STOBHILL, _plan(
            20, "Oral Health Assessment", "<p>Cathy in Room 12 needs prompting.</p>",
            residentNames=["Cathy Brown"],
        ))
        _start(client, STOBHILL, "Oral Health Assessment")

    assert drafts[0]["examples"] == ["Action:\n[Resident] in [Room] needs prompting."]


def test_excluded_care_plans_are_not_retrieved_and_stay_excluded_when_re_saved(
    library_env: None, drafts: list[dict]  # type: ignore[type-arg]
) -> None:
    with TestClient(create_app()) as client:
        _index(client, STOBHILL, _plan(30, "Oral Health Assessment", "Wording."))
        excluded = client.post(f"/library/homes/{STOBHILL}/care-plans/30/exclusion",
                               json={"excluded": True}, headers=KEY)
        resaved = _index(client, STOBHILL, _plan(30, "Oral Health Assessment", "Edited."))
        session = _start(client, STOBHILL, "Oral Health Assessment")
        missing = client.post(f"/library/homes/{STOBHILL}/care-plans/999/exclusion",
                              json={"excluded": True}, headers=KEY)

    assert excluded.status_code == 200
    assert resaved["excluded"] == ["cp-30"]
    assert session["generation"]["exampleIds"] == []
    assert missing.status_code == 404


def test_archived_and_deleted_care_plans_are_not_retrieved(
    library_env: None, drafts: list[dict]  # type: ignore[type-arg]
) -> None:
    with TestClient(create_app()) as client:
        _index(client, STOBHILL,
               _plan(40, "Oral Health Assessment", "Archived.", status="archived"),
               _plan(41, "Oral Health Assessment", "Deleted."))
        client.delete(f"/library/homes/{STOBHILL}/care-plans/41", headers=KEY)
        session = _start(client, STOBHILL, "Oral Health Assessment")

    assert session["generation"]["exampleIds"] == []


def test_staff_written_examples_rank_above_unedited_ai(
    library_env: None, drafts: list[dict], monkeypatch: pytest.MonkeyPatch  # type: ignore[type-arg]
) -> None:
    monkeypatch.setenv("CAREPLAN_AI_LIBRARY_TOP_K", "2")
    with TestClient(create_app()) as client:
        _index(client, STOBHILL,
               _plan(50, "Oral Health Assessment", "Dentures dentures.", source="ai_assisted"),
               _plan(51, "Oral Health Assessment", "Dentures.", source="ai_assisted",
                     edited=True),
               _plan(52, "Oral Health Assessment", "Oral care."))
        session = _start(client, STOBHILL, "Oral Health Assessment")

    assert session["generation"]["exampleIds"] == ["cp-52", "cp-51"]


def test_ai_assisted_plans_are_recognised_from_their_trace(
    library_env: None, drafts: list[dict]  # type: ignore[type-arg]
) -> None:
    from careplan_ai.library.store import get_vector_store

    with TestClient(create_app()) as client:
        session = _start(client, STOBHILL, "Oral Health Assessment")
        sid, url = session["sessionId"], f"/care-plans/sessions/{session['sessionId']}/cards"
        for card in session["cards"]:
            client.post(f"{url}/{card['id']}/decision", json={"decision": "add"}, headers=AGENT)
        ai_text = " ".join(c["text"] for c in session["cards"])
        # Java's save hook indexes the plan first, then the UI finalizes it.
        _index(client, STOBHILL, _plan(70, "Oral Health Assessment", f"<p>{ai_text}</p>"))
        client.post(f"/care-plans/sessions/{sid}/finalize", json={"carePlanId": 70},
                    headers=AGENT)
        # Later staff add their own wording and the plan is re-saved.
        _index(client, STOBHILL, _plan(
            70, "Oral Health Assessment",
            f"<p>{ai_text} Cathy likes to brush her teeth after breakfast in her room.</p>",
        ))
        import asyncio
        store = get_vector_store()
        assert store is not None
        after_finalize_then_edit = asyncio.run(store.get(STOBHILL, ["cp-70"]))["cp-70"]

    assert after_finalize_then_edit.source == "ai_assisted"
    assert after_finalize_then_edit.edited is True


def test_staff_changed_detects_rewording_and_added_text() -> None:
    from careplan_ai.library.service import staff_changed

    ai = ["Staff should help Cathy with her dentures."]

    assert staff_changed("<p>Staff should help Cathy with her dentures.</p>", ai) is False
    assert staff_changed("Staff should support Cathy with her dentures.", ai) is True
    assert staff_changed("Staff should help Cathy with her dentures. She prefers mint "
                         "toothpaste and a soft brush.", ai) is True


def test_the_home_name_is_stored_on_each_record_but_the_library_is_keyed_by_id(
    library_env: None,
) -> None:
    import asyncio

    from careplan_ai.library.store import get_vector_store, namespace_for

    with TestClient(create_app()) as client:
        response = client.post(f"/library/homes/{STOBHILL}/care-plans", headers=KEY, json={
            "homeName": "  Stobhill  ", "items": [_plan(95, "MUST", "Weigh [Resident] weekly.")],
        })
        store = get_vector_store()
        assert store is not None
        entry = asyncio.run(store.get(STOBHILL, ["cp-95"]))["cp-95"]

    assert response.status_code == 200
    assert entry.home_name == "Stobhill"
    assert namespace_for(STOBHILL) == "home-4"


def test_library_switched_off_is_a_no_op(base_env: None) -> None:
    with TestClient(create_app()) as client:
        result = _index(client, STOBHILL, _plan(60, "MUST", "Wording."))

    assert result == {"enabled": False, "indexed": [], "excluded": []}


def test_library_endpoints_require_api_key(library_env: None) -> None:
    with TestClient(create_app()) as client:
        response = client.post(f"/library/homes/{STOBHILL}/care-plans",
                               json={"items": [_plan(1, "MUST", "x")]})

    assert response.status_code == 401

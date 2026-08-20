import pytest
from fastapi.testclient import TestClient

from careplan_ai.api.app import create_app
from careplan_ai.domain.care_plan import CarePlanDraft

pytestmark = pytest.mark.integration


@pytest.fixture
def fake_llm_draft(monkeypatch: pytest.MonkeyPatch) -> CarePlanDraft:
    draft = CarePlanDraft(
        action="Support resident with a supervised mobility exercise plan twice daily.",
        outcome="Reduced fall frequency and improved confidence with walking aids.",
        care_plan_title="Falls Prevention",
        rationale="Based on two recent falls reported in the assessment.",
    )

    async def _fake_generate(payload: object) -> CarePlanDraft:
        return draft

    monkeypatch.setattr(
        "careplan_ai.api.routers.care_plan.generate_care_plan_draft",
        _fake_generate,
    )
    return draft


def test_generate_care_plan_returns_draft(base_env: None, fake_llm_draft: CarePlanDraft) -> None:
    payload = {
        "assessmentTemplateUserId": 42,
        "assessmentTemplateId": 7,
        "assessmentType": "Falls Risk",
        "assessmentResponses": [
            {
                "question": "Has the resident fallen recently?",
                "answer": "Yes, twice in the last month",
            }
        ],
    }

    with TestClient(create_app()) as client:
        response = client.post(
            "/v1/care-plans/generate",
            json=payload,
            headers={"X-Api-Key": "test-api-key"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["action"] == fake_llm_draft.action
    assert body["assessmentTemplateUserId"] == 42
    assert body["assessmentTemplateId"] == 7
    assert body["carePlanTitle"] == "Falls Prevention"


def test_generate_care_plan_requires_api_key(base_env: None) -> None:
    payload = {"assessmentTemplateUserId": 42, "assessmentResponses": []}

    with TestClient(create_app()) as client:
        response = client.post("/v1/care-plans/generate", json=payload)

    assert response.status_code == 401


def test_generate_care_plan_rejects_wrong_api_key(base_env: None) -> None:
    payload = {"assessmentTemplateUserId": 42, "assessmentResponses": []}

    with TestClient(create_app()) as client:
        response = client.post(
            "/v1/care-plans/generate",
            json=payload,
            headers={"X-Api-Key": "wrong-key"},
        )

    assert response.status_code == 401


def test_generate_care_plan_rejects_missing_required_field(base_env: None) -> None:
    with TestClient(create_app()) as client:
        response = client.post(
            "/v1/care-plans/generate",
            json={},
            headers={"X-Api-Key": "test-api-key"},
        )

    assert response.status_code == 422


def test_generate_care_plan_is_rate_limited_per_api_key(
    base_env: None, fake_llm_draft: CarePlanDraft, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CAREPLAN_AI_RATE_LIMIT_REQUESTS", "2")
    monkeypatch.setenv("CAREPLAN_AI_RATE_LIMIT_WINDOW_SECONDS", "60")
    payload = {
        "assessmentTemplateUserId": 42,
        "assessmentResponses": [{"question": "Q", "answer": "A"}],
    }
    headers = {"X-Api-Key": "test-api-key"}

    with TestClient(create_app()) as client:
        first = client.post("/v1/care-plans/generate", json=payload, headers=headers)
        second = client.post("/v1/care-plans/generate", json=payload, headers=headers)
        third = client.post("/v1/care-plans/generate", json=payload, headers=headers)

    assert first.status_code == 200
    assert second.status_code == 200
    assert third.status_code == 429
    assert third.json()["error_code"] == "rate_limit_exceeded"
    assert "Retry-After" in third.headers

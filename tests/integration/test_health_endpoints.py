import pytest
from fastapi.testclient import TestClient

from careplan_ai.api.app import create_app

pytestmark = pytest.mark.integration


def test_liveness_returns_ok(base_env: None) -> None:
    with TestClient(create_app()) as client:
        response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readiness_returns_ready(base_env: None) -> None:
    with TestClient(create_app()) as client:
        response = client.get("/readyz")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch


MOCK_THRESHOLD = 0.35


@pytest.fixture
def client():
    fake_store = {
        "model": MagicMock(predict_proba=lambda X: np.tile(np.array([[0.15, 0.85]]), (len(X), 1))),
        "amount_scaler": MagicMock(transform=lambda X: np.asarray(X)),
        "time_scaler": MagicMock(transform=lambda X: np.asarray(X)),
        "threshold": MOCK_THRESHOLD,
        "model_loaded": True,
        "request_count": 0,
        "fraud_count": 0,
    }
    with patch("app.model_store", fake_store), patch("app.load_artifacts"):
        from app import app

        with TestClient(app) as test_client:
            yield test_client


def valid_payload():
    return {"transaction_id": "test_tx_001", "features": [0.0] * 30}


def test_root_returns_service_info(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["service"] == "Fraud Detection API"


def test_health_returns_loaded_model(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_model_info_returns_metadata(client):
    response = client.get("/model-info")
    assert response.status_code == 200
    assert response.json()["features"] == 30
    assert len(response.json()["feature_order"]) == 30


def test_predict_returns_additive_business_fields(client):
    response = client.post("/predict", json=valid_payload())
    assert response.status_code == 200
    data = response.json()
    assert data["transaction_id"] == "test_tx_001"
    assert data["prediction"] == "fraud"
    assert data["is_fraud"] is True
    assert data["fraud_probability"] == 0.85
    assert data["risk_level"] == "high"
    assert data["recommended_action"] == "hold_or_manual_review"
    assert data["decision_threshold"] == MOCK_THRESHOLD
    assert data["request_id"]


def test_versioned_predict_alias_works(client):
    response = client.post("/api/v1/predict", json=valid_payload())
    assert response.status_code == 200
    assert response.json()["is_fraud"] is True


def test_batch_predict_works(client):
    response = client.post("/predict-batch", json={"transactions": [valid_payload(), valid_payload()]})
    assert response.status_code == 200
    assert response.json()["count"] == 2
    assert len(response.json()["predictions"]) == 2


@pytest.mark.parametrize(
    "payload",
    [
        {"transaction_id": "bad", "features": [0.0] * 29},
        {"transaction_id": "bad", "features": [0.0] * 31},
        {"transaction_id": "bad", "features": ["abc"] * 30},
        {"transaction_id": "bad", "features": [0.0] * 29 + [-1.0]},
        {"features": [0.0] * 30},
    ],
)
def test_invalid_transactions_return_422(client, payload):
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_request_id_is_echoed(client):
    response = client.get("/", headers={"X-Request-ID": "known-request-id"})
    assert response.headers["X-Request-ID"] == "known-request-id"


def test_model_unavailable_returns_controlled_503():
    from app import app

    with patch("app.load_artifacts"), patch("app.model_store", {"model_loaded": False, "threshold": 0.5}):
        with TestClient(app) as unavailable_client:
            response = unavailable_client.post("/predict", json=valid_payload())
    assert response.status_code == 503
    assert response.json()["detail"]["error"] == "model_unavailable"


def test_metrics_endpoint_is_available(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "fraud_api_requests_total" in response.text


def test_api_key_is_enforced_when_configured(client):
    with patch("app.API_KEY", "secret-key"):
        unauthenticated = client.get("/model-info")
        authenticated = client.get("/model-info", headers={"X-API-Key": "secret-key"})
    assert unauthenticated.status_code == 401
    assert authenticated.status_code == 200

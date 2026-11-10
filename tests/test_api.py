from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from api import main as api
from src.train import train_model

ROOT = Path(__file__).parents[1]
SAMPLE = ROOT / "data" / "sample" / "sample_transactions.csv"


def test_health_and_prediction_endpoints(tmp_path, monkeypatch):
    model_path = tmp_path / "fraud_model.pkl"
    train_model(SAMPLE, model_path)
    monkeypatch.setattr(api, "MODEL_PATH", model_path)
    api.app.state.model = None
    client = TestClient(api.app)

    assert client.get("/health").json() == {"status": "ok"}
    transaction = pd.read_csv(SAMPLE).drop(columns="is_fraud").iloc[0]
    payload = transaction.to_dict()
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert set(response.json()) == {"transaction_id", "is_fraud", "fraud_probability"}
    assert response.json()["transaction_id"] == payload["transaction_id"]
    assert 0 <= response.json()["fraud_probability"] <= 1

    batch = pd.read_csv(SAMPLE).drop(columns="is_fraud").iloc[:2].to_dict(orient="records")
    batch_response = client.post("/predict/batch", json=batch)
    assert batch_response.status_code == 200
    assert [item["transaction_id"] for item in batch_response.json()] == [
        item["transaction_id"] for item in batch
    ]
    duplicate_response = client.post("/predict/batch", json=[batch[0], batch[0]])
    assert duplicate_response.status_code == 422
    assert "fraud_predictions_total" in client.get("/metrics").text


def test_prediction_without_model_returns_service_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "MODEL_PATH", tmp_path / "missing.pkl")
    api.app.state.model = None
    response = TestClient(api.app).post(
        "/predict",
        json={
            "transaction_id": "TXN-TEST-001",
            "amount": 10,
            "transaction_hour": 12,
            "merchant_category": "grocery",
            "card_present": True,
            "device_trusted": True,
            "customer_age_days": 100,
            "transactions_last_24h": 1,
        },
    )
    assert response.status_code == 503

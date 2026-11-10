"""FastAPI application for single and batch fraud scoring."""

import os
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from src.monitoring import FRAUD_PROBABILITY, PREDICTIONS
from src.predict import load_model, predict_transactions

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.getenv("FRAUD_MODEL_PATH", PROJECT_ROOT / "models/fraud_model.pkl"))
app = FastAPI(title="Fraud Detection API", version="0.1.0")
app.state.model = None


class Transaction(BaseModel):
    transaction_id: str = Field(min_length=1, max_length=128)
    amount: float = Field(gt=0)
    transaction_hour: int = Field(ge=0, le=23)
    merchant_category: str = Field(min_length=1)
    card_present: bool
    device_trusted: bool
    customer_age_days: int = Field(ge=0)
    transactions_last_24h: int = Field(ge=0)


class Prediction(BaseModel):
    transaction_id: str
    is_fraud: bool
    fraud_probability: float = Field(ge=0, le=1)


def _get_model():
    if app.state.model is None:
        try:
            app.state.model = load_model(MODEL_PATH)
        except FileNotFoundError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
    return app.state.model


def _score(transactions: list[Transaction]) -> list[Prediction]:
    model = _get_model()
    frame = pd.DataFrame([transaction.model_dump() for transaction in transactions])
    try:
        results = predict_transactions(model, frame)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    for result in results:
        label = str(int(result["is_fraud"]))
        PREDICTIONS.labels(prediction=label).inc()
        FRAUD_PROBABILITY.observe(result["fraud_probability"])
    return [
        Prediction(transaction_id=transaction.transaction_id, **result)
        for transaction, result in zip(transactions, results, strict=True)
    ]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(transaction: Transaction) -> Prediction:
    return _score([transaction])[0]


@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(transactions: list[Transaction]) -> list[Prediction]:
    if not transactions:
        raise HTTPException(status_code=422, detail="At least one transaction is required")
    if len(transactions) > 1000:
        raise HTTPException(status_code=413, detail="Batch size cannot exceed 1000")
    transaction_ids = [transaction.transaction_id for transaction in transactions]
    if len(transaction_ids) != len(set(transaction_ids)):
        raise HTTPException(status_code=422, detail="transaction_id values must be unique in a batch")
    return _score(transactions)


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

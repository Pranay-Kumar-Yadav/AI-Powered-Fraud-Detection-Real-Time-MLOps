"""Load a persisted model and score incoming transaction records."""

from pathlib import Path

import joblib
import pandas as pd

from src.feature_engineering import prepare_features


def load_model(model_path: str | Path):
    """Load a model pipeline saved by ``src.train``."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {path}. Train a model first.")
    return joblib.load(path)


def predict_transactions(model, transactions: pd.DataFrame) -> list[dict]:
    """Return fraud labels and probabilities for one or more transactions."""
    features = prepare_features(transactions)
    labels = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]
    return [
        {"is_fraud": bool(label), "fraud_probability": float(probability)}
        for label, probability in zip(labels, probabilities, strict=True)
    ]

from pathlib import Path

import pandas as pd

from src.predict import load_model, predict_transactions
from src.train import train_model

ROOT = Path(__file__).parents[1]
SAMPLE = ROOT / "data" / "sample" / "sample_transactions.csv"


def test_training_saves_a_predictive_model(tmp_path):
    model_path = tmp_path / "fraud_model.pkl"
    metrics = train_model(SAMPLE, model_path)

    assert model_path.is_file()
    assert 0 <= metrics["average_precision"] <= 1
    assert 0 <= metrics["roc_auc"] <= 1
    model = load_model(model_path)
    row = pd.read_csv(SAMPLE).drop(columns="is_fraud").iloc[[0]]
    result = predict_transactions(model, row)
    assert len(result) == 1
    assert isinstance(result[0]["is_fraud"], bool)
    assert 0 <= result[0]["fraud_probability"] <= 1

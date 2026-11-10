"""Evaluate a saved model on a labeled transaction CSV."""

import argparse
import json
from pathlib import Path

from sklearn.metrics import average_precision_score, classification_report, roc_auc_score

from src.data_preprocessing import load_transactions
from src.feature_engineering import split_features_target
from src.predict import load_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def evaluate_model(model_path: str | Path, data_path: str | Path) -> dict:
    transactions = load_transactions(data_path)
    features, target = split_features_target(transactions)
    model = load_model(model_path)
    probabilities = model.predict_proba(features)[:, 1]
    predictions = model.predict(features)
    return {
        "average_precision": float(average_precision_score(target, probabilities)),
        "roc_auc": float(roc_auc_score(target, probabilities)),
        "classification_report": classification_report(
            target, predictions, output_dict=True, zero_division=0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(PROJECT_ROOT / "data/sample/sample_transactions.csv"))
    parser.add_argument("--model", default=str(PROJECT_ROOT / "models/fraud_model.pkl"))
    args = parser.parse_args()
    print(json.dumps(evaluate_model(args.model, args.data), indent=2))


if __name__ == "__main__":
    main()

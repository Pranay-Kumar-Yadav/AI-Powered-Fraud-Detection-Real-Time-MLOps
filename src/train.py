"""Train and persist the fraud classification pipeline."""

import argparse
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data_preprocessing import load_transactions
from src.feature_engineering import build_preprocessor, split_features_target

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def train_model(
    data_path: str | Path,
    model_path: str | Path,
    test_size: float = 0.25,
    random_state: int = 42,
) -> dict[str, float]:
    """Fit on a stratified split, report holdout metrics, and save the pipeline."""
    transactions = load_transactions(data_path)
    features, target = split_features_target(transactions)
    if target.nunique() < 2 or target.value_counts().min() < 2:
        raise ValueError("Training requires at least two examples of each target class")

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )
    model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000)),
        ]
    )
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]
    metrics = {
        "average_precision": float(average_precision_score(y_test, probabilities)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "test_rows": int(len(y_test)),
    }

    output_path = Path(model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    return metrics


def _load_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def main() -> None:
    config = _load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(PROJECT_ROOT / config["data_path"]))
    parser.add_argument("--model", default=str(PROJECT_ROOT / config["model_path"]))
    args = parser.parse_args()
    metrics = train_model(args.data, args.model)
    print(pd.Series(metrics).to_string())
    print(f"Model saved to {args.model}")


if __name__ == "__main__":
    main()

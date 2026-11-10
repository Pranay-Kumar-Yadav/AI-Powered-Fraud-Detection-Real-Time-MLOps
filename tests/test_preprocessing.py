from pathlib import Path

import pytest

from src.data_preprocessing import load_transactions
from src.feature_engineering import split_features_target

DATA_PATH = Path(__file__).parents[1] / "data" / "sample" / "sample_transactions.csv"


def test_load_transactions_validates_sample_data():
    frame = load_transactions(DATA_PATH)
    features, target = split_features_target(frame)

    assert len(frame) == 24
    assert "is_fraud" not in features.columns
    assert "transaction_id" not in features.columns
    assert set(target.unique()) == {0, 1}


def test_load_transactions_rejects_missing_columns(tmp_path):
    malformed = tmp_path / "bad.csv"
    malformed.write_text("transaction_id,is_fraud\nTXN1,0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Missing required columns"):
        load_transactions(malformed)

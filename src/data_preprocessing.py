"""Load and validate transaction data before feature generation."""

from pathlib import Path

import pandas as pd

TARGET_COLUMN = "is_fraud"
ID_COLUMN = "transaction_id"
REQUIRED_COLUMNS = {
    ID_COLUMN,
    "amount",
    "transaction_hour",
    "merchant_category",
    "card_present",
    "device_trusted",
    "customer_age_days",
    "transactions_last_24h",
    TARGET_COLUMN,
}


def load_transactions(path: str | Path) -> pd.DataFrame:
    """Read a transaction CSV and reject missing or malformed required fields."""
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    if frame.empty:
        raise ValueError("Transaction dataset is empty")
    if frame[TARGET_COLUMN].isna().any() or not set(frame[TARGET_COLUMN].unique()) <= {0, 1}:
        raise ValueError("is_fraud must contain only non-null binary values 0 or 1")
    if frame[ID_COLUMN].isna().any():
        raise ValueError("transaction_id must not contain null values")
    return frame

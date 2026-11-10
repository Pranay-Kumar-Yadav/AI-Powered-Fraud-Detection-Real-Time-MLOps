"""Transform raw transactions into model inputs without target leakage."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_preprocessing import TARGET_COLUMN

NUMERIC_FEATURES = [
    "amount",
    "transaction_hour",
    "customer_age_days",
    "transactions_last_24h",
]
BOOLEAN_FEATURES = ["card_present", "device_trusted"]
CATEGORICAL_FEATURES = ["merchant_category"]
FEATURE_COLUMNS = NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Select model inputs and normalize boolean values from CSV/API payloads."""
    missing = set(FEATURE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {', '.join(sorted(missing))}")
    features = frame[FEATURE_COLUMNS].copy()
    for column in BOOLEAN_FEATURES:
        features[column] = features[column].map(
            {True: 1, False: 0, "true": 1, "false": 0, "True": 1, "False": 0}
        )
        if features[column].isna().any():
            raise ValueError(f"{column} must contain boolean values")
    return features


def split_features_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate model inputs and labels, excluding identifiers and target."""
    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")
    return prepare_features(frame), frame[TARGET_COLUMN].astype(int)


def build_preprocessor() -> ColumnTransformer:
    """Build preprocessing fitted only on the training split."""
    numeric = Pipeline([("scale", StandardScaler())])
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, NUMERIC_FEATURES),
            ("boolean", "passthrough", BOOLEAN_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

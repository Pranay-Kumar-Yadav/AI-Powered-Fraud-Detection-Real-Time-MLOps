"""Prometheus metrics shared by API inference handlers."""

from prometheus_client import Counter, Histogram

PREDICTIONS = Counter(
    "fraud_predictions_total",
    "Number of scored transactions, labeled by predicted class.",
    ["prediction"],
)
FRAUD_PROBABILITY = Histogram(
    "fraud_probability",
    "Distribution of model fraud probabilities.",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0),
)

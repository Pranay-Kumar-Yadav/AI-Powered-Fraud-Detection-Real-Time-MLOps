"""Publish CSV transactions to a Kafka topic for streaming experiments."""

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from kafka import KafkaProducer

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def publish_transactions(
    csv_path: str | Path,
    bootstrap_servers: str,
    topic: str = "transactions",
) -> int:
    frame = pd.read_csv(csv_path)
    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers.split(","),
        value_serializer=lambda value: json.dumps(value).encode("utf-8"),
        acks="all",
    )
    try:
        for record in frame.to_dict(orient="records"):
            producer.send(topic, record)
        producer.flush()
    finally:
        producer.close()
    return len(frame)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv", default=str(PROJECT_ROOT / "data/sample/sample_transactions.csv")
    )
    parser.add_argument("--bootstrap-servers", default=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"))
    parser.add_argument("--topic", default="transactions")
    args = parser.parse_args()
    count = publish_transactions(args.csv, args.bootstrap_servers, args.topic)
    print(f"Published {count} transactions to {args.topic}")


if __name__ == "__main__":
    main()

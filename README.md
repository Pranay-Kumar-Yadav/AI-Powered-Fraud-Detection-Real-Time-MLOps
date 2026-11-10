# Fraud Detection MLOps

A compact example of a fraud-classification workflow: validate transaction CSVs, train a scikit-learn pipeline, serve predictions over FastAPI, and expose inference metrics to Prometheus. The bundled data is synthetic and exists only to exercise the code path; it is not suitable for evaluating or deploying a real fraud system.

## Quick Start

Run these commands from this project directory in PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m src.train
python -m pytest
```

Training writes `models/fraud_model.pkl` and reports holdout average precision and ROC AUC. The model pipeline includes scaling and category encoding learned on the training split. `transaction_id` and `is_fraud` are excluded from model inputs.

## Run The API

```powershell
uvicorn api.main:app --reload
```

The API is available at `http://127.0.0.1:8000`; interactive schema docs are at `/docs`. `GET /health` does not require a trained artifact. `POST /predict` scores one transaction, `POST /predict/batch` scores up to 1,000, and `GET /metrics` exposes Prometheus metrics. Prediction endpoints return HTTP 503 until a model has been trained.

Send this JSON request body to `POST http://127.0.0.1:8000/predict`:

```json
{
	"transaction_id": "TXN-20261001-0001",
	"amount": 1250.0,
	"transaction_hour": 2,
	"merchant_category": "electronics",
	"card_present": false,
	"device_trusted": false,
	"customer_age_days": 34,
	"transactions_last_24h": 7
}
```

PowerShell example:

```powershell
$body = @{
	transaction_id = 'TXN-20261001-0001'
	amount = 1250.0
	transaction_hour = 2
	merchant_category = 'electronics'
	card_present = $false
	device_trusted = $false
	customer_age_days = 34
	transactions_last_24h = 7
} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict -ContentType 'application/json' -Body $body
```

The response contains the model's binary decision and estimated fraud probability:

```json
{
	"transaction_id": "TXN-20261001-0001",
	"is_fraud": false,
	"fraud_probability": 0.0831
}
```

`transaction_id` is echoed back so you can match a result to its transaction, including in batch responses. Use a unique ID for every transaction in a batch. `is_fraud` is the model's decision (the current classifier uses a 0.5 probability threshold). `fraud_probability` is the estimated chance of fraud, not confirmation that the prediction is correct. A transaction can only be verified after its actual outcome is known. For example, the sample API request returned `is_fraud: false` with `fraud_probability: 0.08309369684961744`.

For batch scoring, send a JSON array to `POST /predict/batch`; each result includes the matching ID:

```json
[
	{
		"transaction_id": "TXN-1001",
		"amount": 1250.0,
		"transaction_hour": 2,
		"merchant_category": "electronics",
		"card_present": false,
		"device_trusted": false,
		"customer_age_days": 34,
		"transactions_last_24h": 7
	},
	{
		"transaction_id": "TXN-1002",
		"amount": 42.5,
		"transaction_hour": 9,
		"merchant_category": "grocery",
		"card_present": true,
		"device_trusted": true,
		"customer_age_days": 820,
		"transactions_last_24h": 2
	}
]
```

## Example Training Scores

Training the bundled synthetic CSV currently prints scores like these:

```text
average_precision    1.0
roc_auc              1.0
test_rows            6
```

These scores are only a workflow example: the dataset is tiny and synthetic, so perfect results do not indicate real-world fraud detection quality. With your own data, keep a labeled holdout CSV that was not used for training and evaluate it with `python -m src.evaluate --data path/to/holdout.csv --model models/fraud_model.pkl`. The evaluation report includes precision and recall, which help show false alarms and missed fraud cases.

## Containers And Monitoring

Train the model before starting Compose so the API can load the artifact:

```powershell
python -m src.train
docker compose up --build
```

Compose starts the API, Prometheus, and Grafana. Prometheus scrapes `api:8000/metrics`; Grafana is provisioned with that Prometheus data source. This starter does not include a Grafana dashboard, Kafka broker, or MLflow server. `kafka/producer.py` can publish the sample CSV to an existing Kafka broker, for example with `python kafka/producer.py --bootstrap-servers localhost:9092`.

## Notebooks

Open and run `notebooks/01_eda.ipynb`, then `02_feature_engineering.ipynb` and `03_model_training.ipynb`. The notebooks use the bundled sample data. Run the model training cell before starting the API.

## Data Contract

Required CSV fields are `transaction_id`, `amount`, `transaction_hour`, `merchant_category`, `card_present`, `device_trusted`, `customer_age_days`, `transactions_last_24h`, and binary `is_fraud`. API requests require a unique `transaction_id` and omit `is_fraud`; the ID is returned with each prediction. Replace both bundled CSVs with appropriately governed data before adapting this example; use time-aware and leakage-reviewed evaluation for real transaction streams.

## Limitations

The small synthetic dataset and baseline logistic regression are instructional scaffolding, not a production fraud detector. Before real use, add representative historical data, time-based validation, threshold/cost analysis, privacy and security controls, drift monitoring, model/version governance, and operational alerting. Prometheus metrics here cover prediction counts and probability distributions; they do not measure predictive quality without labels.

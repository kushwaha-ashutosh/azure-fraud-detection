# Real-Time Fraud Detection Pipeline on Azure

![CI/CD](https://github.com/kushwaha-ashutosh/azure-fraud-detection/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![Azure](https://img.shields.io/badge/cloud-Azure-0078D4)
![XGBoost](https://img.shields.io/badge/ML-XGBoost-orange)
![Tests](https://img.shields.io/badge/tests-31%20passing-brightgreen)

A production-grade, end-to-end real-time fraud detection system built on **Microsoft Azure**. The pipeline ingests raw payment transactions, enriches them with historical features, scores them using a trained XGBoost model, and triggers alerts — all within **~3 seconds**.

---

## Architecture

```
Transaction Simulator
        │
        ▼
Azure Service Bus (raw-transactions queue)
        │
        ▼
Azure Functions – Stream Processor
        │ ├── Enrich from Cosmos DB (feature cache)
        │ ├── Call ML Inference Function (HTTP)
        │ └── Write scored JSON to Blob Storage
        │
        ▼
Azure Service Bus (scored-transactions queue)
        │
        ▼
Azure Functions – Real-time Action
        └── Trigger alert if fraud_score > 0.5

Cold Path (daily):
Blob Storage → ETL Job → Synapse Analytics → Feature Refresh → Cosmos DB
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Messaging | Azure Service Bus (Basic) |
| Stream processing | Azure Functions (Python) |
| Feature cache | Azure Cosmos DB (free tier) |
| ML inference | XGBoost + scikit-learn, served via Azure Functions HTTP |
| Data lake | Azure Blob Storage (ADLS Gen2) |
| Batch ETL | Python ETL job + Azure Synapse Spark |
| Orchestration | Azure Data Factory |
| Monitoring | Azure Application Insights |
| CI/CD | GitHub Actions → Azure Functions deploy |

---

## ML Model Performance

Trained on the [Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (284,807 transactions, 0.17% fraud rate).

| Metric | Value |
|--------|-------|
| AUC-ROC | 0.9733 |
| Average Precision | 0.8741 |
| Fraud Recall | 83% |
| False Positives | 9 / 56,864 |
| Optimal Threshold | tuned via Optuna (20 trials) |

---

## Project Structure

```
azure-fraud-detection/
├── simulator/              # Transaction data generator
│   ├── schema.py           # Transaction dataclass
│   ├── generator.py        # Synthetic data generation
│   ├── main.py             # CLI entry point
│   └── servicebus_sender.py # Azure Service Bus publisher
├── functions/
│   ├── ml_inference/       # HTTP-triggered ML scoring Function
│   ├── stream_processor/   # Service Bus-triggered pipeline Function
│   └── realtime_action/    # Service Bus-triggered alert Function
├── ml_training/
│   └── fraud_detection/    # XGBoost training pipeline + Optuna
├── batch_etl/
│   └── etl_job.py          # Blob → transform → output ETL
├── orchestration/
│   └── feature_refresh.py  # ETL output → Cosmos DB feature cache
├── notebooks/
│   └── model_evaluation.ipynb  # ROC, PR curve, feature importance
├── tests/                  # 31 pytest tests across all modules
└── .github/workflows/      # GitHub Actions CI/CD
```

---

## How to Run Locally

### Prerequisites
- Python 3.10+
- Azure CLI
- Azure Functions Core Tools v4
- An Azure account (free tier sufficient)

### Setup

```bash
git clone https://github.com/kushwaha-ashutosh/azure-fraud-detection
cd azure-fraud-detection
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

Create a `.env` file based on `.env.example` and fill in your Azure connection strings.

### Run the simulator

```bash
# Output to stdout
python -m simulator.main --count 100 --rate 2

# Send to Azure Service Bus
python -m simulator.main --count 100 --rate 2 --output servicebus
```

### Run tests

```bash
pytest tests/ -v
```

### Train the model locally

```bash
# Download dataset from Kaggle first
kaggle datasets download -d mlg-ulb/creditcardfraud -p ml_training/data
cd ml_training/data && tar -xf creditcardfraud.zip && cd ../..

python -m ml_training.fraud_detection.main
```

---

## How to Deploy to Azure

### 1. Create Azure resources

```bash
az group create --name fraud-detection-rg --location centralindia
az servicebus namespace create --name frauddetectionsbus --resource-group fraud-detection-rg --sku Basic
az cosmosdb create --name frauddetectioncosmos --resource-group fraud-detection-rg --enable-free-tier true
az storage account create --name frauddetectionstorage --resource-group fraud-detection-rg --sku Standard_LRS --kind StorageV2
```

### 2. Deploy Functions

```bash
cd functions/ml_inference && func azure functionapp publish fraud-ml-inference --python
cd ../stream_processor && func azure functionapp publish fraud-stream-processor --python
cd ../realtime_action && func azure functionapp publish fraud-realtime-action --python
```

### 3. CI/CD (automatic)

Every push to `master` triggers GitHub Actions which runs all 31 tests and deploys all 3 Functions automatically.

---

## Key Design Decisions

**Why Azure Functions over Dataflow/Beam?** For a consumption-plan serverless pipeline, Functions provide the same trigger-based processing at zero idle cost, which fits the free tier perfectly.

**Why Service Bus over Event Hubs?** Service Bus Basic tier is free for 10M operations/month. Event Hubs has no free tier.

**Why local ML training?** Azure ML compute has no permanent free tier. Training locally and uploading the model artifact to Blob Storage is the standard pattern for cost-conscious MLOps.

**Why XGBoost over deep learning?** Tabular fraud data benefits from gradient boosting. XGBoost achieves AUC 0.97 with fast inference (<200ms), which is critical for real-time scoring.

---

## Improvements & Future Work

- Add Cosmos DB feature enrichment to stream processor (V1-V28 features currently default to 0)
- Add model retraining trigger when fraud rate drifts beyond threshold
- Add A/B model testing via feature flags
- Add Power BI dashboard connected to Synapse for fraud analytics
- Replace Service Bus with Event Hubs for higher throughput scenarios

---

## Author

Ashutosh Kushwaha — [GitHub](https://github.com/kushwaha-ashutosh) | Built as a portfolio project demonstrating Azure data engineering and MLOps.
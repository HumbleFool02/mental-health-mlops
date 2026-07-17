# Mental Health Text Classification — Production MLOps Pipeline

[![Tests](https://github.com/HumbleFool02/mental-health-mlops/actions/workflows/test.yml/badge.svg)](https://github.com/HumbleFool02/mental-health-mlops/actions/workflows/test.yml)
[![Docker](https://github.com/HumbleFool02/mental-health-mlops/actions/workflows/docker-build.yml/badge.svg)](https://github.com/HumbleFool02/mental-health-mlops/actions/workflows/docker-build.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

End-to-end MLOps pipeline for mental health text classification with production drift detection and real-time monitoring. Fine-tunes DistilBERT on 51,842 samples across six categories, deploys via FastAPI on AWS EC2, and monitors for data, concept, and prediction drift in real time.

**Live Demo:** http://18.221.115.135:8000/ui
**Monitoring Dashboard:** http://18.221.115.135:8501
**API Docs:** http://18.221.115.135:8000/docs

---

## Performance

| Metric | Value |
|--------|-------|
| Overall F1 Score | 0.819 (+8.2% over baseline) |
| Suicidal Class Recall | 74.1% |
| Weighted Precision | 0.832 |
| Inference Latency | 98ms |

Baseline comparison: LightGBM achieved 0.771 F1 — DistilBERT chosen for its deployability (40% smaller, 60% faster than BERT, ~97% performance retained) given memory-constrained EC2 deployment.

---

## Drift Detection

Three-layer monitoring stack, implemented from scratch:

**Data Drift** — Population Stability Index (PSI) and Kolmogorov-Smirnov test across four text features (length, word count, avg word length, unique words). Triggers if PSI > 0.1 on two or more features.

**Concept Drift** — Tracks F1, precision, and recall over time. Alerts if any metric degrades more than 5% from baseline.

**Prediction Drift** — Jensen-Shannon divergence and Wasserstein distance on class distribution. Triggers if JS divergence > 0.1.

### Validation Results

| Test | Result |
|------|--------|
| Length drift detection threshold | 1.3x (30% increase) |
| Population drift detection threshold | 40% dominance (5.7x shift from 7% baseline) |
| Slang/linguistic drift | Not detected — documented limitation |
| Data drift false positive rate | 16% (8% overall across all detectors) |
| Prediction drift false positive rate | 0% |

**Documented limitation:** Statistical features (length, word count) are insensitive to vocabulary substitution. "depressed" → "down bad" doesn't change text length enough to trigger detection — motivating the multi-detector design and flagging the need for embedding-based semantic drift detection as future work.

---

## Architecture

```
Raw Text
    │
    ▼
FastAPI (:8000) ──► DistilBERT (6-class)
    │                     │
    │◄────────────────────┘
    │
    ├──► Drift Detectors (PSI / KS / JS / Wasserstein)
    │         │
    │         ▼
    │    SQLite (drift history)
    │         │
    └──► Streamlit Dashboard (:8501)

Infrastructure: Docker on AWS EC2 t3.small (us-east-1)
Model storage:  S3 (downloaded at build time)
CI/CD:          GitHub Actions → GHCR → SSH deploy
```

---

## Quick Start

### Run with Docker

The model downloads from S3 at build time — AWS credentials required:

```bash
git clone https://github.com/HumbleFool02/mental-health-mlops.git
cd mental-health-mlops

docker build -f Dockerfile.lightweight \
  --build-arg AWS_ACCESS_KEY_ID=<your-key> \
  --build-arg AWS_SECRET_ACCESS_KEY=<your-secret> \
  --build-arg AWS_DEFAULT_REGION=us-east-1 \
  -t mental-health-mlops:latest .

docker run -d \
  --name mlops-app \
  -p 8000:8000 \
  -p 8501:8501 \
  mental-health-mlops:latest
```

API: http://localhost:8000 | Dashboard: http://localhost:8501

### Run Locally (without Docker)

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Download model manually from S3
aws s3 sync s3://mental-health-mlops-models/distilbert_exp2/ models/distilbert_exp2/
aws s3 cp s3://mental-health-mlops-models/data/reference_data.csv data/processed/reference_data.csv

# Start API
uvicorn src.api.app:app --host 0.0.0.0 --port 8000

# Start dashboard (separate terminal)
streamlit run src/dashboard/drift_dashboard.py
```

---

## API Usage

```bash
# Predict
curl -X POST "http://18.221.115.135:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "I feel very anxious and cannot stop worrying"}'

# Health check
curl http://18.221.115.135:8000/health
```

```json
{
  "prediction": "Anxiety",
  "confidence": 0.8934,
  "all_probabilities": {
    "Anxiety": 0.8934,
    "Depression": 0.0523,
    "Suicidal": 0.0234,
    "Normal": 0.0156,
    "Bipolar": 0.0089,
    "Stress": 0.0064
  },
  "timestamp": "2026-07-16T10:30:45.123Z"
}
```

---

## Project Structure

```
mental-health-mlops/
├── .github/workflows/
│   ├── test.yml              # pytest — 15/15 passing
│   ├── docker-build.yml      # builds + pushes to GHCR
│   └── deploy.yml            # SSH deploy to EC2 (manual trigger)
├── src/
│   ├── api/                  # FastAPI app, model loader, schemas
│   ├── monitoring/           # Data, concept, prediction drift detectors
│   └── dashboard/            # Streamlit dashboard + SQLite interface
├── experiments/
│   └── drift_analysis/       # Sensitivity, attribution, FP rate scripts
│       └── results/          # JSON output from validation runs
├── tests/                    # 15 tests — API (9) + drift detection (6)
├── configs/config.yaml       # Central config
├── Dockerfile.lightweight    # Production image (both services)
└── requirements.production.txt
```

---

## CI/CD

Three workflows:

- **test.yml** — runs on every push to `main`/`develop`. FastAPI tests use `TestClient` with mocked model loader (no live server needed). Drift detection tests use synthetic fixtures (no data files needed).
- **docker-build.yml** — runs on push to `main`. Builds `Dockerfile.lightweight`, downloads model from S3, pushes image to GitHub Container Registry, smoke tests `/health`.
- **deploy.yml** — manual trigger only. SSH into EC2, `git pull`, rebuild image, swap containers, poll `/health` for 240s.

---

## Running Tests

```bash
pytest tests/ -v
# 15 passed

pytest tests/ --cov=src --cov-report=html
```

---

## Dataset

51,842 samples across six classes, stratified 50/20/30 train/val/test split:

| Class | Samples | % |
|-------|---------|---|
| Normal | 16,351 | 31.5% |
| Depression | 15,404 | 29.8% |
| Suicidal | 10,653 | 20.6% |
| Anxiety | 3,888 | 7.5% |
| Bipolar | 2,877 | 5.5% |
| Stress | 2,669 | 5.1% |

---

## Deployment

- **Instance:** AWS EC2 t3.small, us-east-1, Ubuntu 22.04 LTS
- **Elastic IP:** 18.221.115.135 (static — survives instance stop/start)
- **Model:** Stored in S3, downloaded at Docker build time
- **Container:** Single Docker container running FastAPI (:8000) and Streamlit (:8501)
- **Auto-restart:** `--restart unless-stopped`

---

*Capstone project — Florida Institute of Technology, MS Software Engineering*

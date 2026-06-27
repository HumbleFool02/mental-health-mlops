# 🏥 Mental Health Text Classification - Production MLOps Pipeline

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-brightgreen.svg)](https://www.docker.com/)
[![AWS](https://img.shields.io/badge/AWS-deployed-orange.svg)](https://aws.amazon.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> End-to-end MLOps pipeline for mental health text classification with comprehensive drift detection and real-time monitoring.

**🌐 Live Demo:** http://98.93.153.242:8000
**📊 Monitoring Dashboard:** http://98.93.153.242:8501

---

## 🎯 Project Overview

This project implements a **production-ready MLOps pipeline** for classifying mental health text into six categories (Normal, Depression, Suicidal, Anxiety, Bipolar, Stress). The system emphasizes **drift detection and monitoring** to ensure reliable long-term performance in production environments.

### Key Features

- 🤖 **High-Performance Model:** 81.9% F1 score using DistilBERT transformer
- 🎯 **Critical Case Detection:** 74.1% recall on suicidal cases (minimizing false negatives)
- 📊 **Multi-Faceted Drift Detection:** Data drift (PSI, KS), Concept drift (performance), Prediction drift (JS divergence)
- 🔍 **Comprehensive Validation:** Sensitivity analysis, feature attribution, false positive testing
- 🚀 **Production API:** FastAPI with <100ms latency
- 📈 **Real-Time Dashboard:** Live drift monitoring with Streamlit
- 🐳 **Containerized:** Docker-based deployment
- ☁️ **Cloud Deployed:** Running on AWS EC2
- ✅ **CI/CD Pipeline:** Automated testing with GitHub Actions

---

## 📊 Performance Metrics

### Classification Performance

| Metric | Value | Improvement vs Baseline |
|--------|-------|------------------------|
| **Overall F1 Score** | 0.819 | +8.2% |
| **Suicidal Recall** | 74.1% | +2.8% |
| **Precision** | 0.832 | +7.5% |
| **Inference Time** | <100ms | - |

### Drift Detection Validation

| Test Type | Result | Assessment |
|-----------|--------|------------|
| **Sensitivity (Length)** | 1.3x threshold | Detects 30% length increase |
| **Sensitivity (Population)** | 40% dominance | Detects 5.7x frequency shift |
| **False Positive Rate** | 8.0% overall | Within acceptable range (<10%) |
| **Prediction Drift FP** | 0.0% | Excellent calibration |

---

## 🏗️ Architecture
```
┌─────────────────────────────────────────────────────────┐
│                    Production System                    │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────┐      ┌──────────────┐                 │
│  │   FastAPI    │◄────►│  DistilBERT  │                 │
│  │   (8000)     │      │    Model     │                 │
│  └──────┬───────┘      └──────────────┘                 │
│         │                                               │
│         ▼                                               │
│  ┌──────────────┐      ┌──────────────┐                 │
│  │ Drift        │      │  SQLite DB   │                 │
│  │ Detectors    │◄────►│  (History)   │                 │
│  └──────┬───────┘      └──────────────┘                 │
│         │                                               │
│         ▼                                               │
│  ┌──────────────┐                                       │
│  │  Streamlit   │                                       │
│  │  Dashboard   │                                       │
│  │   (8501)     │                                       │
│  └──────────────┘                                       │
│                                                         │
└─────────────────────────────────────────────────────────┘
                    │
                    ▼
            ┌───────────────┐
            │   AWS EC2     │
            │   (t2.micro)  │
            └───────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker (optional, for containerized deployment)
- AWS account (optional, for cloud deployment)

### 1. Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/mental-health-mlops.git
cd mental-health-mlops
```

### 2. Install Dependencies
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 3. Run Locally

**Option A: Run API**
```bash
# Start FastAPI server
uvicorn src.api.app:app --host 0.0.0.0 --port 8000

# Access at: http://localhost:8000
# API docs: http://localhost:8000/docs
```

**Option B: Run Dashboard**
```bash
# Start Streamlit dashboard
streamlit run src/dashboard/drift_dashboard.py

# Access at: http://localhost:8501
```

**Option C: Run Both with Docker**
```bash
# Build image
docker build -f Dockerfile.lightweight -t mental-health-mlops:production .

# Run container
docker run -d \
  --name mlops-app \
  -p 8000:8000 \
  -p 8501:8501 \
  mental-health-mlops:production

# Access API: http://localhost:8000
# Access Dashboard: http://localhost:8501
```

---

## 📦 Project Structure
```
mental-health-mlops/
├── data/
│   ├── raw/                    # Original datasets
│   ├── processed/              # Preprocessed data (train/val/test)
│   └── drift_monitoring.db     # Drift history database
├── models/
│   ├── distilbert_exp2/        # Trained DistilBERT model
│   └── baseline/               # Baseline ML models
├── src/
│   ├── api/                    # FastAPI application
│   │   ├── main.py            # API endpoints
│   │   ├── model_loader.py    # Model loading
│   │   └── static/            # Web UI
│   ├── monitoring/             # Drift detection
│   │   ├── data_drift.py      # PSI, KS tests
│   │   ├── prediction_drift.py # JS divergence
│   │   └── drift_simulator.py  # Test scenarios
│   └── dashboard/              # Monitoring dashboard
│       ├── drift_dashboard.py  # Streamlit app
│       ├── drift_database.py   # Database interface
│       └── traffic_simulator.py # Demo traffic
├── experiments/                # Analysis & validation
│   └── drift_analysis/
│       ├── sensitivity_analysis.py
│       ├── feature_attribution.py
│       └── false_positive_test.py
├── tests/                      # Unit & integration tests
├── docs/                       # Documentation
│   └── report/                # LaTeX project report
├── Dockerfile.lightweight      # Production Dockerfile
├── requirements.production.txt # Lean dependencies
└── README.md
```

---

## 🔬 Drift Detection System

### Multi-Detector Approach

The system implements three types of drift detection:

#### 1. **Data Drift** (Statistical Distribution)
- **Methods:** Population Stability Index (PSI), Kolmogorov-Smirnov test
- **Monitors:** Text length, word count, vocabulary changes
- **Threshold:** PSI > 0.1
- **Result:** 16% false positive rate (acceptable)

#### 2. **Concept Drift** (Performance Degradation)
- **Methods:** Performance metric tracking (F1, recall, precision)
- **Monitors:** Model accuracy over time
- **Threshold:** >5% F1 drop

#### 3. **Prediction Drift** (Output Distribution)
- **Methods:** Jensen-Shannon divergence, Wasserstein distance
- **Monitors:** Class distribution changes
- **Threshold:** JS > 0.1
- **Result:** 0% false positive rate (excellent)

### Key Findings

From comprehensive drift analysis:

| Drift Type | Detection Threshold | Key Insight |
|------------|-------------------|-------------|
| **Length Drift** | 1.3x (30% increase) | Detected by `text_length`, `word_count` |
| **Population Drift** | 40% dominance (5.7x shift) | Detected by prediction distribution |
| **Slang/Linguistic** | Not detected | Statistical features insensitive to vocabulary changes |

**Critical Insight:** Statistical drift detection is insufficient for linguistic changes. When "depressed" → "down bad" (slang), text length/word count remain stable, so distribution-based methods fail. This demonstrates why **multiple detector types** are essential.

---

## 🧪 Validation & Testing

### Sensitivity Analysis

Determined minimum drift intensity for detection:
```bash
python experiments/drift_analysis/sensitivity_analysis.py
```

**Results:**
- Length drift detected at **1.3x** factor
- Population drift detected at **40%** dominance
- Slang drift **not detected** (reveals statistical method limitation)

### Feature Attribution

Identified which features detect which drift types:
```bash
python experiments/drift_analysis/feature_attribution.py
```

**Results:**
- `text_length` and `word_count` → Length drift (PSI > 0.3)
- `avg_word_length` → Multi-drift scenarios (PSI = 0.439)
- No single feature catches all drift types

### False Positive Testing

Validated detector reliability on stable data:
```bash
python experiments/drift_analysis/false_positive_test.py
```

**Results:**
- Data drift: 16% FP rate
- Prediction drift: 0% FP rate
- Overall: 8% (within acceptable <10%)

---

## 🎭 Demo: Traffic Simulation

Simulate 30 days of production traffic with gradual drift:
```bash
# Generate drift scenario
python src/dashboard/traffic_simulator.py --days 30 --delay 10 --reset

# Watch dashboard live at: http://localhost:8501
```

**Simulation Schedule:**
- **Days 1-10:** ✅ Normal operations (no drift)
- **Days 11-20:** ⚠️ Gradual drift (text lengths increasing)
- **Days 21-30:** 🚨 Significant drift (population shift detected)

---

## ☁️ AWS Deployment

### Deployed Infrastructure

- **Instance Type:** t2.micro (AWS Free Tier)
- **Region:** us-east-1
- **OS:** Ubuntu 22.04 LTS
- **Container:** Docker
- **Ports:** 8000 (API), 8501 (Dashboard)

### Deployment Steps

1. **Launch EC2 Instance**
```bash
# t2.micro, Ubuntu 22.04, 30GB storage
```

2. **Install Docker**
```bash
sudo apt-get update
sudo apt-get install -y docker.io
sudo usermod -aG docker ubuntu
```

3. **Deploy Application**
```bash
# Clone repo
git clone https://github.com/YOUR_USERNAME/mental-health-mlops.git
cd mental-health-mlops

# Build and run
docker build -f Dockerfile.lightweight -t mental-health-mlops:production .
docker run -d --name mlops-app -p 8000:8000 -p 8501:8501 mental-health-mlops:production
```

4. **Configure Security Group**
- Port 22 (SSH)
- Port 8000 (API) - 0.0.0.0/0
- Port 8501 (Dashboard) - 0.0.0.0/0

**Live URLs:**
- API: http://98.93.153.242:8000
- Dashboard: http://98.93.153.242:8501

---

## 📊 API Usage

### Make a Prediction

**cURL:**
```bash
curl -X POST "http://98.93.153.242:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "I feel very anxious and depressed"}'
```

**Python:**
```python
import requests

response = requests.post(
    "http://98.93.153.242:8000/predict",
    json={"text": "I feel very anxious and depressed"}
)

print(response.json())
# Output: {"prediction": "Anxiety", "confidence": 0.89, ...}
```

**Response Format:**
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
  "timestamp": "2024-12-02T10:30:45.123Z"
}
```

### Health Check
```bash
curl http://98.93.153.242:8000/health
```

---

## 🧪 Running Tests

### Unit Tests
```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_drift_detection.py -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

### Integration Tests
```bash
# Test API endpoints
pytest tests/test_api.py -v

# Test drift detection
pytest tests/test_drift_detection.py -v
```

**Test Results:** 16/16 tests passing ✅

---

## 📈 Monitoring Dashboard

The Streamlit dashboard provides real-time monitoring:

### Features

- **📊 Drift Score Timeline:** Historical PSI/JS divergence over time
- **🎯 Current Status:** Real-time system health (Healthy/Warning/Critical)
- **🔍 Feature Breakdown:** Per-feature drift analysis
- **🔔 Alert Feed:** Recent drift events and warnings
- **📉 Statistics:** Drift rate, average scores, detection frequency

### Dashboard Views

**Access at:** http://98.93.153.242:8501

1. **Overview:** System status, latest drift check
2. **Timeline:** Drift score chart with threshold lines
3. **Features:** Bar chart showing which features drifted
4. **Alerts:** Chronological feed of system events
5. **Recommendations:** Actionable guidance based on drift status

---

## 🛠️ Development

### Install Development Dependencies
```bash
pip install -r requirements.txt  # Full dependencies including dev tools
```

### Code Quality
```bash
# Format code
black src/ tests/

# Lint
flake8 src/ tests/

# Type checking
mypy src/
```

### Pre-commit Hooks
```bash
# Install pre-commit
pip install pre-commit
pre-commit install

# Run manually
pre-commit run --all-files
```

---

## 📚 Documentation

- **API Docs:** http://98.93.153.242:8000/docs (Swagger UI)
- **Project Report:** [docs/report/main.pdf](docs/report/main.pdf)
- **Architecture:** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- **Drift Detection:** [docs/DRIFT_DETECTION.md](docs/DRIFT_DETECTION.md)
- **Deployment Guide:** [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

<!-- ## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

--- -->

## 🙏 Acknowledgments

- **Dataset:** Mental health text data from [source]
- **Models:** Hugging Face Transformers library
- **Deployment:** AWS Free Tier
- **Frameworks:** FastAPI, Streamlit, PyTorch

---

## 🎯 Project Status

**Current Version:** 1.0.0
**Status:** ✅ Production Ready
**Last Updated:** December 2024

### Roadmap

- [x] Data preprocessing pipeline
- [x] Model training (baseline + transformer)
- [x] Drift detection system
- [x] Production API
- [x] Monitoring dashboard
- [x] Docker containerization
- [x] AWS deployment
- [ ] Embedding-based semantic drift detection
- [ ] A/B testing framework
- [ ] Model retraining automation

---

**⭐ If you find this project useful, please consider giving it a star!**

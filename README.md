# Mental Health MLOps Pipeline

A comprehensive MLOps pipeline for sentiment analysis on mental health text data with 7 classes: Normal, Depression, Suicidal, Anxiety, Bipolar, Stress, and Personality Disorder.

## 🎯 Project Overview

This project implements an end-to-end machine learning pipeline with:
- **Experiment Tracking**: MLflow
- **Data Versioning**: DVC
- **Containerization**: Docker
- **Cloud Infrastructure**: AWS (S3, ECR, SageMaker/ECS)
- **CI/CD**: GitHub Actions
- **Drift Detection**: Evidently AI

## 📁 Project Structure

```
mental-health-mlops/
├── src/
│   ├── data/           # Data processing modules
│   ├── models/         # Model training and evaluation
│   ├── evaluation/     # Model evaluation scripts
│   ├── api/           # FastAPI application
│   └── monitoring/    # Drift detection and monitoring
├── data/
│   ├── raw/           # Original dataset (tracked by DVC)
│   └── processed/     # Processed data
├── models/            # Saved model artifacts
├── configs/           # Configuration files
├── notebooks/         # Jupyter notebooks for EDA
├── tests/            # Unit and integration tests
├── scripts/          # Utility scripts
├── .github/
│   └── workflows/    # GitHub Actions CI/CD
└── docs/             # Documentation
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- AWS CLI configured
- Docker installed
- Git

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd mental-health-mlops
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Mac/Linux
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Initialize DVC:
```bash
dvc init
dvc remote add -d myremote s3://mental-health-mlops-data
```

5. Pull data:
```bash
dvc pull
```

## 📊 Dataset

- **Classes**: 7 (Normal, Depression, Suicidal, Anxiety, Bipolar, Stress, Personality Disorder)
- **Size**: ~53,000 samples
- **Format**: CSV with text and label columns

## 🔧 Development Workflow

1. **Experimentation**: Use MLflow to track experiments
2. **Training**: Run training pipeline with `python src/models/train.py`
3. **Evaluation**: Evaluate model performance
4. **Drift Detection**: Monitor for data and model drift
5. **Deployment**: Deploy via Docker to AWS

## 📈 Monitoring

- MLflow UI: Track experiments and models
- Drift Detection: Automated monitoring for data and concept drift
- CloudWatch: Production monitoring and alerting

## 🤝 Contributing

1. Create a feature branch
2. Make changes and test
3. Submit pull request

## 📊 Data Management

**Current Setup (Development)**:
- Raw and processed data files are stored locally
- Tracked in `.gitignore` to avoid large files in Git
- Will be migrated to DVC + S3 storage before deployment

**Data Files** (local only):
- `data/raw/mental_health_data.csv` - Original dataset
- `data/processed/train.csv` - Training set (70%)
- `data/processed/val.csv` - Validation set (15%)
- `data/processed/test.csv` - Test set (15%)
- `data/processed/reference_data.csv` - Reference for drift detection

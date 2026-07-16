"""
API endpoint tests using FastAPI TestClient with a mocked model.

The real DistilBERT model (268 MB) is not available in CI. We inject mock
implementations of model_loader and predictor into sys.modules before
importing app so that the lifespan startup succeeds without touching disk or
GPU, and every endpoint can be exercised without an external process.
"""

import os
import sys
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

# src/api uses bare module names (model_loader, predictor, schemas) because it
# is designed to be run from within that directory.  Add it to sys.path before
# any import of src.api.app.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "api"))

CLASSES = ["Anxiety", "Bipolar", "Depression", "Normal", "Stress", "Suicidal"]
_PROBS = {c: round(1 / len(CLASSES), 4) for c in CLASSES}


def _make_loader_mock() -> MagicMock:
    m = MagicMock()
    m.get_classes.return_value = CLASSES
    return m


def _make_predictor_mock() -> MagicMock:
    m = MagicMock()
    m.predict.return_value = {
        "prediction": "Anxiety",
        "confidence": 0.87,
        "probabilities": _PROBS,
        "class_index": 0,
    }
    m.predict_batch.return_value = [
        {"prediction": c, "confidence": 0.80, "probabilities": _PROBS}
        for c in ["Anxiety", "Normal", "Depression"]
    ]
    m.get_prediction_history.return_value = []
    m.get_prediction_distribution.return_value = {c: 0 for c in CLASSES}
    return m


# Inject mocks into sys.modules before app.py is imported so that
# `from model_loader import ModelLoader` and friends resolve to our fakes.
_loader_inst = _make_loader_mock()
_predictor_inst = _make_predictor_mock()

_ml_mod = MagicMock()
_ml_mod.ModelLoader.return_value = _loader_inst
sys.modules.setdefault("model_loader", _ml_mod)

_pred_mod = MagicMock()
_pred_mod.MentalHealthPredictor.return_value = _predictor_inst
sys.modules.setdefault("predictor", _pred_mod)

from src.api.app import app  # noqa: E402  (must follow sys.modules injection)


@pytest.fixture(scope="module")
def client():
    """TestClient that exercises the full lifespan (startup → tests → shutdown)."""
    with TestClient(app) as c:
        yield c


class TestAPIEndpoints:
    def test_root(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert "message" in data
        assert "endpoints" in data

    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert "uptime_seconds" in data

    def test_model_info(self, client):
        resp = client.get("/model-info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_name"] == "DistilBERT"
        assert data["f1_macro"] == 0.8189
        assert data["num_classes"] == 6
        assert len(data["classes"]) == 6
        assert set(data["classes"]) == set(CLASSES)

    def test_predict_returns_valid_structure(self, client):
        resp = client.post("/predict", json={"text": "I feel anxious and can't sleep"})
        assert resp.status_code == 200
        data = resp.json()
        assert "prediction" in data
        assert "confidence" in data
        assert "probabilities" in data
        assert "drift_detected" in data
        assert data["prediction"] in CLASSES
        assert 0.0 <= data["confidence"] <= 1.0
        assert set(data["probabilities"].keys()) == set(CLASSES)

    def test_predict_empty_text_returns_422(self, client):
        resp = client.post("/predict", json={"text": ""})
        assert resp.status_code == 422

    def test_predict_whitespace_only_returns_422(self, client):
        resp = client.post("/predict", json={"text": "   "})
        assert resp.status_code == 422

    def test_batch_predict(self, client):
        texts = ["I feel anxious", "Everything is great", "I feel hopeless"]
        resp = client.post("/predict_batch", json={"texts": texts})
        assert resp.status_code == 200
        data = resp.json()
        assert "predictions" in data
        assert len(data["predictions"]) == len(texts)
        for pred in data["predictions"]:
            assert "prediction" in pred
            assert "confidence" in pred

    def test_drift_status(self, client):
        resp = client.get("/drift-status")
        assert resp.status_code == 200
        data = resp.json()
        assert "drift_score" in data
        assert "predictions_since_check" in data
        assert isinstance(data["drift_score"], float)
        assert isinstance(data["predictions_since_check"], int)

    def test_prediction_distribution(self, client):
        resp = client.get("/prediction-distribution")
        assert resp.status_code == 200
        data = resp.json()
        assert "distribution" in data
        assert "total_predictions" in data

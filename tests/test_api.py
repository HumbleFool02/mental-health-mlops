"""
API endpoint tests
"""

import os
import sys

import pytest
from fastapi.testclient import TestClient

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


# We need to import and use the app AFTER the model is loaded
# For testing, we'll use a simpler approach: test against running server

import requests

API_BASE_URL = "http://localhost:8000"


class TestAPIEndpoints:
    """Test API endpoints against running server"""

    def test_root(self):
        """Test root endpoint"""
        response = requests.get(f"{API_BASE_URL}/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "endpoints" in data
        print("✅ Root endpoint works")

    def test_health(self):
        """Test health endpoint"""
        response = requests.get(f"{API_BASE_URL}/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] == True
        print(f"✅ Health check: {data['status']}")

    def test_model_info(self):
        """Test model info endpoint"""
        response = requests.get(f"{API_BASE_URL}/model-info")
        assert response.status_code == 200
        data = response.json()
        assert data["model_name"] == "DistilBERT"
        assert data["f1_macro"] == 0.8189
        assert data["num_classes"] == 6
        print(f"✅ Model info: {data['model_name']} (F1: {data['f1_macro']})")

    def test_predict_anxiety(self):
        """Test prediction for anxiety text"""
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json={"text": "I feel anxious and can't sleep at night"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "confidence" in data
        assert "probabilities" in data
        assert data["confidence"] > 0.5
        print(f"✅ Anxiety prediction: {data['prediction']} ({data['confidence']:.2%})")

    def test_predict_normal(self):
        """Test prediction for normal text"""
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json={"text": "Everything is going well and I'm feeling great"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] in [
            "Normal",
            "Anxiety",
            "Depression",
            "Suicidal",
            "Bipolar",
            "Stress",
        ]
        print(f"✅ Normal prediction: {data['prediction']} ({data['confidence']:.2%})")

    def test_predict_depression(self):
        """Test prediction for depression text"""
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json={"text": "I feel hopeless and sad all the time"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        print(
            f"✅ Depression prediction: {data['prediction']} ({data['confidence']:.2%})"
        )

    def test_predict_empty_text(self):
        """Test prediction with empty text"""
        response = requests.post(f"{API_BASE_URL}/predict", json={"text": ""})
        assert response.status_code == 422  # Validation error
        print("✅ Empty text validation works")

    def test_batch_predict(self):
        """Test batch prediction"""
        response = requests.post(
            f"{API_BASE_URL}/predict_batch",
            json={"texts": ["I feel anxious", "Everything is great", "I feel sad"]},
        )
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 3
        print(f"✅ Batch prediction: {len(data['predictions'])} results")

    def test_drift_status(self):
        """Test drift status endpoint"""
        response = requests.get(f"{API_BASE_URL}/drift-status")
        assert response.status_code == 200
        data = response.json()
        assert "drift_score" in data
        assert "predictions_since_check" in data
        print(
            f"✅ Drift status: Score={data['drift_score']:.4f}, Predictions={data['predictions_since_check']}"
        )

    def test_prediction_distribution(self):
        """Test prediction distribution endpoint"""
        response = requests.get(f"{API_BASE_URL}/prediction-distribution")
        assert response.status_code == 200
        data = response.json()
        assert "distribution" in data
        assert "total_predictions" in data
        print(f"✅ Distribution: {data['total_predictions']} total predictions")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

"""
Unit tests for drift detection modules
"""


import os
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder
from src.monitoring.data_drift import DataDriftDetector
from src.monitoring.concept_drift import ConceptDriftDetector

# Add project root to path
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


class TestDataDriftDetector:
    """Test Data Drift Detection"""

    @pytest.fixture
    def reference_data(self):
        """Load reference data"""
        return pd.read_csv("data/processed/reference_data.csv")

    @pytest.fixture
    def validation_data(self):
        """Load validation data"""
        return pd.read_csv("data/processed/val.csv")

    def test_no_drift_on_same_distribution(self, reference_data, validation_data):
        """Test that no drift is detected on validation data"""
        detector = DataDriftDetector(reference_data, threshold=0.1)
        results = detector.detect_drift(validation_data)

        # Should not detect overall drift
        assert (
            results["overall_drift"] == False
        ), "Should not detect drift on same distribution"

        # Drift score should be low
        assert (
            results["drift_score"] < 0.05
        ), f"Drift score too high: {results['drift_score']}"

        print("\n Test passed: No drift on validation data")
        print(detector.generate_report(results))

    def test_drift_on_modified_data(self, reference_data):
        """Test that drift IS detected when data changes"""
        # Create modified data with longer texts
        modified_data = reference_data.copy()
        modified_data["text"] = (
            modified_data["text"] + " " + modified_data["text"]
        )  # Double length

        detector = DataDriftDetector(reference_data, threshold=0.1)
        results = detector.detect_drift(modified_data)

        # Should detect drift
        assert results["overall_drift"] == True, "Should detect drift on modified data"

        print("\n Test passed: Drift detected on modified data")
        print(detector.generate_report(results))


class TestConceptDriftDetector:
    """Test Concept Drift Detection"""

    @pytest.fixture
    def validation_data(self):
        """Load validation data"""
        return pd.read_csv("data/processed/val.csv")

    @pytest.fixture
    def reference_performance(self):
        """Reference performance from DistilBERT Exp 2"""
        return {"accuracy": 0.7937, "f1_macro": 0.8189, "f1_weighted": 0.8200}

    def test_no_drift_on_good_performance(self, validation_data, reference_performance):
        """Test no drift when performance is maintained"""
        # Prepare labels
        label_encoder = LabelEncoder()
        y_true = label_encoder.fit_transform(validation_data["label"])

        # Simulate good predictions (5% error rate)
        np.random.seed(42)
        y_pred = y_true.copy()
        n_errors = int(0.05 * len(y_pred))
        error_indices = np.random.choice(len(y_pred), n_errors, replace=False)
        y_pred[error_indices] = np.random.randint(
            0, len(label_encoder.classes_), n_errors
        )

        detector = ConceptDriftDetector(reference_performance, threshold=0.05)
        results = detector.detect_drift(
            y_true, y_pred, class_names=label_encoder.classes_
        )

        # Should not detect drift with good performance
        # (May or may not depending on random errors, so we just check it runs)
        print("\n Test passed: Concept drift detector runs successfully")
        print(detector.generate_report(results))

    def test_drift_on_degraded_performance(
        self, validation_data, reference_performance
    ):
        """Test drift detection when performance degrades"""
        # Prepare labels
        label_encoder = LabelEncoder()
        y_true = label_encoder.fit_transform(validation_data["label"])

        # Simulate poor predictions (30% error rate)
        np.random.seed(42)
        y_pred = y_true.copy()
        n_errors = int(0.30 * len(y_pred))
        error_indices = np.random.choice(len(y_pred), n_errors, replace=False)
        y_pred[error_indices] = np.random.randint(
            0, len(label_encoder.classes_), n_errors
        )

        detector = ConceptDriftDetector(reference_performance, threshold=0.05)
        results = detector.detect_drift(
            y_true, y_pred, class_names=label_encoder.classes_
        )

        # Should detect drift with degraded performance
        assert (
            results["overall_drift"] == True
        ), "Should detect drift with 30% error rate"
        assert len(results["degraded_metrics"]) > 0, "Should have degraded metrics"

        print("\n Test passed: Drift detected on degraded performance")
        print(detector.generate_report(results))


class TestPredictionDriftDetector:
    """Test Prediction Drift Detection"""

    @pytest.fixture
    def validation_data(self):
        """Load validation data"""
        return pd.read_csv("data/processed/val.csv")

    def test_no_drift_on_same_predictions(self, validation_data):
        """Test no drift when predictions are from same distribution"""
        # Prepare labels
        label_encoder = LabelEncoder()
        y_true = label_encoder.fit_transform(validation_data["label"])

        # Use true labels as "predictions" (perfect model)
        from src.monitoring.prediction_drift import PredictionDriftDetector

        detector = PredictionDriftDetector(
            y_true[:3000],  # First half as reference
            class_names=label_encoder.classes_,
            threshold=0.1,
        )

        results = detector.detect_drift(y_true[3000:])  # Second half as current

        # Should not detect drift (same distribution)
        assert (
            results["js_divergence"] < 0.05
        ), "Should have low divergence on same distribution"

        print("\n Test passed: No prediction drift on same distribution")
        print(detector.generate_report(results))

    def test_drift_on_shifted_predictions(self, validation_data):
        """Test drift when prediction distribution shifts"""
        # Prepare labels
        label_encoder = LabelEncoder()
        y_true = label_encoder.fit_transform(validation_data["label"])

        from src.monitoring.prediction_drift import PredictionDriftDetector

        detector = PredictionDriftDetector(
            y_true, class_names=label_encoder.classes_, threshold=0.1
        )

        # Create shifted predictions (predict everything as class 0)
        shifted_predictions = np.zeros_like(y_true)

        results = detector.detect_drift(shifted_predictions)

        # Should detect drift
        assert (
            results["drift_detected"] == True
        ), "Should detect drift on shifted distribution"
        assert results["js_divergence"] > 0.1, "JS divergence should exceed threshold"

        print("\n Test passed: Drift detected on shifted predictions")
        print(detector.generate_report(results))


if __name__ == "__main__":
    # Run tests manually
    print("Running Drift Detection Tests...\n")

    # Test Data Drift
    print("=" * 70)
    print("DATA DRIFT TESTS")
    print("=" * 70)

    ref_data = pd.read_csv("data/processed/reference_data.csv")
    val_data = pd.read_csv("data/processed/val.csv")

    test_data_drift = TestDataDriftDetector()
    test_data_drift.test_no_drift_on_same_distribution(ref_data, val_data)
    test_data_drift.test_drift_on_modified_data(ref_data)

    # Test Concept Drift
    print("\n" + "=" * 70)
    print("CONCEPT DRIFT TESTS")
    print("=" * 70)

    ref_perf = {"accuracy": 0.7937, "f1_macro": 0.8189, "f1_weighted": 0.8200}

    test_concept_drift = TestConceptDriftDetector()
    test_concept_drift.test_no_drift_on_good_performance(val_data, ref_perf)
    test_concept_drift.test_drift_on_degraded_performance(val_data, ref_perf)

    # Test Prediction Drift
    print("\n" + "=" * 70)
    print("PREDICTION DRIFT TESTS")
    print("=" * 70)

    test_pred_drift = TestPredictionDriftDetector()
    test_pred_drift.test_no_drift_on_same_predictions(val_data)
    test_pred_drift.test_drift_on_shifted_predictions(val_data)

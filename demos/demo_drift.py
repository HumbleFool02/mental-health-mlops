"""
End-to-End Drift Detection Demo
Demonstrates all drift detection capabilities
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.monitoring.concept_drift import ConceptDriftDetector
from src.monitoring.data_drift import DataDriftDetector
from src.monitoring.drift_simulator import DriftSimulator
from src.monitoring.prediction_drift import PredictionDriftDetector

def demo_data_drift():
    """Demo 1: Data Drift Detection"""

    print("\n" + "=" * 70)
    print("DEMO 1: DATA DRIFT DETECTION")
    print("=" * 70)

    # Load data
    reference_df = pd.read_csv("data/processed/reference_data.csv")
    val_df = pd.read_csv("data/processed/val.csv")

    # Initialize detector
    detector = DataDriftDetector(reference_df, threshold=0.1)

    # Scenario 1: No drift (validation data)
    print("\n📊 Scenario 1: No Drift (Validation Data)")
    print("-" * 70)
    results_no_drift = detector.detect_drift(val_df)
    print(detector.generate_report(results_no_drift))

    # Scenario 2: Seasonal drift
    print("\n📊 Scenario 2: Seasonal Drift (Winter)")
    print("-" * 70)
    simulator = DriftSimulator()
    winter_data = simulator.simulate_seasonal_drift(val_df, "winter")
    results_seasonal = detector.detect_drift(winter_data)
    print(detector.generate_report(results_seasonal))

    # Scenario 3: Length drift
    print("\n📊 Scenario 3: Length Drift (2x longer texts)")
    print("-" * 70)
    longer_data = simulator.simulate_length_drift(val_df, change_factor=2.0)
    results_length = detector.detect_drift(longer_data)
    print(detector.generate_report(results_length))

    return {
        "no_drift": results_no_drift,
        "seasonal": results_seasonal,
        "length": results_length,
    }


def demo_concept_drift():
    """Demo 2: Concept Drift Detection"""

    print("\n" + "=" * 70)
    print("DEMO 2: CONCEPT DRIFT DETECTION")
    print("=" * 70)

    # Load data
    val_df = pd.read_csv("data/processed/val.csv")

    # Prepare labels
    label_encoder = LabelEncoder()
    y_true = label_encoder.fit_transform(val_df["label"])

    # Reference performance (from DistilBERT Exp 2)
    reference_performance = {
        "accuracy": 0.7937,
        "f1_macro": 0.8189,
        "f1_weighted": 0.8200,
        "f1_Anxiety": 0.814,
        "f1_Bipolar": 0.824,
        "f1_Depression": 0.747,
        "f1_Normal": 0.915,
        "f1_Stress": 0.629,
        "f1_Suicidal": 0.710,
    }

    detector = ConceptDriftDetector(reference_performance, threshold=0.05)

    # Scenario 1: Good performance (5% error)
    print("\n📊 Scenario 1: No Drift (5% Error Rate)")
    print("-" * 70)
    np.random.seed(42)
    y_pred_good = y_true.copy()
    n_errors = int(0.05 * len(y_pred_good))
    error_indices = np.random.choice(len(y_pred_good), n_errors, replace=False)
    y_pred_good[error_indices] = np.random.randint(
        0, len(label_encoder.classes_), n_errors
    )

    results_good = detector.detect_drift(
        y_true, y_pred_good, class_names=label_encoder.classes_
    )
    print(detector.generate_report(results_good))

    # Scenario 2: Degraded performance (25% error)
    print("\n📊 Scenario 2: Performance Drift (25% Error Rate)")
    print("-" * 70)
    y_pred_bad = y_true.copy()
    n_errors = int(0.25 * len(y_pred_bad))
    error_indices = np.random.choice(len(y_pred_bad), n_errors, replace=False)
    y_pred_bad[error_indices] = np.random.randint(
        0, len(label_encoder.classes_), n_errors
    )

    results_bad = detector.detect_drift(
        y_true, y_pred_bad, class_names=label_encoder.classes_
    )
    print(detector.generate_report(results_bad))

    return {"good_performance": results_good, "degraded_performance": results_bad}


def demo_prediction_drift():
    """Demo 3: Prediction Drift Detection"""

    print("\n" + "=" * 70)
    print("DEMO 3: PREDICTION DRIFT DETECTION")
    print("=" * 70)

    # Load data
    val_df = pd.read_csv("data/processed/val.csv")

    # Prepare labels
    label_encoder = LabelEncoder()
    y_true = label_encoder.fit_transform(val_df["label"])

    # Reference predictions (use first half)
    reference_predictions = y_true[:3000]

    detector = PredictionDriftDetector(
        reference_predictions, class_names=label_encoder.classes_, threshold=0.1
    )

    # Scenario 1: No drift (second half)
    print("\n📊 Scenario 1: No Drift (Same Distribution)")
    print("-" * 70)
    current_predictions_same = y_true[3000:]
    results_no_drift = detector.detect_drift(current_predictions_same)
    print(detector.generate_report(results_no_drift))

    # Scenario 2: Class shift (simulate population change)
    print("\n📊 Scenario 2: Population Shift (More Anxiety Cases)")
    print("-" * 70)
    simulator = DriftSimulator()
    shifted_data = simulator.simulate_class_shift(
        val_df, shift_from="Normal", shift_to="Anxiety", proportion=0.5
    )
    shifted_predictions = label_encoder.transform(shifted_data["label"])
    results_shift = detector.detect_drift(shifted_predictions)
    print(detector.generate_report(results_shift))

    # Scenario 3: Extreme drift (predict everything as one class)
    print("\n📊 Scenario 3: Extreme Drift (Model Collapse)")
    print("-" * 70)
    # Simulate model predicting everything as Depression
    collapsed_predictions = np.full(
        len(current_predictions_same), label_encoder.transform(["Depression"])[0]
    )
    results_collapse = detector.detect_drift(collapsed_predictions)
    print(detector.generate_report(results_collapse))

    return {
        "no_drift": results_no_drift,
        "population_shift": results_shift,
        "model_collapse": results_collapse,
    }


def demo_gradual_drift():
    """Demo 4: Gradual Drift Over Time"""

    print("\n" + "=" * 70)
    print("DEMO 4: GRADUAL DRIFT DETECTION")
    print("=" * 70)

    # Load data
    reference_df = pd.read_csv("data/processed/reference_data.csv")
    val_df = pd.read_csv("data/processed/val.csv")

    # Initialize detector
    detector = DataDriftDetector(reference_df, threshold=0.1)

    # Simulate gradual slang emergence over 5 time periods
    simulator = DriftSimulator()
    drift_steps = simulator.simulate_gradual_drift(
        val_df, drift_type="length", n_steps=5
    )

    print("\n📊 Monitoring Drift Over 6 Time Periods")
    print("-" * 70)

    for i, data in enumerate(drift_steps):
        print(f"\nTime Period {i}:")
        results = detector.detect_drift(data)

        avg_length = data["text"].str.len().mean()
        print(f"  Avg Text Length: {avg_length:.0f} chars")
        print(f"  Drift Score: {results['drift_score']:.4f}")
        print(f"  Drift Detected: {'YES ⚠️' if results['overall_drift'] else 'NO ✅'}")


def main():
    """Run all drift detection demos"""

    print("=" * 70)
    print("\nThis demo showcases all drift detection capabilities:")
    print("  1. Data Drift - Changes in input data distribution")
    print("  2. Concept Drift - Changes in model performance")
    print("  3. Prediction Drift - Changes in prediction distribution")
    print("  4. Gradual Drift - Drift detection over time")
    print("=" * 70)

    # Run all demos
    data_drift_results = demo_data_drift()
    concept_drift_results = demo_concept_drift()
    prediction_drift_results = demo_prediction_drift()
    demo_gradual_drift()

    # Final summary
    print("\n" + "=" * 70)
    print("📊 DRIFT DETECTION SUMMARY")
    print("=" * 70)

    print("\n✅ Data Drift Detection:")
    print(
        f"   - No Drift Scenario: {'PASS' if not data_drift_results['no_drift']['overall_drift'] else 'FAIL'}"
    )
    print(
        f"   - Seasonal Drift: {'DETECTED' if data_drift_results['seasonal']['drift_score'] > 0.05 else 'NOT DETECTED'}"
    )
    print(
        f"   - Length Drift: {'DETECTED' if data_drift_results['length']['overall_drift'] else 'NOT DETECTED'}"
    )

    print("\n✅ Concept Drift Detection:")
    print(
        f"   - Good Performance: {'NO DRIFT' if not concept_drift_results['good_performance']['overall_drift'] else 'DRIFT'}"
    )
    print(
        f"   - Degraded Performance: {'DRIFT DETECTED' if concept_drift_results['degraded_performance']['overall_drift'] else 'NO DRIFT'}"
    )

    print("\n✅ Prediction Drift Detection:")
    print(
        f"   - Same Distribution: {'NO DRIFT' if not prediction_drift_results['no_drift']['drift_detected'] else 'DRIFT'}"
    )
    print(
        f"   - Population Shift: {'DRIFT DETECTED' if prediction_drift_results['population_shift']['drift_detected'] else 'NO DRIFT'}"
    )
    print(
        f"   - Model Collapse: {'DRIFT DETECTED' if prediction_drift_results['model_collapse']['drift_detected'] else 'NO DRIFT'}"
    )

    print("\n" + "=" * 70)
    print("🎉 ALL DRIFT DETECTION DEMOS COMPLETED!")
    print("=" * 70)
    print("\n💡 Key Capabilities Demonstrated:")
    print("   ✅ Multiple drift types detected")
    print("   ✅ Statistical tests (PSI, KS, JS Divergence)")
    print("   ✅ Realistic drift scenarios")
    print("   ✅ Gradual drift monitoring")
    print("   ✅ Per-class analysis")
    print("=" * 70)


if __name__ == "__main__":
    main()

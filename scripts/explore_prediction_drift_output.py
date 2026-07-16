"""
Exploratory script: print the full return structure of PredictionDriftDetector.
Run from the project root: python scripts/explore_prediction_drift_output.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np  # noqa: E402

from src.monitoring.prediction_drift import PredictionDriftDetector  # noqa: E402

print("Creating test predictions...")

label_to_idx = {
    "Anxiety": 0,
    "Bipolar": 1,
    "Depression": 2,
    "Normal": 3,
    "Stress": 4,
    "Suicidal": 5,
}

# Reference: balanced
ref_predictions = np.random.randint(0, 6, 200)

# Current: shifted (more anxiety)
curr_predictions = np.concatenate(
    [np.full(150, 0), np.random.randint(1, 6, 50)]  # 150 Anxiety, 50 others
)

print(f"Reference Anxiety %: {(ref_predictions == 0).mean() * 100:.1f}%")
print(f"Current Anxiety %: {(curr_predictions == 0).mean() * 100:.1f}%")

print("\nCreating detector...")
detector = PredictionDriftDetector(
    reference_predictions=ref_predictions, class_names=list(label_to_idx.keys())
)

print("Running detection...")
results = detector.detect_drift(curr_predictions)

print("\n" + "=" * 80)
print("PREDICTION DRIFT DETECTOR RETURN STRUCTURE")
print("=" * 80)

print("\nKeys returned:")
for key in results.keys():
    print(f"  - {key}: {type(results[key])}")

print("\nFull structure:")
print(json.dumps(results, indent=2, default=str))

print("\n" + "=" * 80)

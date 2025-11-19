"""
Test PredictionDriftDetector return structure
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from src.monitoring.prediction_drift import PredictionDriftDetector
import json

print("Creating test predictions...")

# Create sample predictions
label_to_idx = {
    'Anxiety': 0, 'Bipolar': 1, 'Depression': 2,
    'Normal': 3, 'Stress': 4, 'Suicidal': 5
}

# Reference: balanced
ref_predictions = np.random.randint(0, 6, 200)

# Current: shifted (more anxiety)
curr_predictions = np.concatenate([
    np.full(150, 0),  # 150 Anxiety
    np.random.randint(1, 6, 50)  # 50 others
])

print(f"Reference Anxiety %: {(ref_predictions == 0).mean()*100:.1f}%")
print(f"Current Anxiety %: {(curr_predictions == 0).mean()*100:.1f}%")

print("\nCreating detector...")
detector = PredictionDriftDetector(
    reference_predictions=ref_predictions,
    class_names=list(label_to_idx.keys())
)

print("Running detection...")
results = detector.detect_drift(curr_predictions)

print("\n" + "="*80)
print("PREDICTION DRIFT DETECTOR RETURN STRUCTURE")
print("="*80)

print("\nKeys returned:")
for key in results.keys():
    print(f"  - {key}: {type(results[key])}")

print("\nFull structure:")
print(json.dumps(results, indent=2, default=str))

print("\n" + "="*80)

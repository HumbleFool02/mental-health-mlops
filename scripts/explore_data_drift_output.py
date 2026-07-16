"""
Exploratory script: print the full return structure of DataDriftDetector.
Run from the project root: python scripts/explore_data_drift_output.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd  # noqa: E402

from src.monitoring.data_drift import DataDriftDetector  # noqa: E402

print("Loading data...")
data = pd.read_csv("data/processed/train.csv").sample(200, random_state=42)

print("Creating detector...")
detector = DataDriftDetector(data)

print("Running detection...")
results = detector.detect_drift(data)

print("\n" + "=" * 80)
print("DRIFT DETECTOR RETURN STRUCTURE")
print("=" * 80)

print("\nKeys returned:")
for key in results.keys():
    print(f"  - {key}: {type(results[key])}")

print("\nFull structure:")
print(json.dumps(results, indent=2, default=str))

print("\n" + "=" * 80)

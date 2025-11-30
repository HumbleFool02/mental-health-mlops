"""
Clear drift monitoring database
"""

import os
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")))

from src.dashboard.drift_database import DriftDatabase

db = DriftDatabase()

print("🗑️  Clearing drift monitoring database...")
db.clear_all_data()
print("✅ Database cleared!")

# Show stats
stats = db.get_drift_stats()
print(f"\nVerification:")
print(f"  Total checks: {stats['total_checks']}")
print(f"  Drift events: {stats['drift_detected_count']}")
print(f"  (Should both be 0)")

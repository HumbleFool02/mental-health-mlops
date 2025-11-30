"""
Clear drift monitoring database
"""
from src.dashboard.drift_database import DriftDatabase

db = DriftDatabase()

print("🗑️  Clearing drift monitoring database...")
db.clear_all_data()
print("✅ Database cleared!")

# Show stats
stats = db.get_drift_stats()
print("\nVerification:")
print(f"  Total checks: {stats['total_checks']}")
print(f"  Drift events: {stats['drift_detected_count']}")
print("  (Should both be 0)")

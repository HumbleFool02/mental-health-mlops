"""
Traffic Simulator

Simulates production traffic with gradual drift introduction
"""
import os
import random
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.dashboard.drift_database import DriftDatabase
from src.monitoring.data_drift import DataDriftDetector
from src.monitoring.drift_simulator_v2 import EnhancedDriftSimulator
from src.monitoring.prediction_drift import PredictionDriftDetector


class TrafficSimulator:
    """Simulates production traffic over time"""

    def __init__(self, db_path: str = "data/drift_monitoring.db"):
        """Initialize simulator"""
        self.db = DriftDatabase(db_path)
        self.simulator = EnhancedDriftSimulator()

        # Load reference data
        print("Loading reference data...")
        self.reference_data = pd.read_csv("data/processed/train.csv").sample(
            500, random_state=42
        )

        # Create detectors
        self.data_detector = DataDriftDetector(self.reference_data)

        # For prediction drift
        self.label_to_idx = {
            "Anxiety": 0,
            "Bipolar": 1,
            "Depression": 2,
            "Normal": 3,
            "Stress": 4,
            "Suicidal": 5,
        }
        ref_predictions = self.reference_data["label"].map(self.label_to_idx).values
        self.pred_detector = PredictionDriftDetector(
            reference_predictions=ref_predictions,
            class_names=list(self.label_to_idx.keys()),
        )

        print("✓ Simulator initialized")

    def simulate_day(self, day: int, total_days: int = 30):
        """
        Simulate one day of traffic

        Days 1-10:  Normal (no drift)
        Days 11-20: Gradual drift (length increasing)
        Days 21-30: Significant drift (population shift)
        """
        print(f"\n📅 Day {day}/{total_days}")

        # Determine drift intensity based on day
        if day <= 10:
            # No drift
            drift_factor = 0
            status = "✅ Normal operations"
        elif day <= 20:
            # Gradual drift
            drift_factor = (day - 10) / 10  # 0 to 1
            status = f"⚠️  Gradual drift (intensity: {drift_factor:.1f})"
        else:
            # Significant drift
            drift_factor = 1.0
            status = "🚨 Significant drift"

        print(f"   Status: {status}")

        # Generate sample for this day
        daily_sample = self.reference_data.sample(100, replace=True)

        # Apply drift if needed
        if drift_factor > 0:
            if day <= 20:
                # Length drift
                length_factor = 1.0 + (drift_factor * 0.5)  # 1.0 to 1.5x
                daily_sample = self.simulator.simulate_extreme_length_drift(
                    daily_sample, factor=length_factor, proportion=drift_factor
                )
            else:
                # Population collapse + length drift
                daily_sample = self.simulator.simulate_population_collapse(
                    daily_sample,
                    dominant_class="Anxiety",
                    dominance=0.3 + (0.5 * ((day - 20) / 10)),  # 30% to 80%
                )
                daily_sample = self.simulator.simulate_extreme_length_drift(
                    daily_sample, factor=1.5, proportion=0.8
                )

        # Run drift detection
        data_results = self.data_detector.detect_drift(daily_sample)

        # Log to database
        self.db.log_drift_check("data_drift", data_results)

        # Check if we should alert
        if data_results["overall_drift"]:
            self.db.log_alert(
                "drift_detected",
                f"Data drift detected on day {day} (PSI: {data_results['drift_score']:.4f})",
                severity="critical",
            )
            print(
                f"   🚨 ALERT: Drift detected (PSI: {data_results['drift_score']:.4f})"
            )
        elif data_results["drift_score"] > 0.07:
            self.db.log_alert(
                "drift_warning",
                f"Drift score trending up on day {day} (PSI: {data_results['drift_score']:.4f})",
                severity="warning",
            )
            print(f"   ⚠️  Warning: Drift score = {data_results['drift_score']:.4f}")
        else:
            print(f"   ✅ System healthy (PSI: {data_results['drift_score']:.4f})")

        # Log some predictions
        for i in range(min(10, len(daily_sample))):
            row = daily_sample.iloc[i]
            self.db.log_prediction(
                text=row["text"][:200],  # Truncate long texts
                prediction=row["label"],
                confidence=0.7 + random.random() * 0.3,  # Simulated confidence
            )

    def run_simulation(self, total_days: int = 30, delay: float = 10.0):
        """
        Run complete simulation

        Args:
            total_days: Number of days to simulate
            delay: Seconds between days (10 = 5 min for 30 days)
        """
        print("\n" + "=" * 80)
        print("🎭 TRAFFIC SIMULATOR")
        print("=" * 80)
        print(f"\nSimulating {total_days} days of production traffic")
        print(
            f"Delay: {delay} seconds per day (~{(total_days * delay / 60):.1f} min total)"
        )
        print("\nDrift schedule:")
        print("  Days 1-10:  ✅ Normal (no drift)")
        print("  Days 11-20: ⚠️  Gradual drift (length increasing)")
        print("  Days 21-30: 🚨 Significant drift (population shift)")
        print("\nStarting simulation...\n")

        for day in range(1, total_days + 1):
            self.simulate_day(day, total_days)

            if day < total_days:
                print(f"   ⏳ Waiting {delay}s until next day...")
                time.sleep(delay)

        print("\n" + "=" * 80)
        print("✅ SIMULATION COMPLETE")
        print("=" * 80)

        # Show summary
        stats = self.db.get_drift_stats()
        print(f"\n📊 Summary:")
        print(f"   Total drift checks: {stats['total_checks']}")
        print(f"   Drift detected: {stats['drift_detected_count']} times")
        print(f"   Drift rate: {stats['drift_rate']*100:.1f}%")
        print(f"   Average drift score: {stats['avg_drift_score']:.4f}")

        recent_alerts = self.db.get_recent_alerts(limit=5)
        print(f"\n🔔 Recent Alerts:")
        for alert in recent_alerts:
            severity_icon = {"critical": "🚨", "warning": "⚠️", "info": "ℹ️"}.get(
                alert["severity"], "•"
            )
            print(f"   {severity_icon} {alert['message']}")


def main():
    """Run simulation"""
    import argparse

    parser = argparse.ArgumentParser(description="Simulate production traffic")
    parser.add_argument(
        "--days", type=int, default=30, help="Number of days to simulate"
    )
    parser.add_argument(
        "--delay", type=float, default=10.0, help="Seconds between days"
    )
    parser.add_argument(
        "--reset", action="store_true", help="Clear existing data first"
    )

    args = parser.parse_args()

    # Create simulator
    sim = TrafficSimulator()

    # Reset if requested
    if args.reset:
        print("🗑️  Clearing existing data...")
        sim.db.clear_all_data()
        print("✓ Data cleared\n")

    # Run simulation
    sim.run_simulation(total_days=args.days, delay=args.delay)


if __name__ == "__main__":
    main()

"""
Enhanced Drift Detection Demo - Dramatic Changes
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

from src.monitoring.data_drift import DataDriftDetector
from src.monitoring.drift_simulator_v2 import EnhancedDriftSimulator
from src.monitoring.prediction_drift import PredictionDriftDetector


def print_drift_results(results, detector_type="DATA"):
    """
    Print drift detection results in a clean format
    """
    print("\n" + "=" * 80)
    print(f"🔍 {detector_type} DRIFT DETECTION RESULTS")
    print("=" * 80)

    # Overall status
    if "overall_drift" in results:
        status = "🚨 DRIFT DETECTED!" if results["overall_drift"] else "✅ NO DRIFT"
        print(f"\n{status}")

    if "drift_detected" in results:
        drift_val = results["drift_detected"]
        # Handle numpy bool
        if hasattr(drift_val, "item"):
            drift_val = drift_val.item()
        status = "🚨 DRIFT DETECTED!" if drift_val else "✅ NO DRIFT"
        print(f"\n{status}")

    # Basic metrics
    if "drift_score" in results:
        print(f"📊 Overall Drift Score (PSI): {results['drift_score']:.4f}")

    if "n_samples" in results:
        print(f"📦 Samples analyzed: {results['n_samples']}")

    if "features_with_drift" in results:
        print(f"🚨 Features with drift: {results['features_with_drift']}")

    # Per-feature analysis (for data drift)
    if "features" in results:
        print(f"\n📋 Per-Feature Analysis:")
        for feature, metrics in results["features"].items():
            # Handle string "False"/"True" or boolean
            drift_detected = metrics.get("drift_detected", False)
            if isinstance(drift_detected, str):
                drift_detected = drift_detected.lower() == "true"

            status_icon = "🚨 DRIFT" if drift_detected else "✅ OK"
            psi = metrics.get("psi", 0)
            mean_diff = metrics.get("mean_diff_pct", 0)

            print(
                f"   {feature:20s}: PSI={psi:.4f}, Change={mean_diff:+.1f}% {status_icon}"
            )

    # Prediction drift metrics
    if "js_divergence" in results:
        print(f"\n📊 JS Divergence: {results['js_divergence']:.4f}")
        threshold = 0.1
        status = "🚨 ABOVE" if results["js_divergence"] > threshold else "✅ BELOW"
        print(f"   Threshold: {threshold} {status}")

    if "wasserstein_distance" in results:
        print(f"📊 Wasserstein Distance: {results['wasserstein_distance']:.4f}")

    # Distribution comparison - CORRECTED
    if "distribution_comparison" in results:
        print(f"\n📊 Class Distribution Changes:")
        comp = results["distribution_comparison"]

        for cls, metrics in comp.items():
            # Correct keys: reference_frequency and current_frequency
            ref_freq = metrics.get("reference_frequency", 0)
            curr_freq = metrics.get("current_frequency", 0)
            diff_pct = metrics.get("difference_pct", 0)

            ref_pct = ref_freq * 100
            curr_pct = curr_freq * 100

            # Indicator based on absolute change
            if abs(diff_pct) > 100:
                indicator = "🚨🚨"  # Extreme change (>100%)
            elif abs(diff_pct) > 50:
                indicator = "🚨"  # Large change (>50%)
            elif abs(diff_pct) > 20:
                indicator = "⚠️"  # Moderate change (>20%)
            else:
                indicator = "✅"  # Small change (<20%)

            print(
                f"   {cls:15s}: {ref_pct:5.1f}% → {curr_pct:5.1f}% ({diff_pct:+6.1f}%) {indicator}"
            )

    print("=" * 80)


def demo_scenario_1_extreme_length():
    """Demo: Extreme text length change"""
    print("\n" + "=" * 80)
    print("🎬 SCENARIO 1: EXTREME TEXT LENGTH CHANGE")
    print("=" * 80)
    print("Real-world: Users start writing much longer descriptions")
    print("Impact: Text 3x longer than training data\n")

    # Load data
    print("Loading data...")
    reference_data = pd.read_csv("data/processed/train.csv").sample(
        1000, random_state=42
    )

    # Simulate drift
    print("Simulating extreme length drift...")
    simulator = EnhancedDriftSimulator()
    drifted_data = simulator.simulate_extreme_length_drift(
        reference_data, factor=3.0, proportion=1.0  # Triple the length!  # All samples
    )

    # Generate report
    report = simulator.generate_comparison_report(reference_data, drifted_data)
    simulator.print_report(report)

    # Detect drift
    print("\n🔍 RUNNING DRIFT DETECTION...")
    detector = DataDriftDetector(reference_data)
    results = detector.detect_drift(drifted_data)

    print_drift_results(results, "DATA")

    # Summary
    if results["overall_drift"]:
        print("\n✅ SUCCESS: Drift correctly detected!")
        print(
            f"   System identified {results['features_with_drift']} features with significant drift"
        )
    else:
        print("\n⚠️  WARNING: No drift detected (may need more dramatic changes)")

    return results


def demo_scenario_2_heavy_slang():
    """Demo: Heavy Gen Z slang injection"""
    print("\n" + "=" * 80)
    print("🎬 SCENARIO 2: HEAVY SLANG INJECTION")
    print("=" * 80)
    print("Real-world: Platform becomes popular with Gen Z")
    print("Impact: 80% of text uses slang ('down bad', 'no cap', etc.)\n")

    # Load data
    print("Loading data...")
    reference_data = pd.read_csv("data/processed/train.csv").sample(
        1000, random_state=42
    )

    # Simulate drift
    print("Simulating heavy slang injection...")
    simulator = EnhancedDriftSimulator()
    drifted_data = simulator.simulate_heavy_slang(
        reference_data, intensity=0.8  # 80% transformation rate
    )

    # Generate report
    report = simulator.generate_comparison_report(reference_data, drifted_data)
    simulator.print_report(report)

    # Detect drift
    print("\n🔍 RUNNING DRIFT DETECTION...")
    detector = DataDriftDetector(reference_data)
    results = detector.detect_drift(drifted_data)

    print_drift_results(results, "DATA")

    # Summary
    if results["overall_drift"]:
        print("\n✅ SUCCESS: Vocabulary drift detected!")
        print(f"   Changed text detected through statistical features")
    else:
        print("\n⚠️  Note: Slang changes vocabulary, not necessarily length/word count")
        print("   This demonstrates the importance of multi-detector approach")

    return results


def demo_scenario_3_population_collapse():
    """Demo: One class becomes dominant"""
    print("\n" + "=" * 80)
    print("🎬 SCENARIO 3: POPULATION COLLAPSE")
    print("=" * 80)
    print("Real-world: Platform becomes known for anxiety support")
    print("Impact: Anxiety cases go from 7% → 85%!\n")

    # Load data
    print("Loading data...")
    reference_data = pd.read_csv("data/processed/train.csv").sample(
        1000, random_state=42
    )

    print(
        f"Original Anxiety %: {(reference_data['label'] == 'Anxiety').mean()*100:.1f}%"
    )

    # Simulate drift
    print("\nSimulating population collapse...")
    simulator = EnhancedDriftSimulator()
    drifted_data = simulator.simulate_population_collapse(
        reference_data, dominant_class="Anxiety", dominance=0.85
    )

    print(f"Drifted Anxiety %: {(drifted_data['label'] == 'Anxiety').mean()*100:.1f}%")

    # Generate report
    report = simulator.generate_comparison_report(reference_data, drifted_data)
    simulator.print_report(report)

    # For prediction drift, we need class indices
    print("\n🔍 RUNNING PREDICTION DRIFT DETECTION...")

    # Map labels to indices
    label_to_idx = {
        "Anxiety": 0,
        "Bipolar": 1,
        "Depression": 2,
        "Normal": 3,
        "Stress": 4,
        "Suicidal": 5,
    }

    ref_predictions = reference_data["label"].map(label_to_idx).values
    drift_predictions = drifted_data["label"].map(label_to_idx).values

    pred_detector = PredictionDriftDetector(
        reference_predictions=ref_predictions, class_names=list(label_to_idx.keys())
    )
    pred_results = pred_detector.detect_drift(drift_predictions)

    print_drift_results(pred_results, "PREDICTION")

    # Summary
    if pred_results.get("drift_detected", False):
        print("\n✅ SUCCESS: Population shift detected!")
        print(f"   JS Divergence: {pred_results['js_divergence']:.4f} (threshold: 0.1)")
        print(f"   This dramatic shift would trigger immediate investigation")

    return pred_results


def demo_scenario_4_catastrophic():
    """Demo: Multiple drifts combined"""
    print("\n" + "=" * 80)
    print("🎬 SCENARIO 4: CATASTROPHIC MULTI-DRIFT")
    print("=" * 80)
    print("Real-world: Multiple factors change simultaneously")
    print("Impact: Everything drifts at once!\n")

    # Load data
    print("Loading data...")
    reference_data = pd.read_csv("data/processed/train.csv").sample(
        1000, random_state=42
    )

    # Simulate catastrophic drift
    print("Simulating catastrophic multi-drift...")
    simulator = EnhancedDriftSimulator()
    drifted_data = simulator.simulate_catastrophic_drift(reference_data)

    # Generate report
    report = simulator.generate_comparison_report(reference_data, drifted_data)
    simulator.print_report(report)

    # Detect both data drift and prediction drift
    print("\n🔍 RUNNING COMPREHENSIVE DRIFT DETECTION...")

    # Data drift
    print("\n1️⃣  Checking DATA DRIFT...")
    data_detector = DataDriftDetector(reference_data)
    data_results = data_detector.detect_drift(drifted_data)
    print_drift_results(data_results, "DATA")

    # Prediction drift
    print("\n2️⃣  Checking PREDICTION DRIFT...")
    label_to_idx = {
        "Anxiety": 0,
        "Bipolar": 1,
        "Depression": 2,
        "Normal": 3,
        "Stress": 4,
        "Suicidal": 5,
    }

    ref_predictions = reference_data["label"].map(label_to_idx).values
    drift_predictions = drifted_data["label"].map(label_to_idx).values

    pred_detector = PredictionDriftDetector(
        reference_predictions=ref_predictions, class_names=list(label_to_idx.keys())
    )
    pred_results = pred_detector.detect_drift(drift_predictions)
    print_drift_results(pred_results, "PREDICTION")

    # Final summary
    print("\n" + "=" * 80)
    print("🚨 CATASTROPHIC DRIFT ASSESSMENT")
    print("=" * 80)

    data_drift = data_results.get("overall_drift", False)
    pred_drift = pred_results.get("drift_detected", False)

    if data_drift and pred_drift:
        print("🚨 CRITICAL: Both data and prediction drift detected!")
        print("\n📋 Recommended Actions:")
        print("   1. ⚠️  Immediate model retraining required")
        print("   2. 🔄 Consider rolling back to previous model version")
        print("   3. 🔍 Investigate root cause of drift")
        print("   4. 📊 Analyze user behavior changes")
        print("   5. 🛡️  Increase monitoring frequency")
    elif data_drift or pred_drift:
        print("⚠️  WARNING: Drift detected in one subsystem")
        print("   Continue monitoring closely")
    else:
        print("✅ System stable (no drift detected)")

    print("=" * 80)

    return data_results, pred_results


def demo_quick_verification():
    """Quick test to verify everything works"""
    print("\n" + "=" * 80)
    print("🧪 QUICK SYSTEM VERIFICATION")
    print("=" * 80)
    print("Running quick test to ensure all components work...\n")

    try:
        # Load small sample
        print("1. Loading data... ", end="")
        reference_data = pd.read_csv("data/processed/train.csv").sample(
            100, random_state=42
        )
        print("✓")

        # Create drift
        print("2. Simulating drift... ", end="")
        simulator = EnhancedDriftSimulator()
        drifted_data = simulator.simulate_extreme_length_drift(
            reference_data, factor=2.0, proportion=1.0
        )
        print("✓")

        # Detect
        print("3. Running detection... ", end="")
        detector = DataDriftDetector(reference_data)
        results = detector.detect_drift(drifted_data)
        print("✓")

        print("\n✅ All systems operational!")
        print(f"   - Samples: {results['n_samples']}")
        print(f"   - Drift detected: {'Yes 🚨' if results['overall_drift'] else 'No ✅'}")
        print(f"   - Drift score: {results['drift_score']:.4f}")

        print("\n" + "=" * 80)
        return True

    except Exception as e:
        print("❌")
        print(f"\n❌ System check failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all enhanced drift scenarios"""
    print("\n" + "🔬" * 40)
    print("         ENHANCED DRIFT DETECTION DEMONSTRATION")
    print("🔬" * 40)
    print("\nThis demo shows dramatic, realistic drift scenarios")
    print("that clearly trigger drift detection systems.")
    print("\nEach scenario simulates real-world production issues:")
    print("  • Extreme length changes (users write 3x longer)")
    print("  • Language evolution (Gen Z slang injection)")
    print("  • Population shifts (one class dominates)")
    print("  • Catastrophic multi-drift (everything at once)")

    # Quick verification
    print("\n" + "─" * 80)
    if not demo_quick_verification():
        print("\n❌ System check failed. Please fix errors before continuing.")
        return

    input("\n✅ System check passed! Press Enter to start full demonstration...")

    # Main scenarios
    scenarios = [
        ("Extreme Length Change", demo_scenario_1_extreme_length),
        ("Heavy Slang Injection", demo_scenario_2_heavy_slang),
        ("Population Collapse", demo_scenario_3_population_collapse),
        ("Catastrophic Multi-Drift", demo_scenario_4_catastrophic),
    ]

    for i, (name, func) in enumerate(scenarios, 1):
        print(f"\n{'='*80}")
        print(f"SCENARIO {i}/{len(scenarios)}: {name.upper()}")
        print(f"{'='*80}")

        try:
            func()
            print(f"\n✅ Scenario {i} completed successfully!")
        except Exception as e:
            print(f"\n❌ Error in scenario {i}: {e}")
            print("\nDebugging information:")
            import traceback

            traceback.print_exc()

            user_input = input("\nContinue to next scenario? (y/n): ")
            if user_input.lower() != "y":
                break

        if i < len(scenarios):
            user_input = input(
                "\nPress Enter to continue to next scenario (or 'q' to quit)..."
            )
            if user_input.lower() == "q":
                print("\nDemo interrupted by user.")
                break

    # Final summary
    print("\n" + "🎉" * 40)
    print("         DEMONSTRATION COMPLETE!")
    print("🎉" * 40)
    print("\n📊 Key Takeaways:")
    print("   ✅ Dramatic changes are reliably detected")
    print("   ✅ Multiple drift types can be monitored simultaneously")
    print("   ✅ System provides clear, actionable alerts")
    print("   ✅ Statistical rigor with industry-standard methods")
    print("   ✅ Production-ready monitoring capabilities")

    print("\n💡 This demonstrates:")
    print("   • Early warning system for model degradation")
    print("   • Protection against silent failures")
    print("   • Automated quality assurance")
    print("   • Real-world deployment readiness")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

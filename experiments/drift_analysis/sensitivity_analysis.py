"""
Drift Detection Sensitivity Analysis

Tests: At what drift intensity does the detector trigger?
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
from src.monitoring.data_drift import DataDriftDetector
from src.monitoring.drift_simulator_v2 import EnhancedDriftSimulator
import json
from datetime import datetime


def test_length_sensitivity():
    """
    Test: How much length change triggers detection?
    
    Test factors: 1.1x, 1.2x, 1.3x, ..., 3.0x
    Find: Minimum factor that triggers drift
    """
    print("\n" + "="*80)
    print("🔬 TEST 1: LENGTH DRIFT SENSITIVITY")
    print("="*80)
    print("Question: At what length change does detector trigger?\n")
    
    # Load reference data
    reference_data = pd.read_csv('data/processed/train.csv').sample(500, random_state=42)
    detector = DataDriftDetector(reference_data)
    
    # Test different length factors
    factors = [1.1, 1.2, 1.3, 1.4, 1.5, 1.75, 2.0, 2.5, 3.0]
    results = []
    
    simulator = EnhancedDriftSimulator()
    
    for factor in factors:
        print(f"Testing {factor}x length increase...", end=' ')
        
        # Create drifted data
        drifted_data = simulator.simulate_extreme_length_drift(
            reference_data.copy(),
            factor=factor,
            proportion=1.0
        )
        
        # Detect drift
        drift_results = detector.detect_drift(drifted_data)
        
        # Store results
        results.append({
            'factor': factor,
            'drift_detected': drift_results['overall_drift'],
            'drift_score': drift_results['drift_score'],
            'features_with_drift': drift_results['features_with_drift']
        })
        
        status = "🚨 DETECTED" if drift_results['overall_drift'] else "✅ NOT DETECTED"
        print(f"{status} (PSI: {drift_results['drift_score']:.4f})")
    
    # Find threshold
    print("\n" + "-"*80)
    print("📊 SENSITIVITY ANALYSIS RESULTS:")
    print("-"*80)
    
    first_detection = None
    for r in results:
        if r['drift_detected'] and first_detection is None:
            first_detection = r['factor']
        
        print(f"  {r['factor']:.1f}x: PSI={r['drift_score']:.4f}, "
              f"Detected={r['drift_detected']}, "
              f"Features={r['features_with_drift']}")
    
    if first_detection:
        print(f"\n✅ Detection Threshold: ~{first_detection}x length change")
        print(f"   (Detector triggers when text becomes {first_detection}x original length)")
    
    # Save results
    output_file = 'experiments/results/sensitivity_length.json'
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'test': 'length_sensitivity',
            'threshold_found': first_detection,
            'results': results
        }, f, indent=2)
    
    print(f"\n💾 Saved to: {output_file}")
    
    return results


def test_slang_sensitivity():
    """
    Test: How much slang triggers detection?
    
    Test intensities: 10%, 20%, 30%, ..., 100%
    """
    print("\n" + "="*80)
    print("🔬 TEST 2: SLANG DRIFT SENSITIVITY")
    print("="*80)
    print("Question: How much slang substitution triggers detection?\n")
    
    # Load reference data
    reference_data = pd.read_csv('data/processed/train.csv').sample(500, random_state=43)
    detector = DataDriftDetector(reference_data)
    
    # Test different slang intensities
    intensities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    results = []
    
    simulator = EnhancedDriftSimulator()
    
    for intensity in intensities:
        print(f"Testing {intensity*100:.0f}% slang intensity...", end=' ')
        
        # Create drifted data
        drifted_data = simulator.simulate_heavy_slang(
            reference_data.copy(),
            intensity=intensity
        )
        
        # Detect drift
        drift_results = detector.detect_drift(drifted_data)
        
        # Store results
        results.append({
            'intensity': intensity,
            'drift_detected': drift_results['overall_drift'],
            'drift_score': drift_results['drift_score'],
            'features_with_drift': drift_results['features_with_drift']
        })
        
        status = "🚨 DETECTED" if drift_results['overall_drift'] else "✅ NOT DETECTED"
        print(f"{status} (PSI: {drift_results['drift_score']:.4f})")
    
    # Find threshold
    print("\n" + "-"*80)
    print("📊 SLANG SENSITIVITY RESULTS:")
    print("-"*80)
    
    first_detection = None
    for r in results:
        if r['drift_detected'] and first_detection is None:
            first_detection = r['intensity']
        
        print(f"  {r['intensity']*100:3.0f}%: PSI={r['drift_score']:.4f}, "
              f"Detected={r['drift_detected']}, "
              f"Features={r['features_with_drift']}")
    
    if first_detection:
        print(f"\n✅ Detection Threshold: ~{first_detection*100:.0f}% slang substitution")
    else:
        print(f"\n🤔 Interesting: Slang drift not detected at any intensity!")
        print(f"   Insight: Slang changes vocabulary but not statistical features")
        print(f"   This is why we need multiple drift detector types!")
    
    # Save results
    output_file = 'experiments/results/sensitivity_slang.json'
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'test': 'slang_sensitivity',
            'threshold_found': first_detection,
            'results': results,
            'insight': 'Slang drift may not trigger statistical features' if not first_detection else None
        }, f, indent=2)
    
    print(f"\n💾 Saved to: {output_file}")
    
    return results


def test_population_sensitivity():
    """
    Test: How much class imbalance triggers detection?
    
    Test dominance: 40%, 50%, 60%, ..., 90%
    """
    print("\n" + "="*80)
    print("🔬 TEST 3: POPULATION DRIFT SENSITIVITY")
    print("="*80)
    print("Question: How much class imbalance triggers detection?\n")
    
    # Load reference data
    reference_data = pd.read_csv('data/processed/train.csv').sample(500, random_state=44)
    
    # For prediction drift, we need class indices
    label_to_idx = {
        'Anxiety': 0, 'Bipolar': 1, 'Depression': 2,
        'Normal': 3, 'Stress': 4, 'Suicidal': 5
    }
    
    ref_predictions = reference_data['label'].map(label_to_idx).values
    
    from src.monitoring.prediction_drift import PredictionDriftDetector
    detector = PredictionDriftDetector(
        reference_predictions=ref_predictions,
        class_names=list(label_to_idx.keys())
    )
    
    # Test different dominance levels
    dominance_levels = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    results = []
    
    simulator = EnhancedDriftSimulator()
    
    original_anxiety_pct = (reference_data['label'] == 'Anxiety').mean()
    print(f"Original Anxiety: {original_anxiety_pct*100:.1f}%\n")
    
    for dominance in dominance_levels:
        print(f"Testing {dominance*100:.0f}% Anxiety dominance...", end=' ')
        
        # Create drifted data
        drifted_data = simulator.simulate_population_collapse(
            reference_data.copy(),
            dominant_class='Anxiety',
            dominance=dominance
        )
        
        # Get predictions
        curr_predictions = drifted_data['label'].map(label_to_idx).values
        
        # Detect drift
        drift_results = detector.detect_drift(curr_predictions)
        
        # Store results
        drift_detected = drift_results['drift_detected']
        if hasattr(drift_detected, 'item'):  # numpy bool
            drift_detected = drift_detected.item()
        
        results.append({
            'dominance': float(dominance),
            'drift_detected': bool(drift_detected),
            'js_divergence': float(drift_results['js_divergence']),
            'wasserstein_distance': float(drift_results['wasserstein_distance'])
        })
        
        status = "🚨 DETECTED" if drift_results['drift_detected'] else "✅ NOT DETECTED"
        print(f"{status} (JS: {drift_results['js_divergence']:.4f})")
    
    # Find threshold
    print("\n" + "-"*80)
    print("📊 POPULATION SENSITIVITY RESULTS:")
    print("-"*80)
    
    first_detection = None
    for r in results:
        if r['drift_detected'] and first_detection is None:
            first_detection = r['dominance']
        
        print(f"  {r['dominance']*100:3.0f}%: JS={r['js_divergence']:.4f}, "
              f"Detected={r['drift_detected']}")
    
    if first_detection:
        change_from_original = (first_detection - original_anxiety_pct) / original_anxiety_pct * 100
        print(f"\n✅ Detection Threshold: ~{first_detection*100:.0f}% dominance")
        print(f"   (Detector triggers when Anxiety increases by {change_from_original:.0f}%)")
    
    # Save results
    output_file = 'experiments/results/sensitivity_population.json'
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'test': 'population_sensitivity',
            'original_anxiety_pct': original_anxiety_pct,
            'threshold_found': first_detection,
            'results': results
        }, f, indent=2)
    
    print(f"\n💾 Saved to: {output_file}")
    
    return results

def generate_sensitivity_summary():
    """
    Generate summary report of all sensitivity tests
    """
    print("\n" + "="*80)
    print("📊 SENSITIVITY ANALYSIS SUMMARY")
    print("="*80)
    
    # Load all results
    import glob
    result_files = glob.glob('experiments/results/sensitivity_*.json')
    
    if not result_files:
        print("No results found. Run tests first.")
        return
    
    print("\n🎯 Detection Thresholds:\n")
    
    for file in sorted(result_files):
        try:
            with open(file, 'r') as f:
                data = json.load(f)
            
            test_name = data['test'].replace('_', ' ').title()
            threshold = data.get('threshold_found')
            
            if threshold:
                if 'length' in data['test']:
                    print(f"  • Length Drift: Triggers at ~{threshold}x length change")
                elif 'slang' in data['test']:
                    print(f"  • Slang Drift: Triggers at ~{threshold*100:.0f}% substitution")
                elif 'population' in data['test']:
                    print(f"  • Population Drift: Triggers at ~{threshold*100:.0f}% dominance")
            else:
                print(f"  • {test_name}: No threshold found (detector not sensitive to this drift type)")
        
        except json.JSONDecodeError as e:
            print(f"  ⚠️  Error reading {file}: {e}")
            print(f"     File may be corrupted, skipping...")
            continue
        except Exception as e:
            print(f"  ⚠️  Error processing {file}: {e}")
            continue
    
    print("\n💡 Key Insights:")
    print("  1. Different drift types have different detection sensitivities")
    print("  2. Statistical features catch length/population changes well")
    print("  3. Linguistic changes (slang) may need different detection methods")
    print("  4. Multi-detector approach is essential for comprehensive monitoring")
    
    print("\n" + "="*80)

def main():
    """
    Run complete sensitivity analysis
    """
    print("\n" + "🔬"*40)
    print("    DRIFT DETECTION SENSITIVITY ANALYSIS")
    print("🔬"*40)
    print("\nGoal: Determine detection thresholds for different drift types\n")
    
    # Create results directory
    os.makedirs('experiments/results', exist_ok=True)
    
    tests = [
        ("Length Drift Sensitivity", test_length_sensitivity),
        ("Slang Drift Sensitivity", test_slang_sensitivity),
        ("Population Drift Sensitivity", test_population_sensitivity),
    ]
    
    for i, (name, func) in enumerate(tests, 1):
        print(f"\n{'='*80}")
        print(f"RUNNING TEST {i}/{len(tests)}: {name}")
        print(f"{'='*80}")
        
        try:
            func()
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
        
        if i < len(tests):
            input("\nPress Enter to continue to next test...")
    
    # Generate summary
    generate_sensitivity_summary()
    
    print("\n" + "🎉"*40)
    print("    SENSITIVITY ANALYSIS COMPLETE!")
    print("🎉"*40)
    print("\n✅ Results saved to: experiments/results/")
    print("✅ Ready for report inclusion")


if __name__ == "__main__":
    main()
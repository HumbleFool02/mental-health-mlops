"""
False Positive Rate Analysis

Tests: How often does detector trigger false alarms on stable data?
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
from src.monitoring.data_drift import DataDriftDetector
from src.monitoring.prediction_drift import PredictionDriftDetector
import json
from datetime import datetime
from tqdm import tqdm


def test_data_drift_false_positives(n_runs=100):
    """
    Test false positive rate for data drift detector
    
    Method: Sample from same distribution repeatedly
    Expected: Should rarely detect drift (< 5%)
    """
    print("\n" + "="*80)
    print("🔬 TEST: DATA DRIFT FALSE POSITIVE RATE")
    print("="*80)
    print(f"\nRunning {n_runs} trials on stable data (same distribution)...\n")
    
    # Load reference data
    full_data = pd.read_csv('data/processed/train.csv')
    reference_data = full_data.sample(500, random_state=42)
    
    # Create detector
    detector = DataDriftDetector(reference_data)
    
    # Track results
    false_positives = 0
    drift_scores = []
    features_triggered = []
    
    print("Running trials...")
    for i in tqdm(range(n_runs), desc="Progress"):
        # Sample from SAME distribution (should NOT drift)
        test_data = full_data.sample(500, random_state=1000+i)
        
        # Run detection
        results = detector.detect_drift(test_data)
        
        # Check if false positive
        if results['overall_drift']:
            false_positives += 1
            features_triggered.append(results['features_with_drift'])
        
        drift_scores.append(results['drift_score'])
    
    # Calculate statistics
    fp_rate = false_positives / n_runs
    avg_score = np.mean(drift_scores)
    max_score = np.max(drift_scores)
    std_score = np.std(drift_scores)
    
    # Print results
    print("\n" + "-"*80)
    print("📊 RESULTS:")
    print("-"*80)
    print(f"  Total runs: {n_runs}")
    print(f"  False positives: {false_positives}")
    print(f"  False positive rate: {fp_rate*100:.1f}%")
    print(f"\n  Drift Score Statistics:")
    print(f"    Mean: {avg_score:.4f}")
    print(f"    Std:  {std_score:.4f}")
    print(f"    Max:  {max_score:.4f}")
    
    if false_positives > 0:
        print(f"\n  False Alarm Details:")
        print(f"    Average features triggered: {np.mean(features_triggered):.1f}")
        print(f"    Max features triggered: {max(features_triggered)}")
    
    # Interpretation
    print("\n  Interpretation:")
    if fp_rate < 0.05:
        print("    ✅ EXCELLENT: FP rate < 5% (well-calibrated)")
    elif fp_rate < 0.10:
        print("    ✅ GOOD: FP rate < 10% (acceptable)")
    else:
        print("    ⚠️  WARNING: FP rate > 10% (may need threshold tuning)")
    
    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'test': 'data_drift_false_positives',
        'n_runs': n_runs,
        'false_positives': int(false_positives),
        'false_positive_rate': float(fp_rate),
        'drift_scores': {
            'mean': float(avg_score),
            'std': float(std_score),
            'max': float(max_score),
            'all_scores': [float(s) for s in drift_scores]
        }
    }
    
    output_file = 'experiments/results/false_positive_data_drift.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n💾 Saved to: {output_file}")
    print("-"*80)
    
    return output


def test_prediction_drift_false_positives(n_runs=100):
    """
    Test false positive rate for prediction drift detector
    
    Method: Sample predictions from same distribution repeatedly
    Expected: Should rarely detect drift (< 5%)
    """
    print("\n" + "="*80)
    print("🔬 TEST: PREDICTION DRIFT FALSE POSITIVE RATE")
    print("="*80)
    print(f"\nRunning {n_runs} trials on stable predictions...\n")
    
    # Load reference data
    full_data = pd.read_csv('data/processed/train.csv')
    reference_data = full_data.sample(500, random_state=42)
    
    # Map labels to indices
    label_to_idx = {
        'Anxiety': 0, 'Bipolar': 1, 'Depression': 2,
        'Normal': 3, 'Stress': 4, 'Suicidal': 5
    }
    
    ref_predictions = reference_data['label'].map(label_to_idx).values
    
    # Create detector
    detector = PredictionDriftDetector(
        reference_predictions=ref_predictions,
        class_names=list(label_to_idx.keys())
    )
    
    # Track results
    false_positives = 0
    js_scores = []
    wasserstein_scores = []
    
    print("Running trials...")
    for i in tqdm(range(n_runs), desc="Progress"):
        # Sample from SAME distribution (should NOT drift)
        test_data = full_data.sample(500, random_state=2000+i)
        test_predictions = test_data['label'].map(label_to_idx).values
        
        # Run detection
        results = detector.detect_drift(test_predictions)
        
        # Check if false positive
        drift_detected = results['drift_detected']
        if hasattr(drift_detected, 'item'):
            drift_detected = drift_detected.item()
        
        if drift_detected:
            false_positives += 1
        
        js_scores.append(results['js_divergence'])
        wasserstein_scores.append(results['wasserstein_distance'])
    
    # Calculate statistics
    fp_rate = false_positives / n_runs
    avg_js = np.mean(js_scores)
    max_js = np.max(js_scores)
    std_js = np.std(js_scores)
    
    # Print results
    print("\n" + "-"*80)
    print("📊 RESULTS:")
    print("-"*80)
    print(f"  Total runs: {n_runs}")
    print(f"  False positives: {false_positives}")
    print(f"  False positive rate: {fp_rate*100:.1f}%")
    print(f"\n  JS Divergence Statistics:")
    print(f"    Mean: {avg_js:.4f}")
    print(f"    Std:  {std_js:.4f}")
    print(f"    Max:  {max_js:.4f}")
    print(f"    Threshold: 0.1")
    
    # Interpretation
    print("\n  Interpretation:")
    if fp_rate < 0.05:
        print("    ✅ EXCELLENT: FP rate < 5% (well-calibrated)")
    elif fp_rate < 0.10:
        print("    ✅ GOOD: FP rate < 10% (acceptable)")
    else:
        print("    ⚠️  WARNING: FP rate > 10% (may need threshold tuning)")
    
    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'test': 'prediction_drift_false_positives',
        'n_runs': n_runs,
        'false_positives': int(false_positives),
        'false_positive_rate': float(fp_rate),
        'js_divergence': {
            'mean': float(avg_js),
            'std': float(std_js),
            'max': float(max_js),
            'threshold': 0.1,
            'all_scores': [float(s) for s in js_scores]
        },
        'wasserstein_distance': {
            'mean': float(np.mean(wasserstein_scores)),
            'std': float(np.std(wasserstein_scores)),
            'max': float(np.max(wasserstein_scores))
        }
    }
    
    output_file = 'experiments/results/false_positive_prediction_drift.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n💾 Saved to: {output_file}")
    print("-"*80)
    
    return output


def generate_false_positive_summary():
    """
    Generate summary of false positive tests
    """
    print("\n" + "="*80)
    print("📊 FALSE POSITIVE ANALYSIS SUMMARY")
    print("="*80)
    
    import glob
    result_files = glob.glob('experiments/results/false_positive_*.json')
    
    if not result_files:
        print("\nNo results found. Run tests first.")
        return
    
    print("\n🎯 FALSE POSITIVE RATES:\n")
    
    total_fp_rate = []
    
    for file in sorted(result_files):
        try:
            with open(file, 'r') as f:
                data = json.load(f)
            
            test_name = data['test'].replace('_', ' ').title()
            fp_rate = data['false_positive_rate']
            n_runs = data['n_runs']
            false_positives = data['false_positives']
            
            total_fp_rate.append(fp_rate)
            
            status = "✅" if fp_rate < 0.05 else ("⚠️" if fp_rate < 0.10 else "🚨")
            
            print(f"  {status} {test_name}:")
            print(f"     Rate: {fp_rate*100:.1f}% ({false_positives}/{n_runs} runs)")
            
            # Show score statistics
            if 'drift_scores' in data:
                scores = data['drift_scores']
                print(f"     PSI: μ={scores['mean']:.4f}, σ={scores['std']:.4f}, max={scores['max']:.4f}")
            elif 'js_divergence' in data:
                scores = data['js_divergence']
                print(f"     JS:  μ={scores['mean']:.4f}, σ={scores['std']:.4f}, max={scores['max']:.4f}")
            print()
        
        except Exception as e:
            print(f"  ⚠️  Error reading {file}: {e}\n")
            continue
    
    if total_fp_rate:
        avg_fp = np.mean(total_fp_rate)
        print(f"  Overall Average FP Rate: {avg_fp*100:.1f}%\n")
    
    print("💡 KEY INSIGHTS:\n")
    print("  1. False positive rates indicate detector reliability")
    print("  2. Rates < 5% considered excellent for production")
    print("  3. Low FP rate proves thresholds are well-calibrated")
    print("  4. System won't 'cry wolf' - alerts are meaningful")
    
    print("\n  For Production:")
    if avg_fp := (np.mean(total_fp_rate) if total_fp_rate else 0):
        if avg_fp < 0.05:
            print("    ✅ System ready for deployment (FP < 5%)")
        elif avg_fp < 0.10:
            print("    ✅ Acceptable for deployment (FP < 10%)")
        else:
            print("    ⚠️  Consider threshold tuning before deployment")
    
    print("\n" + "="*80)


def main():
    """
    Run complete false positive analysis
    """
    print("\n" + "🔬"*40)
    print("    FALSE POSITIVE RATE ANALYSIS")
    print("🔬"*40)
    print("\nGoal: Determine reliability - how often do we get false alarms?")
    print("Method: Test detector on stable data (same distribution)")
    print("Expected: < 5% false positive rate\n")
    
    # Create results directory
    os.makedirs('experiments/results', exist_ok=True)
    
    # Ask user for number of runs
    print("Recommended: 100 runs (~2-3 minutes)")
    print("Quick test: 50 runs (~1 minute)")
    print("Thorough: 200 runs (~5 minutes)")
    
    try:
        n_runs = int(input("\nNumber of runs [100]: ") or "100")
    except ValueError:
        n_runs = 100
        print(f"Using default: {n_runs} runs")
    
    # Run tests
    tests = [
        ("Data Drift False Positives", lambda: test_data_drift_false_positives(n_runs)),
        ("Prediction Drift False Positives", lambda: test_prediction_drift_false_positives(n_runs)),
    ]
    
    results = []
    
    for i, (name, func) in enumerate(tests, 1):
        print(f"\n{'='*80}")
        print(f"RUNNING TEST {i}/{len(tests)}: {name}")
        print(f"{'='*80}")
        
        try:
            result = func()
            results.append(result)
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
        
        if i < len(tests):
            input("\nPress Enter to continue...")
    
    # Generate summary
    generate_false_positive_summary()
    
    print("\n" + "🎉"*40)
    print("    FALSE POSITIVE ANALYSIS COMPLETE!")
    print("🎉"*40)
    print("\n✅ Results saved to: experiments/results/false_positive_*.json")
    print("✅ Use these findings to demonstrate system reliability")


if __name__ == "__main__":
    main()
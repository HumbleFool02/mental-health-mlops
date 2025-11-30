"""
Feature Attribution Analysis

Determines which features detect which drift types
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
from src.monitoring.data_drift import DataDriftDetector
from src.monitoring.drift_simulator_v2 import EnhancedDriftSimulator
import json
from datetime import datetime


def analyze_feature_response(drift_type: str, drift_func, drift_params: dict):
    """
    Analyze how each feature responds to a specific drift type
    
    Args:
        drift_type: Name of drift (e.g., "length_drift")
        drift_func: Function to generate drift
        drift_params: Parameters for drift function
    """
    print(f"\n{'='*80}")
    print(f"🔬 ANALYZING: {drift_type.replace('_', ' ').title()}")
    print(f"{'='*80}\n")
    
    # Load reference data
    reference_data = pd.read_csv('data/processed/train.csv').sample(500, random_state=42)
    
    # Generate drifted data
    simulator = EnhancedDriftSimulator()
    drifted_data = drift_func(reference_data.copy(), **drift_params)
    
    # Run drift detection
    detector = DataDriftDetector(reference_data)
    results = detector.detect_drift(drifted_data)
    
    # Extract feature-level results
    feature_analysis = {}
    
    if 'features' in results:
        for feature, metrics in results['features'].items():
            # Handle string "False"/"True" or boolean
            drift_detected = metrics.get('drift_detected', 'False')
            if isinstance(drift_detected, str):
                drift_detected = drift_detected.lower() == 'true'
            
            feature_analysis[feature] = {
                'psi': float(metrics.get('psi', 0)),
                'ks_statistic': float(metrics.get('ks_statistic', 0)),
                'ks_pvalue': float(metrics.get('ks_pvalue', 1.0)),
                'drift_detected': bool(drift_detected),
                'mean_change_pct': float(metrics.get('mean_diff_pct', 0))
            }
    
    # Print results
    print("📊 Feature Response Analysis:\n")
    
    # Sort by PSI (most sensitive first)
    sorted_features = sorted(
        feature_analysis.items(), 
        key=lambda x: x[1]['psi'], 
        reverse=True
    )
    
    for feature, metrics in sorted_features:
        psi = metrics['psi']
        change = metrics['mean_change_pct']
        detected = metrics['drift_detected']
        
        # Sensitivity indicator
        if psi > 0.2:
            sensitivity = "🔴 HIGH"
        elif psi > 0.1:
            sensitivity = "🟡 MODERATE"
        elif psi > 0.05:
            sensitivity = "🟢 LOW"
        else:
            sensitivity = "⚪ NONE"
        
        status = "✓ DETECTED" if detected else "✗ not detected"
        
        print(f"  {feature:20s}: PSI={psi:6.4f} {sensitivity:15s} "
              f"Change={change:+7.1f}% [{status}]")
    
    # Summary statistics
    total_features = len(feature_analysis)
    detected_features = sum(1 for m in feature_analysis.values() if m['drift_detected'])
    avg_psi = np.mean([m['psi'] for m in feature_analysis.values()])
    max_psi = max([m['psi'] for m in feature_analysis.values()])
    
    print(f"\n📈 Summary:")
    print(f"  Total features: {total_features}")
    print(f"  Features with drift: {detected_features}")
    print(f"  Average PSI: {avg_psi:.4f}")
    print(f"  Max PSI: {max_psi:.4f}")
    
    return {
        'drift_type': drift_type,
        'feature_analysis': feature_analysis,
        'summary': {
            'total_features': total_features,
            'detected_features': detected_features,
            'avg_psi': float(avg_psi),
            'max_psi': float(max_psi)
        }
    }


def test_length_drift_attribution():
    """Which features detect length drift?"""
    print("\n" + "="*80)
    print("🎯 TEST 1: LENGTH DRIFT FEATURE ATTRIBUTION")
    print("="*80)
    
    simulator = EnhancedDriftSimulator()
    
    result = analyze_feature_response(
        drift_type="length_drift",
        drift_func=simulator.simulate_extreme_length_drift,
        drift_params={'factor': 2.0, 'proportion': 1.0}
    )
    
    # Save results
    output_file = 'experiments/results/attribution_length.json'
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n💾 Saved to: {output_file}")
    
    return result


def test_slang_drift_attribution():
    """Which features detect slang drift?"""
    print("\n" + "="*80)
    print("🎯 TEST 2: SLANG DRIFT FEATURE ATTRIBUTION")
    print("="*80)
    
    simulator = EnhancedDriftSimulator()
    
    result = analyze_feature_response(
        drift_type="slang_drift",
        drift_func=simulator.simulate_heavy_slang,
        drift_params={'intensity': 0.8}
    )
    
    # Save results
    output_file = 'experiments/results/attribution_slang.json'
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n💾 Saved to: {output_file}")
    
    return result


def test_formality_drift_attribution():
    """Which features detect formality shift?"""
    print("\n" + "="*80)
    print("🎯 TEST 3: FORMALITY DRIFT FEATURE ATTRIBUTION")
    print("="*80)
    
    simulator = EnhancedDriftSimulator()
    
    result = analyze_feature_response(
        drift_type="formality_drift",
        drift_func=simulator.simulate_formality_shift,
        drift_params={'direction': 'informal', 'intensity': 0.8}
    )
    
    # Save results
    output_file = 'experiments/results/attribution_formality.json'
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n💾 Saved to: {output_file}")
    
    return result


def test_multi_drift_attribution():
    """Which features detect catastrophic multi-drift?"""
    print("\n" + "="*80)
    print("🎯 TEST 4: MULTI-DRIFT FEATURE ATTRIBUTION")
    print("="*80)
    
    simulator = EnhancedDriftSimulator()
    
    result = analyze_feature_response(
        drift_type="multi_drift",
        drift_func=simulator.simulate_catastrophic_drift,
        drift_params={}
    )
    
    # Save results
    output_file = 'experiments/results/attribution_multi.json'
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n💾 Saved to: {output_file}")
    
    return result


def generate_attribution_summary():
    """
    Generate comprehensive feature attribution summary
    """
    print("\n" + "="*80)
    print("📊 FEATURE ATTRIBUTION SUMMARY")
    print("="*80)
    
    import glob
    result_files = glob.glob('experiments/results/attribution_*.json')
    
    if not result_files:
        print("No results found. Run tests first.")
        return
    
    # Load all results
    all_results = {}
    for file in sorted(result_files):
        try:
            with open(file, 'r') as f:
                data = json.load(f)
            drift_type = data['drift_type']
            all_results[drift_type] = data
        except Exception as e:
            print(f"⚠️  Error loading {file}: {e}")
            continue
    
    if not all_results:
        print("No valid results found.")
        return
    
    # Create feature comparison table
    print("\n📋 FEATURE SENSITIVITY BY DRIFT TYPE:\n")
    
    # Get all features
    features = set()
    for result in all_results.values():
        if 'feature_analysis' in result:
            features.update(result['feature_analysis'].keys())
    
    features = sorted(features)
    
    # Print header
    print(f"{'Feature':<20s} | ", end='')
    for drift_type in sorted(all_results.keys()):
        print(f"{drift_type.replace('_', ' ').title():<15s} | ", end='')
    print()
    print("-" * (20 + 18 * len(all_results)))
    
    # Print each feature
    for feature in features:
        print(f"{feature:<20s} | ", end='')
        
        for drift_type in sorted(all_results.keys()):
            result = all_results[drift_type]
            if 'feature_analysis' in result and feature in result['feature_analysis']:
                psi = result['feature_analysis'][feature]['psi']
                
                # Format with sensitivity indicator
                if psi > 0.2:
                    indicator = "🔴"
                elif psi > 0.1:
                    indicator = "🟡"
                elif psi > 0.05:
                    indicator = "🟢"
                else:
                    indicator = "⚪"
                
                print(f"{indicator} {psi:5.3f}        | ", end='')
            else:
                print(f"  N/A           | ", end='')
        
        print()
    
    # Key insights
    print("\n💡 KEY INSIGHTS:\n")
    
    # Find most sensitive feature per drift type
    for drift_type, result in sorted(all_results.items()):
        if 'feature_analysis' not in result:
            continue
        
        # Sort by PSI
        sorted_features = sorted(
            result['feature_analysis'].items(),
            key=lambda x: x[1]['psi'],
            reverse=True
        )
        
        if sorted_features:
            top_feature = sorted_features[0]
            print(f"  • {drift_type.replace('_', ' ').title()}:")
            print(f"    Most sensitive: {top_feature[0]} (PSI={top_feature[1]['psi']:.3f})")
            
            # Show top 2-3 features
            if len(sorted_features) > 1:
                print(f"    Also responds: ", end='')
                for feat, metrics in sorted_features[1:3]:
                    if metrics['psi'] > 0.05:
                        print(f"{feat} (PSI={metrics['psi']:.3f}), ", end='')
                print()
    
    # Find robust features (low PSI across all drifts)
    print("\n  • Robust Features (stable across drift types):")
    for feature in features:
        psi_values = []
        for result in all_results.values():
            if 'feature_analysis' in result and feature in result['feature_analysis']:
                psi_values.append(result['feature_analysis'][feature]['psi'])
        
        if psi_values:
            avg_psi = np.mean(psi_values)
            if avg_psi < 0.05:
                print(f"    - {feature}: Average PSI={avg_psi:.4f}")
    
    # Overall conclusions
    print("\n  • Drift Detection Strategy:")
    print("    1. Multiple features needed - no single feature catches all drift")
    print("    2. Structural features (length, count) detect magnitude changes")
    print("    3. Linguistic features less sensitive to vocabulary changes")
    print("    4. Combined monitoring provides comprehensive coverage")
    
    print("\n" + "="*80)
    
    # Save summary
    summary_file = 'experiments/results/attribution_summary.json'
    with open(summary_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'all_results': all_results,
            'features_analyzed': list(features)
        }, f, indent=2)
    
    print(f"\n💾 Summary saved to: {summary_file}")


def main():
    """
    Run complete feature attribution analysis
    """
    print("\n" + "🔬"*40)
    print("    FEATURE ATTRIBUTION ANALYSIS")
    print("🔬"*40)
    print("\nGoal: Determine which features detect which drift types\n")
    
    # Create results directory
    os.makedirs('experiments/results', exist_ok=True)
    
    tests = [
        ("Length Drift", test_length_drift_attribution),
        ("Slang Drift", test_slang_drift_attribution),
        ("Formality Drift", test_formality_drift_attribution),
        ("Multi-Drift (Catastrophic)", test_multi_drift_attribution),
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
    generate_attribution_summary()
    
    print("\n" + "🎉"*40)
    print("    FEATURE ATTRIBUTION COMPLETE!")
    print("🎉"*40)
    print("\n✅ Results saved to: experiments/results/attribution_*.json")
    print("✅ Use these findings to explain which features detect which drift")


if __name__ == "__main__":
    main()
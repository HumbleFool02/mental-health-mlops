"""
Analyze LLM-generated drift data
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.monitoring.data_drift import DataDriftDetector


def analyze_drift(original_file, transformed_file, drift_type):
    """
    Analyze drift between original and LLM-transformed texts
    """
    print("\n" + "=" * 80)
    print(f"🔍 ANALYZING: {drift_type}")
    print("=" * 80)

    # Load data
    print("\n1️⃣  Loading data...")
    try:
        original_df = pd.read_csv(original_file)
        print(f"   ✓ Loaded {len(original_df)} original samples")
    except FileNotFoundError:
        print(f"   ❌ File not found: {original_file}")
        return

    try:
        # Load transformed (might be .txt or .csv)
        if transformed_file.endswith(".txt"):
            with open(transformed_file, "r") as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
            # Remove numbering (1., 2., etc.)
            transformed_texts = []
            for line in lines:
                # Try to remove leading number
                if line[0].isdigit() and ". " in line:
                    transformed_texts.append(line.split(". ", 1)[1])
                else:
                    transformed_texts.append(line)
            transformed_df = pd.DataFrame({"text": transformed_texts})
        else:
            transformed_df = pd.read_csv(transformed_file)

        print(f"   ✓ Loaded {len(transformed_df)} transformed samples")
    except FileNotFoundError:
        print(f"   ❌ File not found: {transformed_file}")
        print("   Create this file by:")
        print("   1. Go to claude.ai or ChatGPT")
        print("   2. Use prompt from data/llm_experiments/prompts.md")
        print("   3. Save output to {transformed_file}")
        return

    # Statistics
    print("\n2️⃣  Comparing statistics...")
    orig_lengths = original_df["text"].str.len()
    trans_lengths = transformed_df["text"].str.len()

    print(f"   Original avg length: {orig_lengths.mean():.0f} chars")
    print(f"   Transformed avg length: {trans_lengths.mean():.0f} chars")
    print(
        f"   Change: {((trans_lengths.mean() / orig_lengths.mean()) - 1) * 100:+.1f}%"
    )

    # Show examples
    print("\n3️⃣  Examples...")
    for i in range(min(3, len(original_df))):
        print(f"\n   Example {i + 1}:")
        print(f"   Original ({len(original_df.iloc[i]['text'])} chars):")
        print(f"     '{original_df.iloc[i]['text'][:100]}...'")
        if i < len(transformed_df):
            print(f"   Transformed ({len(transformed_df.iloc[i]['text'])} chars):")
            print(f"     '{transformed_df.iloc[i]['text'][:100]}...'")

    # Run drift detection
    print("\n4️⃣  Running drift detection...")
    detector = DataDriftDetector(original_df)
    results = detector.detect_drift(transformed_df)

    # Show results
    drift_status = "🚨 DRIFT DETECTED" if results["overall_drift"] else "✅ NO DRIFT"
    print(f"\n   {drift_status}")
    print(f"   Drift Score (PSI): {results['drift_score']:.4f}")
    print(f"   Features with drift: {results['features_with_drift']}")

    if results.get("features"):
        print("\n   Per-feature:")
        for feat, metrics in results["features"].items():
            drift = metrics.get("drift_detected", "False")
            if isinstance(drift, str):
                drift = drift.lower() == "true"
            icon = "🚨" if drift else "✅"
            psi = metrics.get("psi", 0)
            print(f"     {icon} {feat}: PSI={psi:.4f}")

    print("\n" + "=" * 80)

    return results


def main():
    """Run analysis on all available LLM-generated data"""
    print("\n" + "🤖" * 40)
    print("    LLM-GENERATED DRIFT ANALYSIS")
    print("🤖" * 40)

    experiments = [
        (
            "Length Drift (3x)",
            "data/llm_experiments/original_samples.csv",
            "data/llm_experiments/promptsOutputs1/length_drift.txt",
        ),
        (
            "Slang Drift (Gen Z)",
            "data/llm_experiments/original_samples.csv",
            "data/llm_experiments/promptsOutputs1/slang_drift.txt",
        ),
        (
            "Formality Drift (Clinical)",
            "data/llm_experiments/original_samples.csv",
            "data/llm_experiments/promptsOutputs1/formality_drift.txt",
        ),
        (
            "Adversarial Drift (Subtle)",
            "data/llm_experiments/original_samples.csv",
            "data/llm_experiments/promptsOutputs1/adversarial_drift.txt",
        ),
        (
            "Multi-Drift (Combined)",
            "data/llm_experiments/original_samples.csv",
            "data/llm_experiments/promptsOutputs1/multi_drift.txt",
        ),
    ]

    results_summary = []

    for name, orig_file, trans_file in experiments:
        result = analyze_drift(orig_file, trans_file, name)
        if result:
            results_summary.append(
                {
                    "drift_type": name,
                    "drift_detected": result.get("overall_drift", False),
                    "drift_score": result.get("drift_score", 0),
                    "features_with_drift": result.get("features_with_drift", 0),
                }
            )

        input("\nPress Enter to continue...")

    # Summary
    print("\n" + "=" * 80)
    print("📊 SUMMARY: LLM-GENERATED DRIFT DETECTION")
    print("=" * 80)

    if results_summary:
        for r in results_summary:
            status = "🚨 DETECTED" if r["drift_detected"] else "✅ NOT DETECTED"
            print(f"\n{r['drift_type']}:")
            print(f"  Status: {status}")
            print(f"  Score: {r['drift_score']:.4f}")
            print(f"  Features: {r['features_with_drift']}")
    else:
        print("\nNo results yet. Follow the workflow:")
        print("1. Run: python scripts/extract_samples_for_llm.py")
        print("2. Use prompts from: data/llm_experiments/prompts.md")
        print("3. Save LLM outputs to: data/llm_experiments/")
        print("4. Run this script again")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

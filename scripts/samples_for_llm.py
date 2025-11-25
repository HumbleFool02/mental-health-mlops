"""
Extract sample texts for LLM transformation
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Load data
data = pd.read_csv("data/processed/train.csv")

# Sample 10 diverse texts
samples = data.sample(10, random_state=42)

# Print formatted for LLM
print("=" * 80)
print("SAMPLE TEXTS FOR LLM TRANSFORMATION")
print("=" * 80)
print("\nCopy the text below and paste into Claude.ai or ChatGPT with a prompt:")
print("\n" + "-" * 80 + "\n")

for i, row in enumerate(samples.itertuples(), 1):
    print(f"{i}. {row.text}")

print("\n" + "-" * 80)
print("\nSave the LLM output to: data/llm_experiments/")
print("=" * 80)

# Also save to file
output_file = "data/llm_experiments/original_samples.txt"
with open(output_file, "w") as f:
    for i, row in enumerate(samples.itertuples(), 1):
        f.write(f"{i}. {row.text}\n")

print(f"\n✓ Saved to: {output_file}")

# Save as CSV with labels for later comparison
samples.to_csv("data/llm_experiments/original_samples.csv", index=False)
print(f"✓ Saved CSV to: data/llm_experiments/original_samples.csv")

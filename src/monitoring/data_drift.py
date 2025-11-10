from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats


class DataDriftDetector:
    """
    Detects drift in input data features
    """

    def __init__(self, reference_data: pd.DataFrame, threshold: float = 0.1):
        """
        Initialize drift detector

        Args:
            reference_data: Reference dataset (training data)
            threshold: PSI threshold for drift detection (default: 0.1)
        """
        self.reference_data = reference_data
        self.threshold = threshold
        self.reference_stats = self._compute_statistics(reference_data)

    def _compute_statistics(self, data: pd.DataFrame) -> Dict:
        """Compute statistical features of text data"""

        stats_dict = {
            "text_length": data["text"].str.len().values,
            "word_count": data["text"].str.split().str.len().values,
            "avg_word_length": data["text"]
            .apply(lambda x: np.mean([len(word) for word in str(x).split()]))
            .values,
            "unique_words": data["text"]
            .apply(lambda x: len(set(str(x).split())))
            .values,
        }

        return {
            key: {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "median": np.median(values),
                "q25": np.percentile(values, 25),
                "q75": np.percentile(values, 75),
                "distribution": values,
            }
            for key, values in stats_dict.items()
        }

    def compute_psi(
        self, expected: np.ndarray, actual: np.ndarray, bins: int = 10
    ) -> float:
        """
        Calculate Population Stability Index (PSI)

        PSI < 0.1: No significant change
        0.1 <= PSI < 0.2: Moderate change
        PSI >= 0.2: Significant change
        """

        # Create bins based on expected distribution
        breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
        breakpoints = np.unique(breakpoints)  # Remove duplicates

        if len(breakpoints) < 2:
            return 0.0

        # Calculate frequencies
        expected_freq = np.histogram(expected, bins=breakpoints)[0]
        actual_freq = np.histogram(actual, bins=breakpoints)[0]

        # Add small value to avoid division by zero
        expected_freq = expected_freq + 1e-6
        actual_freq = actual_freq + 1e-6

        # Normalize
        expected_pct = expected_freq / np.sum(expected_freq)
        actual_pct = actual_freq / np.sum(actual_freq)

        # Calculate PSI
        psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))

        return psi

    def ks_test(self, expected: np.ndarray, actual: np.ndarray) -> Tuple[float, float]:
        """
        Kolmogorov-Smirnov test for distribution difference

        Returns:
            statistic: KS statistic (0-1, higher = more different)
            p_value: p-value (< 0.05 indicates significant difference)
        """
        statistic, p_value = stats.ks_2samp(expected, actual)
        return statistic, p_value

    def detect_drift(self, current_data: pd.DataFrame) -> Dict:
        """
        Detect drift in current data compared to reference

        Returns:
            Dictionary with drift detection results
        """

        current_stats = self._compute_statistics(current_data)

        drift_results = {
            "timestamp": datetime.now().isoformat(),
            "n_samples": len(current_data),
            "features": {},
            "overall_drift": False,
            "drift_score": 0.0,
        }

        drift_count = 0
        total_psi = 0.0

        # Check each feature
        for feature in ["text_length", "word_count", "avg_word_length", "unique_words"]:
            ref_dist = self.reference_stats[feature]["distribution"]
            curr_dist = current_stats[feature]["distribution"]

            # Calculate PSI
            psi = self.compute_psi(ref_dist, curr_dist)

            # KS test
            ks_stat, ks_pval = self.ks_test(ref_dist, curr_dist)

            # Determine drift
            has_drift = psi > self.threshold or ks_pval < 0.05

            if has_drift:
                drift_count += 1

            total_psi += psi

            drift_results["features"][feature] = {
                "psi": float(psi),
                "ks_statistic": float(ks_stat),
                "ks_pvalue": float(ks_pval),
                "drift_detected": has_drift,
                "reference_mean": float(self.reference_stats[feature]["mean"]),
                "current_mean": float(current_stats[feature]["mean"]),
                "mean_diff_pct": float(
                    (
                        current_stats[feature]["mean"]
                        - self.reference_stats[feature]["mean"]
                    )
                    / self.reference_stats[feature]["mean"]
                    * 100
                ),
            }

        # Overall drift if 2+ features show drift
        drift_results["overall_drift"] = drift_count >= 2
        drift_results["drift_score"] = total_psi / len(drift_results["features"])
        drift_results["features_with_drift"] = drift_count

        return drift_results

    def generate_report(self, drift_results: Dict) -> str:
        """Generate human-readable drift report"""

        report = []
        report.append("=" * 70)
        report.append("DATA DRIFT DETECTION REPORT")
        report.append("=" * 70)
        report.append(f"Timestamp: {drift_results['timestamp']}")
        report.append(f"Samples analyzed: {drift_results['n_samples']}")
        report.append(f"Overall Drift Score: {drift_results['drift_score']:.4f}")
        report.append(
            f"Overall Drift: {'YES ⚠️' if drift_results['overall_drift'] else 'NO ✅'}"
        )
        report.append("")
        report.append("Feature-wise Analysis:")
        report.append("-" * 70)

        for feature, metrics in drift_results["features"].items():
            report.append(f"\n{feature.replace('_', ' ').title()}:")
            report.append(
                f"  PSI: {metrics['psi']:.4f} {'⚠️ DRIFT' if metrics['drift_detected'] else '✅ OK'}"
            )
            report.append(f"  KS Statistic: {metrics['ks_statistic']:.4f}")
            report.append(f"  Reference Mean: {metrics['reference_mean']:.2f}")
            report.append(f"  Current Mean: {metrics['current_mean']:.2f}")
            report.append(f"  Change: {metrics['mean_diff_pct']:+.1f}%")

        report.append("")
        report.append("=" * 70)

        return "\n".join(report)


if __name__ == "__main__":
    # Test the drift detector
    print("Testing Data Drift Detector...")

    # Load reference data
    reference_df = pd.read_csv("data/processed/reference_data.csv")

    # Initialize detector
    detector = DataDriftDetector(reference_df, threshold=0.1)

    # Test with validation data (should have no drift)
    val_df = pd.read_csv("data/processed/val.csv")
    results = detector.detect_drift(val_df)

    print(detector.generate_report(results))

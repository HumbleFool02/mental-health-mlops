"""
Prediction Drift Detection Module
Monitors changes in model prediction distributions
"""

from datetime import datetime
from typing import Dict, List

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance


class PredictionDriftDetector:
    """
    Detects drift in model predictions (output distribution)
    Prediction drift = distribution of predictions changes over time
    """

    def __init__(
        self,
        reference_predictions: np.ndarray,
        class_names: List[str] = None,
        threshold: float = 0.1,
    ):
        """
        Initialize prediction drift detector

        Args:
            reference_predictions: Baseline prediction labels
            class_names: List of class names
            threshold: JS divergence threshold (default: 0.1)
        """
        self.reference_predictions = reference_predictions
        self.class_names = class_names
        self.threshold = threshold

        # Compute reference distribution
        self.reference_distribution = self._compute_distribution(reference_predictions)

    def _compute_distribution(self, predictions: np.ndarray) -> np.ndarray:
        """Compute prediction distribution (class frequencies)"""

        unique, counts = np.unique(predictions, return_counts=True)

        # Create full distribution (including zero counts for missing classes)
        n_classes = (
            len(self.class_names) if self.class_names is not None else len(unique)
        )
        distribution = np.zeros(n_classes)

        for label, count in zip(unique, counts):
            distribution[int(label)] = count

        # Normalize to probabilities
        distribution = distribution / np.sum(distribution)

        return distribution

    def compute_js_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        Compute Jensen-Shannon Divergence

        JS Divergence:
        - 0: Identical distributions
        - 0-0.1: Very similar
        - 0.1-0.2: Moderately different
        - >0.2: Very different
        """
        return jensenshannon(p, q)

    def compute_wasserstein_distance(
        self, ref_pred: np.ndarray, curr_pred: np.ndarray
    ) -> float:
        """
        Compute Wasserstein Distance (Earth Mover's Distance)
        Measures effort to transform one distribution into another
        """
        return wasserstein_distance(ref_pred, curr_pred)

    def detect_drift(self, current_predictions: np.ndarray) -> Dict:
        """
        Detect prediction drift by comparing distributions

        Returns:
            Dictionary with drift detection results
        """

        # Compute current distribution
        current_distribution = self._compute_distribution(current_predictions)

        # Calculate divergence metrics
        js_div = self.compute_js_divergence(
            self.reference_distribution, current_distribution
        )

        wasserstein_dist = self.compute_wasserstein_distance(
            self.reference_predictions, current_predictions
        )

        # Detect drift
        drift_detected = js_div > self.threshold

        drift_results = {
            "timestamp": datetime.now().isoformat(),
            "n_samples": len(current_predictions),
            "js_divergence": float(js_div),
            "wasserstein_distance": float(wasserstein_dist),
            "drift_detected": drift_detected,
            "reference_distribution": self.reference_distribution.tolist(),
            "current_distribution": current_distribution.tolist(),
            "distribution_comparison": {},
        }

        # Compare per-class frequencies
        if self.class_names is not None:
            for idx, class_name in enumerate(self.class_names):
                ref_freq = self.reference_distribution[idx]
                curr_freq = current_distribution[idx]
                diff = curr_freq - ref_freq
                diff_pct = (diff / ref_freq * 100) if ref_freq > 0 else 0

                drift_results["distribution_comparison"][class_name] = {
                    "reference_frequency": float(ref_freq),
                    "current_frequency": float(curr_freq),
                    "difference": float(diff),
                    "difference_pct": float(diff_pct),
                }

        return drift_results

    def detect_confidence_drift(
        self, reference_probabilities: np.ndarray, current_probabilities: np.ndarray
    ) -> Dict:
        """
        Detect drift in prediction confidence scores

        Args:
            reference_probabilities: Max probabilities from reference predictions
            current_probabilities: Max probabilities from current predictions
        """

        ref_conf_mean = np.mean(reference_probabilities)
        ref_conf_std = np.std(reference_probabilities)

        curr_conf_mean = np.mean(current_probabilities)
        curr_conf_std = np.std(current_probabilities)

        # Check if confidence significantly decreased
        confidence_drop = ref_conf_mean - curr_conf_mean
        drift_detected = confidence_drop > 0.05  # 5% threshold

        return {
            "timestamp": datetime.now().isoformat(),
            "reference_confidence_mean": float(ref_conf_mean),
            "reference_confidence_std": float(ref_conf_std),
            "current_confidence_mean": float(curr_conf_mean),
            "current_confidence_std": float(curr_conf_std),
            "confidence_drop": float(confidence_drop),
            "drift_detected": drift_detected,
        }

    def generate_report(self, drift_results: Dict) -> str:
        """Generate human-readable prediction drift report"""

        report = []
        report.append("=" * 70)
        report.append("PREDICTION DRIFT DETECTION REPORT")
        report.append("=" * 70)
        report.append(f"Timestamp: {drift_results['timestamp']}")
        report.append(f"Samples analyzed: {drift_results['n_samples']}")
        report.append(
            f"Overall Drift: {'YES ⚠️' if drift_results['drift_detected'] else 'NO ✅'}"
        )
        report.append("")
        report.append("Divergence Metrics:")
        report.append(
            f"  JS Divergence: {drift_results['js_divergence']:.4f} (threshold: {self.threshold})"
        )
        report.append(
            f"  Wasserstein Distance: {drift_results['wasserstein_distance']:.4f}"
        )

        if (
            "distribution_comparison" in drift_results
            and drift_results["distribution_comparison"]
        ):
            report.append("")
            report.append("Per-Class Distribution Comparison:")
            report.append("-" * 70)

            for class_name, metrics in drift_results["distribution_comparison"].items():
                report.append(f"\n{class_name}:")
                report.append(
                    f"  Reference: {metrics['reference_frequency']:.3f} ({metrics['reference_frequency'] * 100:.1f}%)"
                )
                report.append(
                    f"  Current:   {metrics['current_frequency']:.3f} ({metrics['current_frequency'] * 100:.1f}%)"
                )
                report.append(
                    f"  Change:    {metrics['difference']:+.3f} ({metrics['difference_pct']:+.1f}%)"
                )

        report.append("")
        report.append("=" * 70)

        return "\n".join(report)


if __name__ == "__main__":
    pass

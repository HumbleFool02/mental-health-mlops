"""
Concept Drift Detection Module
Monitors changes in model performance over time
"""

from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class ConceptDriftDetector:
    """
    Detects drift in model performance (concept drift)
    Concept drift = relationship between input and output changes
    """

    def __init__(
        self,
        reference_performance: Dict[str, float],
        threshold: float = 0.05,
        window_size: int = 1000,
    ):
        """
        Initialize concept drift detector

        Args:
            reference_performance: Baseline performance metrics
            threshold: Performance drop threshold (default: 5%)
            window_size: Number of samples in sliding window
        """
        self.reference_performance = reference_performance
        self.threshold = threshold
        self.window_size = window_size
        self.performance_history = []

    def compute_performance_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str] = None
    ) -> Dict:
        """
        Compute comprehensive performance metrics
        """

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_weighted": f1_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
            "precision_macro": precision_score(
                y_true, y_pred, average="macro", zero_division=0
            ),
            "recall_macro": recall_score(
                y_true, y_pred, average="macro", zero_division=0
            ),
        }

        # Per-class metrics if class names provided
        if class_names is not None and len(class_names) > 0:
            f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
            precision_per_class = precision_score(
                y_true, y_pred, average=None, zero_division=0
            )
            recall_per_class = recall_score(
                y_true, y_pred, average=None, zero_division=0
            )

            for idx, class_name in enumerate(class_names):
                metrics[f"f1_{class_name}"] = f1_per_class[idx]
                metrics[f"precision_{class_name}"] = precision_per_class[idx]
                metrics[f"recall_{class_name}"] = recall_per_class[idx]

        return metrics

    def detect_drift(
        self, y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str] = None
    ) -> Dict:
        """
        Detect concept drift by comparing current performance to reference

        Returns:
            Dictionary with drift detection results
        """

        # Compute current metrics
        current_metrics = self.compute_performance_metrics(y_true, y_pred, class_names)

        # Store in history
        self.performance_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "metrics": current_metrics,
                "n_samples": len(y_true),
            }
        )

        # Keep only recent history
        if len(self.performance_history) > 10:
            self.performance_history = self.performance_history[-10:]

        # Compare key metrics
        drift_results = {
            "timestamp": datetime.now().isoformat(),
            "n_samples": len(y_true),
            "current_performance": current_metrics,
            "reference_performance": self.reference_performance,
            "metrics_comparison": {},
            "overall_drift": False,
            "degraded_metrics": [],
        }

        # Check each metric
        key_metrics = ["accuracy", "f1_macro", "f1_weighted"]

        for metric in key_metrics:
            if metric in self.reference_performance:
                ref_value = self.reference_performance[metric]
                curr_value = current_metrics[metric]

                # Calculate degradation
                degradation = ref_value - curr_value
                degradation_pct = (degradation / ref_value) * 100

                # Check if degradation exceeds threshold
                has_drift = degradation > self.threshold

                drift_results["metrics_comparison"][metric] = {
                    "reference": float(ref_value),
                    "current": float(curr_value),
                    "degradation": float(degradation),
                    "degradation_pct": float(degradation_pct),
                    "drift_detected": has_drift,
                }

                if has_drift:
                    drift_results["degraded_metrics"].append(metric)

        # Overall drift if any key metric degraded
        drift_results["overall_drift"] = len(drift_results["degraded_metrics"]) > 0

        # Add per-class analysis if available
        if class_names is not None and len(class_names) > 0:
            drift_results["per_class_drift"] = {}

            for class_name in class_names:
                f1_key = f"f1_{class_name}"
                if f1_key in self.reference_performance and f1_key in current_metrics:
                    ref_f1 = self.reference_performance[f1_key]
                    curr_f1 = current_metrics[f1_key]
                    degradation = ref_f1 - curr_f1

                    drift_results["per_class_drift"][class_name] = {
                        "reference_f1": float(ref_f1),
                        "current_f1": float(curr_f1),
                        "degradation": float(degradation),
                        "drift_detected": degradation > self.threshold,
                    }

        return drift_results

    def generate_report(self, drift_results: Dict) -> str:
        """Generate human-readable concept drift report"""

        report = []
        report.append("=" * 70)
        report.append("CONCEPT DRIFT DETECTION REPORT")
        report.append("=" * 70)
        report.append(f"Timestamp: {drift_results['timestamp']}")
        report.append(f"Samples analyzed: {drift_results['n_samples']}")
        report.append(
            f"Overall Drift: {'YES ⚠️' if drift_results['overall_drift'] else 'NO ✅'}"
        )

        if drift_results["degraded_metrics"]:
            report.append(
                f"Degraded Metrics: {', '.join(drift_results['degraded_metrics'])}"
            )

        report.append("")
        report.append("Performance Comparison:")
        report.append("-" * 70)

        for metric, comparison in drift_results["metrics_comparison"].items():
            status = "⚠️ DRIFT" if comparison["drift_detected"] else "✅ OK"
            report.append(f"\n{metric.upper().replace('_', ' ')}:")
            report.append(f"  Reference: {comparison['reference']:.4f}")
            report.append(f"  Current:   {comparison['current']:.4f}")
            report.append(
                f"  Change:    {comparison['degradation_pct']:+.1f}% {status}"
            )

        # Per-class drift if available
        if "per_class_drift" in drift_results and drift_results["per_class_drift"]:
            report.append("")
            report.append("Per-Class Performance:")
            report.append("-" * 70)

            for class_name, metrics in drift_results["per_class_drift"].items():
                status = "⚠️" if metrics["drift_detected"] else "✅"
                report.append(f"\n{class_name}:")
                report.append(f"  Reference F1: {metrics['reference_f1']:.4f}")
                report.append(f"  Current F1:   {metrics['current_f1']:.4f}")
                report.append(f"  Change:       {metrics['degradation']:+.4f} {status}")

        report.append("")
        report.append("=" * 70)

        return "\n".join(report)

    def get_performance_trend(self) -> pd.DataFrame:
        """Get performance history as DataFrame for visualization"""

        if not self.performance_history:
            return pd.DataFrame()

        records = []
        for entry in self.performance_history:
            record = {"timestamp": entry["timestamp"], "n_samples": entry["n_samples"]}
            record.update(entry["metrics"])
            records.append(record)

        return pd.DataFrame(records)

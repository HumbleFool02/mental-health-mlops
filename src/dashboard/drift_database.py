"""
Drift Monitoring Database

Stores drift checks and predictions for dashboard visualization
"""

import json
import os
import sqlite3
import sys
from datetime import datetime
from typing import Dict, List

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


class DriftDatabase:
    """SQLite database for drift monitoring"""

    def __init__(self, db_path: str = "data/drift_monitoring.db"):
        """Initialize database connection"""
        self.db_path = db_path

        # Create directory if needed
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        # Create tables
        self._create_tables()

    def _create_tables(self):
        """Create database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Drift checks table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS drift_checks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                drift_type TEXT NOT NULL,
                drift_detected INTEGER NOT NULL,
                drift_score REAL NOT NULL,
                features_with_drift INTEGER,
                details TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Predictions table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                text TEXT NOT NULL,
                prediction TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Alerts table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                message TEXT NOT NULL,
                severity TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        conn.commit()
        conn.close()

    def log_drift_check(self, drift_type: str, results: Dict):
        """Log a drift check result"""

        # Helper to convert numpy types
        def convert_value(val):
            if hasattr(val, "item"):  # numpy scalar
                return val.item()
            elif hasattr(val, "tolist"):  # numpy array
                return val.tolist()
            return val

        # Extract values and convert numpy types
        drift_detected = results.get("overall_drift", False) or results.get(
            "drift_detected", False
        )
        if hasattr(drift_detected, "item"):
            drift_detected = drift_detected.item()

        drift_score = results.get("drift_score", 0) or results.get("js_divergence", 0)
        if hasattr(drift_score, "item"):
            drift_score = drift_score.item()

        features_with_drift = results.get("features_with_drift", 0)
        if hasattr(features_with_drift, "item"):
            features_with_drift = features_with_drift.item()

        # Convert entire results dict
        def convert_dict(d):
            if isinstance(d, dict):
                return {k: convert_dict(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [convert_dict(item) for item in d]
            else:
                return convert_value(d)

        clean_results = convert_dict(results)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO drift_checks
            (timestamp, drift_type, drift_detected, drift_score,
            features_with_drift, details)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                datetime.now().isoformat(),
                drift_type,
                int(drift_detected),
                float(drift_score),
                int(features_with_drift),
                json.dumps(clean_results),
            ),
        )

        conn.commit()
        conn.close()

    def log_prediction(self, text: str, prediction: str, confidence: float):
        """Log a prediction"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO predictions (timestamp, text, prediction, confidence)
            VALUES (?, ?, ?, ?)
        """,
            (datetime.now().isoformat(), text, prediction, float(confidence)),
        )

        conn.commit()
        conn.close()

    def log_alert(self, alert_type: str, message: str, severity: str = "warning"):
        """Log an alert"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO alerts (timestamp, alert_type, message, severity)
            VALUES (?, ?, ?, ?)
        """,
            (datetime.now().isoformat(), alert_type, message, severity),
        )

        conn.commit()
        conn.close()

    def get_drift_history(self, limit: int = 100) -> List[Dict]:
        """Get drift check history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT timestamp, drift_type, drift_detected, drift_score,
                   features_with_drift, details
            FROM drift_checks
            ORDER BY timestamp DESC
            LIMIT ?
        """,
            (limit,),
        )

        results = []
        for row in cursor.fetchall():
            results.append(
                {
                    "timestamp": row[0],
                    "drift_type": row[1],
                    "drift_detected": bool(row[2]),
                    "drift_score": row[3],
                    "features_with_drift": row[4],
                    "details": json.loads(row[5]) if row[5] else {},
                }
            )

        conn.close()
        return results

    def get_recent_alerts(self, limit: int = 10) -> List[Dict]:
        """Get recent alerts"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT timestamp, alert_type, message, severity
            FROM alerts
            ORDER BY timestamp DESC
            LIMIT ?
        """,
            (limit,),
        )

        results = []
        for row in cursor.fetchall():
            results.append(
                {
                    "timestamp": row[0],
                    "alert_type": row[1],
                    "message": row[2],
                    "severity": row[3],
                }
            )

        conn.close()
        return results

    def get_latest_drift_check(self) -> Dict:
        """Get the most recent drift check"""
        history = self.get_drift_history(limit=1)
        return history[0] if history else None

    def get_drift_stats(self) -> Dict:
        """Get drift statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Total checks
        cursor.execute("SELECT COUNT(*) FROM drift_checks")
        total_checks = cursor.fetchone()[0]

        # Drift detected
        cursor.execute("SELECT COUNT(*) FROM drift_checks WHERE drift_detected = 1")
        drift_detected = cursor.fetchone()[0]

        # Average drift score
        cursor.execute("SELECT AVG(drift_score) FROM drift_checks")
        avg_score = cursor.fetchone()[0] or 0

        conn.close()

        return {
            "total_checks": total_checks,
            "drift_detected_count": drift_detected,
            "drift_rate": drift_detected / total_checks if total_checks > 0 else 0,
            "avg_drift_score": avg_score,
        }

    def clear_all_data(self):
        """Clear all data (for testing/reset)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("DELETE FROM drift_checks")
        cursor.execute("DELETE FROM predictions")
        cursor.execute("DELETE FROM alerts")

        conn.commit()
        conn.close()

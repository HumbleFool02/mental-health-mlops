"""
Drift Monitoring Dashboard

Real-time visualization of drift detection
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import time
from datetime import datetime, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.drift_database import DriftDatabase

# Page config
st.set_page_config(
    page_title="Drift Monitoring Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .status-healthy {
        color: #28a745;
        font-weight: bold;
    }
    .status-warning {
        color: #ffc107;
        font-weight: bold;
    }
    .status-critical {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""",
    unsafe_allow_html=True,
)


# Initialize database
@st.cache_resource
def get_database():
    return DriftDatabase()


db = get_database()

# Header
st.markdown(
    '<div class="main-header">🏥 Mental Health Model - Drift Monitoring Dashboard</div>',
    unsafe_allow_html=True,
)
st.markdown("---")

# Auto-refresh toggle
col1, col2 = st.columns([3, 1])
with col2:
    auto_refresh = st.checkbox("Auto-refresh (5s)", value=True)

if auto_refresh:
    time.sleep(5)
    st.rerun()

# Get latest data
latest = db.get_latest_drift_check()
stats = db.get_drift_stats()
history = db.get_drift_history(limit=100)
recent_alerts = db.get_recent_alerts(limit=10)

# Current Status Section
st.header("📊 Current Status")

if latest:
    col1, col2, col3, col4 = st.columns(4)

    # Status
    with col1:
        if latest["drift_detected"]:
            status_html = '<span class="status-critical">🚨 DRIFT DETECTED</span>'
        elif latest["drift_score"] > 0.07:
            status_html = '<span class="status-warning">⚠️ WARNING</span>'
        else:
            status_html = '<span class="status-healthy">✅ HEALTHY</span>'

        st.markdown("### Status")
        st.markdown(status_html, unsafe_allow_html=True)

    # Last Check
    with col2:
        st.metric(
            "Last Check",
            datetime.fromisoformat(latest["timestamp"]).strftime("%H:%M:%S"),
            delta=None,
        )

    # Drift Score
    with col3:
        delta_color = "inverse" if latest["drift_detected"] else "normal"
        st.metric(
            "Drift Score (PSI)",
            f"{latest['drift_score']:.4f}",
            delta=f"Threshold: 0.1",
            delta_color=delta_color,
        )

    # Features with Drift
    with col4:
        st.metric("Features w/ Drift", latest["features_with_drift"], delta=None)
else:
    st.info("⏳ No drift checks yet. Start the traffic simulator to generate data.")

st.markdown("---")

# Drift Timeline
if history:
    st.header("📈 Drift Score Timeline")

    # Prepare data
    df_history = pd.DataFrame(history)
    df_history["timestamp"] = pd.to_datetime(df_history["timestamp"])
    df_history = df_history.sort_values("timestamp")

    # Create plot
    fig = go.Figure()

    # Add drift score line
    fig.add_trace(
        go.Scatter(
            x=df_history["timestamp"],
            y=df_history["drift_score"],
            mode="lines+markers",
            name="Drift Score",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=6),
        )
    )

    # Add threshold line
    fig.add_hline(
        y=0.1,
        line_dash="dash",
        line_color="red",
        annotation_text="Threshold (0.1)",
        annotation_position="right",
    )

    # Add warning line
    fig.add_hline(
        y=0.07,
        line_dash="dash",
        line_color="orange",
        annotation_text="Warning (0.07)",
        annotation_position="right",
    )

    # Highlight drift events
    drift_events = df_history[df_history["drift_detected"] == True]
    if not drift_events.empty:
        fig.add_trace(
            go.Scatter(
                x=drift_events["timestamp"],
                y=drift_events["drift_score"],
                mode="markers",
                name="Drift Detected",
                marker=dict(size=12, color="red", symbol="x"),
            )
        )

    fig.update_layout(
        title="Drift Score Over Time",
        xaxis_title="Timestamp",
        yaxis_title="Drift Score (PSI)",
        hovermode="x unified",
        height=400,
    )

    st.plotly_chart(fig, width="stretch")

    st.markdown("---")

# Feature Breakdown
if latest and "details" in latest and "features" in latest["details"]:
    st.header("🔍 Feature Breakdown")

    features = latest["details"]["features"]

    # Create dataframe
    feature_data = []
    for feature, metrics in features.items():
        drift_detected = metrics.get("drift_detected", "False")
        if isinstance(drift_detected, str):
            drift_detected = drift_detected.lower() == "true"

        feature_data.append(
            {
                "Feature": feature,
                "PSI": metrics.get("psi", 0),
                "Mean Change %": metrics.get("mean_diff_pct", 0),
                "Drift Detected": "🚨 Yes" if drift_detected else "✅ No",
            }
        )

    df_features = pd.DataFrame(feature_data)
    df_features = df_features.sort_values("PSI", ascending=False)

    # Display as table
    st.dataframe(df_features, width="stretch", hide_index=True)

    # Bar chart
    fig_features = px.bar(
        df_features,
        x="Feature",
        y="PSI",
        title="Feature Sensitivity (PSI Scores)",
        color="PSI",
        color_continuous_scale=["green", "yellow", "red"],
    )

    fig_features.add_hline(
        y=0.1, line_dash="dash", line_color="red", annotation_text="Threshold"
    )

    st.plotly_chart(fig_features, width="stretch")

    st.markdown("---")

# Alerts Feed
st.header("🔔 Recent Alerts")

if recent_alerts:
    for alert in recent_alerts:
        severity = alert["severity"]

        if severity == "critical":
            icon = "🚨"
            color = "#dc3545"
        elif severity == "warning":
            icon = "⚠️"
            color = "#ffc107"
        else:
            icon = "ℹ️"
            color = "#17a2b8"

        timestamp = datetime.fromisoformat(alert["timestamp"]).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        st.markdown(
            f"""
            <div style="
                border-left: 4px solid {color};
                padding: 10px;
                margin: 10px 0;
                background-color: #f8f9fa;
            ">
                <strong>{icon} {timestamp}</strong><br/>
                {alert['message']}
            </div>
            """,
            unsafe_allow_html=True,
        )
else:
    st.info("No alerts yet")

st.markdown("---")

# Statistics
st.header("📊 Statistics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Checks", stats["total_checks"])

with col2:
    st.metric("Drift Events", stats["drift_detected_count"])

with col3:
    st.metric("Drift Rate", f"{stats['drift_rate']*100:.1f}%")

with col4:
    st.metric("Avg Drift Score", f"{stats['avg_drift_score']:.4f}")

# Drift Detection Rate Over Time
if history:
    st.markdown("---")
    st.subheader("Drift Detection Rate")

    # Calculate rolling drift rate
    df_history["drift_numeric"] = df_history["drift_detected"].astype(int)

    # Group by day and calculate rate
    df_history["date"] = df_history["timestamp"].dt.date
    daily_stats = (
        df_history.groupby("date")
        .agg({"drift_numeric": ["sum", "count", "mean"]})
        .reset_index()
    )
    daily_stats.columns = ["date", "drift_count", "total_checks", "drift_rate"]

    fig_rate = go.Figure()

    fig_rate.add_trace(
        go.Bar(
            x=daily_stats["date"],
            y=daily_stats["drift_rate"] * 100,
            name="Drift Rate",
            marker_color="#ff7f0e",
        )
    )

    fig_rate.update_layout(
        title="Daily Drift Detection Rate",
        xaxis_title="Date",
        yaxis_title="Drift Rate (%)",
        height=300,
    )

    st.plotly_chart(fig_rate, width="stretch")

# Sidebar
st.sidebar.header("⚙️ Configuration")

st.sidebar.markdown("### Thresholds")
st.sidebar.text(f"PSI Threshold: 0.1")
st.sidebar.text(f"Warning Level: 0.07")

st.sidebar.markdown("### System Info")
st.sidebar.text(f"Database: drift_monitoring.db")
st.sidebar.text(f"Last Updated: {datetime.now().strftime('%H:%M:%S')}")

st.sidebar.markdown("---")

st.sidebar.markdown("### Actions")
if st.sidebar.button("🔄 Refresh Now"):
    st.rerun()

if st.sidebar.button("🗑️ Clear All Data"):
    if st.sidebar.checkbox("Confirm clear?"):
        db.clear_all_data()
        st.sidebar.success("✅ Data cleared!")
        st.rerun()

# Recommendations
if latest:
    st.markdown("---")
    st.header("💡 Recommendations")

    if latest["drift_detected"]:
        st.error(
            """
        🚨 **IMMEDIATE ACTION REQUIRED**

        Drift has been detected in the model. Recommended actions:
        1. **Investigate root cause**: Review recent data changes
        2. **Schedule retraining**: Model should be retrained within 7 days
        3. **Monitor closely**: Increase monitoring frequency
        4. **Consider rollback**: If performance degrades significantly
        """
        )
    elif latest["drift_score"] > 0.07:
        st.warning(
            """
        ⚠️ **WARNING: Drift Trending Up**

        Drift score is approaching threshold. Recommended actions:
        1. **Monitor closely**: Check drift scores daily
        2. **Prepare for retraining**: Gather recent data
        3. **Review data quality**: Ensure incoming data is clean
        """
        )
    else:
        st.success(
            """
        ✅ **SYSTEM HEALTHY**

        No drift detected. Continue normal operations:
        1. **Routine monitoring**: Weekly drift checks
        2. **Model performance**: Track F1 score monthly
        3. **Data quality**: Maintain data validation pipeline
        """
        )

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666;">
        Mental Health Model Drift Monitoring Dashboard v1.0 |
        Built with Streamlit & Plotly
    </div>
    """,
    unsafe_allow_html=True,
)

import streamlit as st
import requests
import json
import pandas as pd
import os
import plotly.express as px

st.set_page_config(page_title="Anomaly Detection Dashboard", layout="wide")

# --------------------------
# Define label mappings
# --------------------------
primary_account_options = {"ALL OTHER LOANS": 0}
secondary_account_options = {"DEFERRED COSTS": 0, "DEFERRED ORIGINATION FEES": 1}
currency_options = {"USD": 0}

# --------------------------
# TABS: Prediction + Feedback + Training Logs
# --------------------------
tab1, tab2, tab3 = st.tabs(["🔍 Detect Anomaly", "📊 Feedback Dashboard", "📈 Training Logs"])

# --------------------------
# 🔍 TAB 1: Anomaly Detection
# --------------------------
with tab1:
    st.title("🧠 Anomaly Detection")

    balance_diff = st.number_input("🔢 Balance Difference", value=-20000.0)
    primary_account = st.selectbox("🏦 Primary Account", list(primary_account_options.keys()))
    secondary_account = st.selectbox("📂 Secondary Account", list(secondary_account_options.keys()))
    currency = st.selectbox("💱 Currency", list(currency_options.keys()))

    payload = {
        "Balance Difference": balance_diff,
        "Primary Account": primary_account_options[primary_account],
        "Secondary Account": secondary_account_options[secondary_account],
        "Currency": currency_options[currency]
    }

    if "anomaly_result" not in st.session_state:
        st.session_state.anomaly_result = None

    if st.button("Detect"):
        try:
            response = requests.post("http://backend:8000/detect_anomaly/", json=payload)
            result = response.json()
            st.session_state.anomaly_result = result
        except Exception as e:
            st.error(f"API Error: {e}")

    if st.session_state.anomaly_result:
        result = st.session_state.anomaly_result
        st.success(f"🚨 Anomaly: {result['Anomaly']}")
        feedback = st.radio("Was this prediction correct?", ["Yes", "No"], key="feedback_radio")

        if st.button("Submit Feedback"):
            feedback_data = {
                "feedback": feedback,
                "input": payload,
                "prediction": result["Anomaly"]
            }

            file_path = "/ui/feedback.json"
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    existing = json.load(f)
            else:
                existing = []

            existing.append(feedback_data)
            with open(file_path, "w") as f:
                json.dump(existing, f, indent=2)

            st.success("✅ Feedback recorded.")
            st.write("📁 Saved to:", file_path)
            st.session_state.anomaly_result = None

# --------------------------
# 📊 TAB 2: Feedback Dashboard
# --------------------------
with tab2:
    st.title("📊 Feedback Analytics")

    file_path = "/ui/feedback.json"  # Fixed path
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            data = json.load(f)

        df = pd.DataFrame(data)

        if not df.empty:
            col1, col2 = st.columns(2)

            with col1:
                st.metric("Total Feedback", len(df))
                st.bar_chart(df["feedback"].value_counts())

            with col2:
                st.metric("Prediction Accuracy", f"{(df['feedback'] == 'Yes').mean() * 100:.2f}%")
                st.bar_chart(df["prediction"].value_counts())

            st.subheader("📉 Balance Difference vs. Feedback")
            df["balance"] = df["input"].apply(lambda x: x["Balance Difference"])
            fig = px.histogram(df, x="balance", color="feedback", nbins=30)
            st.plotly_chart(fig)
        else:
            st.info("No feedback data yet.")
    else:
        st.warning("feedback.json not found at /ui/feedback.json")

# --------------------------
# 📈 TAB 3: Training Logs
# --------------------------
with tab3:
    st.title("📈 Model Training Logs")

    training_log_path = "/shared/training_logs.json"

    if os.path.exists(training_log_path):
        with open(training_log_path, "r") as f:
            logs = json.load(f)

        if logs:
            df_logs = pd.DataFrame(logs)
            df_logs["timestamp"] = pd.to_datetime(df_logs["timestamp"])
            df_logs = df_logs.sort_values(by="timestamp")

            st.subheader("📊 Accuracy Over Time")
            st.line_chart(df_logs.set_index("timestamp")[["accuracy"]])

            if "precision_0" in df_logs.columns:
                st.subheader("🎯 Precision & Recall by Class")
                metric_cols = ["precision_0", "recall_0", "precision_1", "recall_1"]

                fig = px.line(
                    df_logs,
                    x="timestamp",
                    y=metric_cols,
                    title="Model Precision & Recall Over Time",
                    markers=True
                )
                st.plotly_chart(fig, use_container_width=True)

            st.dataframe(df_logs, use_container_width=True)

        else:
            st.warning("⚠️ Training log file exists but is empty.")
    else:
        st.error("❌ training_logs.json not found at /shared.")


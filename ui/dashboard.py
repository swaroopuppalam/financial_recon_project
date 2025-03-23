import streamlit as st
import requests
import json
import pandas as pd
import os
import plotly.express as px

st.set_page_config(page_title="Anomaly Detection Dashboard", layout="wide")

# --------------------------
# Load anomaly config
# --------------------------
CONFIG_PATH = "/ml/config.json"
FEATURE_IMPORTANCE_PATH = "/shared/feature_importance.json"
FEEDBACK_PATH = "/ui/feedback.json"

def load_config():
    default = {"anomaly_rules": {"quantity_threshold": 1.0, "price_threshold": 0.01}}
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH) as f:
                return json.load(f)
        else:
            st.warning("⚠️ Could not find /ml/config.json. Using default thresholds.")
            return default
    except Exception as e:
        st.warning(f"⚠️ Failed to load config: {e}")
        return default

config = load_config()
threshold_qty = config["anomaly_rules"]["quantity_threshold"]
threshold_price = config["anomaly_rules"]["price_threshold"]

# --------------------------
# Label Mappings
# --------------------------
primary_account_options = {"ALL OTHER LOANS": 0}
secondary_account_options = {"DEFERRED COSTS": 0, "DEFERRED ORIGINATION FEES": 1}
currency_options = {"USD": 0}

# --------------------------
# Tabs
# --------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 Detect Anomaly", "📊 Feedback Dashboard", "📈 Training Logs", "⚙️ Anomaly Config", "🧠 Model Insights"
])

# --------------------------
# 🔍 TAB 1: Anomaly Detection
# --------------------------
with tab1:
    st.title("🧠 Anomaly Detection")

    st.markdown("### ⚙️ Current Anomaly Thresholds")
    st.markdown(f"""
    - **Quantity Difference >** `{threshold_qty}`
    - **Price Difference >** `{threshold_price}`
    """)

    balance_diff = st.number_input("🔢 Balance Difference", value=-20000.0)
    primary_account = st.selectbox("🏦 Primary Account", list(primary_account_options.keys()))
    secondary_account = st.selectbox("📂 Secondary Account", list(secondary_account_options.keys()))
    currency = st.selectbox("💱 Currency", list(currency_options.keys()))

    payload = {
        "Balance Difference": balance_diff,
        "Primary Account": primary_account_options[primary_account],
        "Secondary Account": secondary_account_options[secondary_account],
        "Currency": currency_options[currency],
        "QUANTITYDIFFERENCE": 0.0,
        "PRICEDIFFERENCE": 0.0
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
        if "Anomaly" in result:
            label = "Yes" if result["Anomaly"] == 1 else "No"
            st.success(f"🚨 Anomaly: {label}")
            # ✅ Show explanation if present
            if "explanation" in result:
                st.markdown("#### 🧾 Explanation")
                for reason in result["explanation"]:
                    st.markdown(f"- {reason}")
            feedback = st.radio("Was this prediction correct?", ["Yes", "No"], key="feedback_radio")

            if st.button("Submit Feedback"):
                feedback_data = {
                    "feedback": feedback,
                    "input": payload,
                    "prediction": int(result["Anomaly"])
                }

                if os.path.exists(FEEDBACK_PATH):
                    with open(FEEDBACK_PATH, "r") as f:
                        existing = json.load(f)
                else:
                    existing = []

                existing.append(feedback_data)
                with open(FEEDBACK_PATH, "w") as f:
                    json.dump(existing, f, indent=2)

                st.success("✅ Feedback recorded.")
                st.write("📁 Saved to:", FEEDBACK_PATH)
                st.session_state.anomaly_result = None
        else:
            st.error(f"❌ Prediction failed: {result.get('error', 'Unknown error')}")

# --------------------------
# 📊 TAB 2: Feedback Dashboard
# --------------------------
with tab2:
    st.title("📊 Feedback Analytics")

    if os.path.exists(FEEDBACK_PATH):
        with open(FEEDBACK_PATH, "r") as f:
            data = json.load(f)

        df = pd.DataFrame(data)
        df["prediction"] = df["prediction"].astype(str)

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
            st.line_chart(df_logs.set_index("timestamp")["accuracy"])

            metrics = ["precision_0", "recall_0", "precision_1", "recall_1"]
            available_metrics = [m for m in metrics if m in df_logs.columns]

            if available_metrics:
                st.subheader("🔍 Precision & Recall")
                st.line_chart(df_logs.set_index("timestamp")[available_metrics])

            st.subheader("📋 Full Training Log")
            st.dataframe(df_logs.tail(10))
        else:
            st.info("Training log is currently empty.")
    else:
        st.warning("No training history found at /shared/training_logs.json.")

# --------------------------
# ⚙️ TAB 4: Anomaly Config
# --------------------------
with tab4:
    st.title("⚙️ Update Anomaly Thresholds")

    current_qty = float(config["anomaly_rules"]["quantity_threshold"])
    current_price = float(config["anomaly_rules"]["price_threshold"])

    new_qty_thresh = st.number_input("📏 Quantity Difference Threshold", min_value=0.0, step=0.01, value=current_qty)
    new_price_thresh = st.number_input("💰 Price Difference Threshold", min_value=0.0, step=0.01, value=current_price)

    if st.button("💾 Save Thresholds"):
        new_config = {
            "anomaly_rules": {
                "quantity_threshold": new_qty_thresh,
                "price_threshold": new_price_thresh
            }
        }
        try:
            with open(CONFIG_PATH, "w") as f:
                json.dump(new_config, f, indent=2)
            st.success("✅ Thresholds updated. Please retrain the model to reflect changes.")
        except Exception as e:
            st.error(f"❌ Failed to save config: {e}")

# --------------------------
# 🧠 TAB 5: Model Insights
# --------------------------
with tab5:
    st.title("🧠 Feature Importance (Model Decision Logic)")

    if os.path.exists(FEATURE_IMPORTANCE_PATH):
        with open(FEATURE_IMPORTANCE_PATH, "r") as f:
            importance = json.load(f)

        df_feat = pd.DataFrame({
            "Feature": list(importance.keys()),
            "Importance": list(importance.values())
        }).sort_values(by="Importance", ascending=False)

        st.bar_chart(df_feat.set_index("Feature"))
        st.dataframe(df_feat)
    else:
        st.info("Feature importance file not found. Please retrain model.")

import streamlit as st
import requests
import json
import pandas as pd
import os
import plotly.express as px

st.set_page_config(page_title="Anomaly Detection Dashboard", layout="wide")

# --------------------------
# Constants
# --------------------------
CONFIG_PATH = "/ml/config.json"
FEATURE_IMPORTANCE_PATH = "/shared/feature_importance.json"
FEEDBACK_PATH = "/ui/feedback.json"
TRAINING_LOGS_PATH = "/shared/training_logs.json"
UPLOAD_ENDPOINT = "http://backend:8000/upload_dataset/"
DETECT_ENDPOINT = "http://backend:8000/detect_anomaly/"
FEEDBACK_EXPORT_NAME = "feedback_export.csv"

# --------------------------
# Load helpers
# --------------------------
def load_json(path, default=None):
    try:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    except:
        pass
    return default

config = load_json(CONFIG_PATH, {"anomaly_rules": {"quantity_threshold": 1.0, "price_threshold": 0.01}})
threshold_qty = config["anomaly_rules"].get("quantity_threshold", 1.0)
threshold_price = config["anomaly_rules"].get("price_threshold", 0.01)

primary_account_options = {"ALL OTHER LOANS": 0}
secondary_account_options = {"DEFERRED COSTS": 0, "DEFERRED ORIGINATION FEES": 1}
currency_options = {"USD": 0}

# --------------------------
# Tabs
# --------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "🔍 Predict Anomaly", "📤 Upload CSV", "📊 Feedback Dashboard",
    "📈 Training Logs", "⚙️ Threshold Config", "🧠 Model Insights", "📥 Export Feedback", "🔬 Debug Prediction"
])

# --------------------------
# 🔍 TAB 1: Predict Anomaly
# --------------------------
with tab1:
    st.title("🔍 Predict Anomaly")

    st.markdown("### ⚙️ Current Thresholds")
    st.markdown(f"- Quantity Difference > `{threshold_qty}`")
    st.markdown(f"- Price Difference > `{threshold_price}`")

    balance_diff = st.number_input("💰 Balance Difference", value=-20000.0)
    primary_account = st.selectbox("🏦 Primary Account", list(primary_account_options.keys()))
    secondary_account = st.selectbox("📂 Secondary Account", list(secondary_account_options.keys()))
    currency = st.selectbox("💱 Currency", list(currency_options.keys()))
    qty_diff = st.number_input("📦 QUANTITYDIFFERENCE", value=0.0)
    price_diff = st.number_input("💰 PRICEDIFFERENCE", value=0.0)

    payload = {
        "Balance Difference": balance_diff,
        "Primary Account": primary_account_options[primary_account],
        "Secondary Account": secondary_account_options[secondary_account],
        "Currency": currency_options[currency],
        "QUANTITYDIFFERENCE": qty_diff,
        "PRICEDIFFERENCE": price_diff
    }

    if "anomaly_result" not in st.session_state:
        st.session_state.anomaly_result = None

    if st.button("🚨 Detect"):
        try:
            response = requests.post(DETECT_ENDPOINT, json=payload)
            result = response.json()
            st.session_state.anomaly_result = result
        except Exception as e:
            st.error(f"API Error: {e}")

    if st.session_state.anomaly_result:
        result = st.session_state.anomaly_result
        if "Anomaly" in result:
            label = "Yes" if result["Anomaly"] == 1 else "No"
            st.success(f"🚨 Anomaly: {label}")
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
                st.session_state.anomaly_result = None
        else:
            st.error("❌ No valid response from backend.")

# --------------------------
# 📤 TAB 2: Upload CSV
# --------------------------
with tab2:
    st.title("📤 Upload Dataset for Retraining")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        try:
            response = requests.post(
                UPLOAD_ENDPOINT,
                files={"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
            )
            result = response.json()
            st.success(result.get("message", "Uploaded successfully"))
        except Exception as e:
            st.error(f"Upload failed: {e}")

# --------------------------
# 📊 TAB 3: Feedback Dashboard
# --------------------------
with tab3:
    st.title("📊 Feedback Analytics")
    data = load_json(FEEDBACK_PATH, [])
    df = pd.DataFrame(data)
    if not df.empty:
        df["prediction"] = df["prediction"].astype(str)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Feedback", len(df))
            st.bar_chart(df["feedback"].value_counts())
        with col2:
            acc = (df["feedback"] == "Yes").mean() * 100
            st.metric("Prediction Accuracy", f"{acc:.2f}%")
            st.bar_chart(df["prediction"].value_counts())
        st.subheader("📉 Balance Difference vs Feedback")
        df["balance"] = df["input"].apply(lambda x: x.get("Balance Difference", 0.0))
        fig = px.histogram(df, x="balance", color="feedback", nbins=30)
        st.plotly_chart(fig)
    else:
        st.info("No feedback yet.")

# --------------------------
# 📈 TAB 4: Training Logs
# --------------------------
with tab4:
    st.title("📈 Model Training Logs")
    logs = load_json(TRAINING_LOGS_PATH, [])
    if logs:
        df_logs = pd.DataFrame(logs)
        df_logs["timestamp"] = pd.to_datetime(df_logs["timestamp"])
        df_logs = df_logs.sort_values("timestamp")
        st.subheader("📊 Accuracy Over Time")
        st.line_chart(df_logs.set_index("timestamp")["accuracy"])
        metrics = ["precision_0", "recall_0", "precision_1", "recall_1"]
        if all(m in df_logs.columns for m in metrics):
            st.subheader("🔍 Precision & Recall")
            st.line_chart(df_logs.set_index("timestamp")[metrics])
        st.subheader("📋 Recent Training Logs")
        st.dataframe(df_logs.tail(10))
    else:
        st.warning("Training logs not found.")

# --------------------------
# ⚙️ TAB 5: Threshold Config
# --------------------------
with tab5:
    st.title("⚙️ Update Anomaly Thresholds")
    current_qty = float(threshold_qty)
    current_price = float(threshold_price)
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
# 🧠 TAB 6: Model Insights
# --------------------------
with tab6:
    st.title("🧠 Feature Importance")
    importance = load_json(FEATURE_IMPORTANCE_PATH, {})
    if importance:
        df_feat = pd.DataFrame({
            "Feature": list(importance.keys()),
            "Importance": list(importance.values())
        }).sort_values("Importance", ascending=False)
        st.bar_chart(df_feat.set_index("Feature"))
        st.dataframe(df_feat)
    else:
        st.warning("Feature importance not available.")

# --------------------------
# 📥 TAB 7: Export Feedback
# --------------------------
with tab7:
    st.title("📥 Export Feedback")
    data = load_json(FEEDBACK_PATH, [])
    if data:
        df = pd.DataFrame(data)
        st.download_button("⬇️ Download Feedback CSV", df.to_csv(index=False), file_name=FEEDBACK_EXPORT_NAME)
    else:
        st.warning("⚠️ No feedback available to export.")

# --------------------------
# 🔬 TAB 8: Debug Prediction
# --------------------------
with tab8:
    st.title("🔬 Debug Prediction Engine")
    st.markdown("Upload or enter a row to inspect how the model reasons.")

    sample_input = {
        "Balance Difference": st.number_input("🔢 Balance Difference", value=0.0, key="debug_bd"),
        "Primary Account": primary_account_options[
            st.selectbox("🏦 Primary Account", list(primary_account_options.keys()), key="debug_pa")
        ],
        "Secondary Account": secondary_account_options[
            st.selectbox("📂 Secondary Account", list(secondary_account_options.keys()), key="debug_sa")
        ],
        "Currency": currency_options[
            st.selectbox("💱 Currency", list(currency_options.keys()), key="debug_curr")
        ],
        "QUANTITYDIFFERENCE": st.number_input("📦 QUANTITYDIFFERENCE", value=0.0, key="debug_qty"),
        "PRICEDIFFERENCE": st.number_input("💰 PRICEDIFFERENCE", value=0.0, key="debug_price")
    }

    if st.button("🔍 Run Debug Prediction", key="debug_button"):
        try:
            response = requests.post(DETECT_ENDPOINT, json=sample_input)
            result = response.json()
            if "Anomaly" in result:
                label = "Yes" if result["Anomaly"] == 1 else "No"
                st.success(f"🚨 Anomaly: {label}")
                if "explanation" in result:
                    st.markdown("### 📋 Explanation")
                    for item in result["explanation"]:
                        st.markdown(f"- {item}")
            else:
                st.warning(f"Backend error: {result}")
        except Exception as e:
            st.error(f"Debug error: {e}")


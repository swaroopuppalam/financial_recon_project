import streamlit as st
import requests
import json
import pandas as pd
import os
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve

st.set_page_config(page_title="Anomaly Detection Dashboard", layout="wide")

# --------------------------
# Constants & Paths
# --------------------------
CONFIG_PATH = "/ml/config.json"
FEATURE_IMPORTANCE_PATH = "/shared/feature_importance.json"
FEEDBACK_PATH = "/ui/feedback.json"
TRAINING_LOGS_PATH = "/shared/training_logs.json"
AGENT_LOG_PATH = "/shared/agent_log.json"
UPLOAD_ENDPOINT = "http://backend:8000/upload_dataset/"
DETECT_ENDPOINT = "http://backend:8000/detect_anomaly/"
TRIGGER_RETRAIN_ENDPOINT = "http://backend:8000/trigger_retrain/"
MODEL_LIST_ENDPOINT = "http://backend:8000/list_models/"
MODEL_SWITCH_ENDPOINT = "http://backend:8000/switch_model/"
AGENT_LOGS_ENDPOINT = "http://backend:8000/agent_logs/"
DATASET_DIR = "/ml"
FEEDBACK_EXPORT_NAME = "feedback_export.csv"

# --------------------------
# Load JSON helpers
# --------------------------
def load_json(path, default=None):
    try:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    except:
        pass
    return default

# --------------------------
# Load config
# --------------------------
config = load_json(CONFIG_PATH, {"anomaly_rules": {"quantity_threshold": 1.0, "price_threshold": 0.01}})
threshold_qty = config["anomaly_rules"].get("quantity_threshold", 1.0)
threshold_price = config["anomaly_rules"].get("price_threshold", 0.01)

# Label maps
primary_account_options = {"ALL OTHER LOANS": 0}
secondary_account_options = {"DEFERRED COSTS": 0, "DEFERRED ORIGINATION FEES": 1}
currency_options = {"USD": 0}

# --------------------------
# Tabs
# --------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "🔍 Predict Anomaly", "📤 Upload CSV", "📊 Feedback Dashboard",
    "📈 Training Logs", "⚙️ Threshold Config", "🧠 Model Insights",
    "📥 Export Feedback", "🔬 Debug Prediction", "🤖 Agent Actions"
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

    # Initialize session state
    if "anomaly_result" not in st.session_state:
        st.session_state.anomaly_result = None

    if "latest_payload" not in st.session_state:
        st.session_state.latest_payload = {}

    if st.button("🚨 Detect"):
        try:
            response = requests.post(DETECT_ENDPOINT, json=payload)
            result = response.json()
            if "error" in result:
                st.error(f"❌ Backend Error: {result['error']}")
            elif "Anomaly" in result:
                st.session_state.anomaly_result = result
                st.session_state.latest_payload = payload
            else:
                st.warning("⚠️ No 'Anomaly' field in backend response.")
        except Exception as e:
            st.error(f"❌ Failed to reach backend: {e}")

    result = st.session_state.anomaly_result
    if result and "Anomaly" in result:
        st.success(f"🚨 Anomaly: {'Yes' if result['Anomaly'] else 'No'}")
        if "explanation" in result:
            st.markdown("### 🧾 Explanation")
            for reason in result["explanation"]:
                st.markdown(f"- {reason}")

        st.markdown("### ✅ Was this prediction correct?")
        feedback = st.radio("Feedback:", ["Yes", "No"], key="feedback_radio")
        if st.button("📩 Submit Feedback"):
            feedback_payload = {
                "feedback": feedback,
                "input": st.session_state.latest_payload,
                "prediction": int(result["Anomaly"])
            }
            try:
                res = requests.post("http://backend:8000/submit_feedback/", json=feedback_payload)
                if res.status_code == 200:
                    st.success("📝 Feedback submitted successfully.")
                    st.session_state.anomaly_result = None
                else:
                    st.error("❌ Feedback submission failed.")
            except Exception as e:
                st.error(f"❌ Error submitting feedback: {e}")

    if st.button("♻️ Trigger Model Retraining"):
        try:
            response = requests.post(TRIGGER_RETRAIN_ENDPOINT)
            st.success(response.json().get("message", "Retraining triggered"))
        except Exception as e:
            st.error(f"Retrain trigger failed: {e}")

# --------------------------
# 📤 TAB 2: Upload CSV
# --------------------------
with tab2:
    st.title("📤 Upload Dataset & Trigger Retrain")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        try:
            response = requests.post(
                UPLOAD_ENDPOINT,
                files={"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
            )
            result = response.json()
            st.success(result.get("message", "Uploaded successfully. Model will retrain."))
        except Exception as e:
            st.error(f"Upload failed: {e}")

    st.subheader("📚 Uploaded Datasets")
    if os.path.exists(DATASET_DIR):
        files = [f for f in os.listdir(DATASET_DIR) if f.endswith(".csv")]
        if files:
            file = st.selectbox("📄 Select a dataset to view", files)
            df = pd.read_csv(os.path.join(DATASET_DIR, file))
            st.dataframe(df.head(50))
        else:
            st.info("No datasets found.")
    else:
        st.warning("/ml folder missing.")

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

        st.line_chart(df_logs.set_index("timestamp")["accuracy"])

        metrics = ["precision_0", "recall_0", "precision_1", "recall_1"]
        if all(m in df_logs.columns for m in metrics):
            st.subheader("🔍 Precision & Recall")
            st.line_chart(df_logs.set_index("timestamp")[metrics])

        if "y_test" in df_logs.columns and "y_pred" in df_logs.columns:
            y_test = df_logs.iloc[-1]["y_test"]
            y_pred = df_logs.iloc[-1]["y_pred"]
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay(cm).plot(ax=ax)
            st.pyplot(fig)

            try:
                auc = roc_auc_score(y_test, y_pred)
                fpr, tpr, _ = roc_curve(y_test, y_pred)
                st.subheader(f"🧪 ROC AUC: {auc:.2f}")
                fig2, ax2 = plt.subplots()
                ax2.plot(fpr, tpr)
                ax2.set_title("ROC Curve")
                ax2.set_xlabel("False Positive Rate")
                ax2.set_ylabel("True Positive Rate")
                st.pyplot(fig2)
            except:
                st.warning("Unable to compute ROC/AUC.")

        st.dataframe(df_logs.tail(10))
    else:
        st.info("No training logs available.")
    YTEST_OUTPUT_PATH = "/shared/y_true_pred.json"
    ytp = load_json(YTEST_OUTPUT_PATH, {})
    if ytp and "y_test" in ytp and "y_pred" in ytp:
        st.subheader("📉 Confusion Matrix")
        cm = confusion_matrix(ytp["y_test"], ytp["y_pred"])
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(cm).plot(ax=ax)
        st.pyplot(fig)

        try:
            auc = roc_auc_score(ytp["y_test"], ytp["y_pred"])
            fpr, tpr, _ = roc_curve(ytp["y_test"], ytp["y_pred"])
            st.subheader(f"🧪 ROC AUC: {auc:.2f}")
            fig2, ax2 = plt.subplots()
            ax2.plot(fpr, tpr)
            ax2.set_title("ROC Curve")
            ax2.set_xlabel("False Positive Rate")
            ax2.set_ylabel("True Positive Rate")
            st.pyplot(fig2)
        except Exception as e:
            st.warning(f"⚠️ Unable to compute ROC/AUC: {e}")

# --------------------------
# ⚙️ TAB 5: Threshold Config
# --------------------------
with tab5:
    st.title("⚙️ Update Thresholds")
    new_qty_thresh = st.number_input("📏 Quantity Threshold", min_value=0.0, step=0.01, value=float(threshold_qty))
    new_price_thresh = st.number_input("💰 Price Threshold", min_value=0.0, step=0.01, value=float(threshold_price))
    if st.button("💾 Save Thresholds"):
        try:
            with open(CONFIG_PATH, "w") as f:
                json.dump({"anomaly_rules": {
                    "quantity_threshold": new_qty_thresh,
                    "price_threshold": new_price_thresh
                }}, f, indent=2)
            st.success("✅ Thresholds updated.")
        except Exception as e:
            st.error(f"❌ Failed to update: {e}")

# --------------------------
# 🧠 TAB 6: Feature Importance
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
        st.warning("No feature importance available.")

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
        st.warning("⚠️ No feedback available.")

# --------------------------
# 🔬 TAB 8: Debug Prediction
# --------------------------
with tab8:
    st.title("🔬 Debug Prediction Engine")
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
                st.success(f"🚨 Anomaly: {'Yes' if result['Anomaly'] else 'No'}")
                if "explanation" in result:
                    st.markdown("### 📋 Explanation")
                    for item in result["explanation"]:
                        st.markdown(f"- {item}")
            else:
                st.warning(f"Backend error: {result}")
        except Exception as e:
            st.error(f"Debug error: {e}")

# --------------------------
# 🤖 TAB 9: Agent Actions
# --------------------------
with tab9:
    st.title("🤖 Agent Actions & Logs")

    # Load agent logs
    try:
        res = requests.get(AGENT_LOGS_ENDPOINT)
        logs = res.json().get("logs", [])
        if logs:
            df = pd.DataFrame(logs)
            st.dataframe(df[::-1])
        else:
            st.info("No agent actions recorded yet.")
    except Exception as e:
        st.error(f"Failed to load agent logs: {e}")

    st.divider()
    st.subheader("🛠️ Manually Trigger Agent Action")
    anomaly_id = st.text_input("🔢 Anomaly ID", value="123456")
    reason = st.selectbox("📂 Reason Bucket", [
        "Price Mismatch", "Quantity Mismatch", "Timing Issue",
        "GL vs IHub Difference", "Unknown Reason"
    ])
    if st.button("🚀 Trigger Actions Manually"):
        try:
            payload = {
                "anomaly_id": anomaly_id,
                "reason": reason
            }
            existing_logs = load_json(AGENT_LOG_PATH, [])
            task_time = strftime("%Y-%m-%d %H:%M:%S")
            new_tasks = [
                {"action": "Create JIRA Ticket", "reason": reason, "anomaly_id": anomaly_id, "status": "submitted", "timestamp": task_time},
                {"action": "Send Email Alert", "reason": reason, "anomaly_id": anomaly_id, "status": "sent", "timestamp": task_time},
                {"action": "Create Resolution Task", "reason": reason, "anomaly_id": anomaly_id, "status": "queued", "timestamp": task_time}
            ]
            updated = existing_logs + new_tasks
            with open(AGENT_LOG_PATH, "w") as f:
                json.dump(updated, f, indent=2)
            st.success("✅ Agent actions triggered and logged.")
        except Exception as e:
            st.error(f"Failed to trigger actions: {e}")
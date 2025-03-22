import pandas as pd
import joblib
import os
import json
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, precision_score, recall_score, accuracy_score
)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample

print("\U0001F504 Retraining Model from Feedback (Incremental Learning Mode)...")

# ✅ Paths
FEEDBACK_PATH = "/ui/feedback.json"
MODEL_PATH = "/app/model.pkl"
TRAINING_LOGS = "/shared/training_logs.json"
MODEL_BACKUP_DIR = "/app/model_backups"

# --------------------------
# 🔹 Load Existing Model (Keep Previous Knowledge)
# --------------------------
if os.path.exists(MODEL_PATH):
    print("\U0001F9E0 Loading Existing Model for Incremental Learning...")
    model = joblib.load(MODEL_PATH)
    os.makedirs(MODEL_BACKUP_DIR, exist_ok=True)
    timestamp = int(time.time())
    backup_path = os.path.join(MODEL_BACKUP_DIR, f"model_{timestamp}.pkl")
    joblib.dump(model, backup_path)
    print(f"\U0001F4C2 Previous model backed up: {backup_path}")
else:
    print("⚠️ No existing model found. Training from scratch...")
    model = None

# --------------------------
# 🔹 Load Historical & Real-Time Data
# --------------------------
def safe_read_csv(path):
    try:
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        return df
    except Exception:
        print(f"⚠️ Warning: Could not load {path}. Proceeding without it.")
        return pd.DataFrame(columns=["Balance Difference", "Primary Account", "Secondary Account", "Currency", "label"])

historical_df = safe_read_csv("/ml/historical_reconciliation.csv")
real_time_df = safe_read_csv("/ml/real_time_transaction.csv")

# --------------------------
# 🔹 Load & Normalize Feedback
# --------------------------
if os.path.exists(FEEDBACK_PATH):
    try:
        with open(FEEDBACK_PATH, "r") as f:
            feedback = json.load(f)
        feedback_df = pd.DataFrame(feedback)
        feedback_features = pd.json_normalize(feedback_df["input"], errors="ignore")
        feedback_features["label"] = feedback_df["feedback"].apply(lambda x: 1 if x == "Yes" else 0)
    except Exception as e:
        print(f"❌ ERROR loading feedback.json: {e}")
        feedback_features = pd.DataFrame(columns=["Balance Difference", "Primary Account", "Secondary Account", "Currency", "label"])
else:
    print("⚠️ No feedback.json found. Proceeding without feedback.")
    feedback_features = pd.DataFrame(columns=["Balance Difference", "Primary Account", "Secondary Account", "Currency", "label"])

# --------------------------
# 🔹 Label Encoding
# --------------------------
categorical_features = ["Primary Account", "Secondary Account", "Currency"]
required_features = ["Balance Difference"] + categorical_features

label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    combined = pd.concat([historical_df[col], real_time_df[col], feedback_features[col]], axis=0)
    le.fit(combined.astype(str))
    for df in [historical_df, real_time_df, feedback_features]:
        df[col] = le.transform(df[col].astype(str))
    label_encoders[col] = le

# --------------------------
# 🔹 Label Anomalies
# --------------------------
historical_df["label"] = historical_df["Balance Difference"].apply(lambda x: 0 if abs(x) < 1 else 1)
real_time_df["label"] = real_time_df["Balance Difference"].apply(lambda x: 0 if abs(x) < 1 else 1)

# --------------------------
# 🔹 Cap feedback to 20%
# --------------------------
feedback_capped = feedback_features.sample(frac=0.2, random_state=42) if not feedback_features.empty else feedback_features

# --------------------------
# 🔹 Merge & Balance
# --------------------------
full_data = pd.concat([
    historical_df[required_features + ["label"]],
    real_time_df[required_features + ["label"]],
    feedback_capped
], ignore_index=True)

normal_df = full_data[full_data["label"] == 0]
anomaly_df = full_data[full_data["label"] == 1]

min_len = min(len(normal_df), len(anomaly_df))
normal_df = resample(normal_df, replace=True, n_samples=min_len, random_state=42)
anomaly_df = resample(anomaly_df, replace=True, n_samples=min_len, random_state=42)

balanced = pd.concat([normal_df, anomaly_df]).sample(frac=1, random_state=42).reset_index(drop=True)
X, y = balanced.drop("label", axis=1), balanced["label"]

# --------------------------
# 🔹 Train Model
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

if model:
    print("🔁 Incrementally Training the Existing Model...")
    model.fit(X_train, y_train)
else:
    print("📌 Training New Model from Scratch...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight={0: 2, 1: 3},
        random_state=42
    )
    model.fit(X_train, y_train)

# --------------------------
# 🔹 Evaluate & Save if Better
# --------------------------
y_pred = model.predict(X_test)
print("📊 Model Performance After Incremental Learning:")
print(classification_report(y_test, y_pred))

accuracy_new = round(accuracy_score(y_test, y_pred), 3)
accuracy_prev = 0.0
if os.path.exists(TRAINING_LOGS):
    with open(TRAINING_LOGS, "r") as f:
        previous_logs = json.load(f)
        if previous_logs:
            accuracy_prev = previous_logs[-1].get("accuracy", 0.0)

if accuracy_new >= accuracy_prev:
    joblib.dump(model, MODEL_PATH)
    print("✅ Model retrained and saved.")
else:
    print("⚠️ Retrained model did not improve. Keeping previous version.")

# --------------------------
# 🔹 Save Training Logs
# --------------------------
log_entry = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "accuracy": round(accuracy_score(y_test, y_pred), 3),
    "precision_0": round(precision_score(y_test, y_pred, pos_label=0), 3),
    "recall_0": round(recall_score(y_test, y_pred, pos_label=0), 3),
    "precision_1": round(precision_score(y_test, y_pred, pos_label=1), 3),
    "recall_1": round(recall_score(y_test, y_pred, pos_label=1), 3)
}

logs = []
if os.path.exists(TRAINING_LOGS):
    with open(TRAINING_LOGS, "r") as f:
        logs = json.load(f)
logs.append(log_entry)
with open(TRAINING_LOGS, "w") as f:
    json.dump(logs, f, indent=2)

print("📜 Logged training results to", TRAINING_LOGS)
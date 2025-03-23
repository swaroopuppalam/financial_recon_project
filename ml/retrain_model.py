import pandas as pd
import joblib
import os
import json
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    accuracy_score
)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample

print("🔄 Retraining Model from Feedback (Integrated Data Mode)...")

# --------------------------
# 🔹 Paths
# --------------------------
FEEDBACK_PATH = "/ui/feedback.json"
MODEL_PATH = "/app/model.pkl"
TRAINING_LOGS = "/shared/training_logs.json"
CONFIG_PATH = "/ml/config.json"  # ✅ NEW
CATALYST_PATH = "/ml/Catalyst_Reconciliation.csv"
REALTIME_PATH = "/ml/real_time_transaction.csv"
HISTORICAL_PATH = "/ml/historical_reconciliation.csv"

# --------------------------
# 🔹 Load model (if exists)
# --------------------------
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

# --------------------------
# 🔹 Load anomaly config (NEW)
# --------------------------
DEFAULT_CONFIG = {"quantity_threshold": 1.0, "price_threshold": 0.01}
if os.path.exists(CONFIG_PATH):
    try:
        with open(CONFIG_PATH, "r") as f:
            config = json.load(f)
        print(f"🧪 Loaded anomaly thresholds: {config}")
    except:
        config = DEFAULT_CONFIG
        print("⚠️ Failed to load config. Using defaults.")
else:
    config = {"anomaly_rules": {"quantity_threshold": 1, "price_threshold": 0.01}}  # Default fallback
    print("⚠️ Config file not found. Using defaults.")

# --------------------------
# 🔹 Load data safely
# --------------------------
def safe_read_csv(path):
    try:
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        return df
    except:
        print(f"⚠️ Warning: Could not read {path}")
        return pd.DataFrame()

historical_df = safe_read_csv(HISTORICAL_PATH)
real_time_df = safe_read_csv(REALTIME_PATH)
catalyst_df = safe_read_csv(CATALYST_PATH)

# --------------------------
# 🔹 Derive anomaly from Catalyst
# --------------------------
if not catalyst_df.empty:
    if "Anomaly" in catalyst_df.columns:
        # ✅ Use existing Anomaly column if present
        catalyst_df["label"] = catalyst_df["Anomaly"].astype(int)
        print("✅ Using existing 'Anomaly' column as label.")
    else:
        # 🔁 Fallback to config-based threshold logic
        threshold_qty = config["anomaly_rules"]["quantity_threshold"]
        threshold_price = config["anomaly_rules"]["price_threshold"]

        catalyst_df["label"] = (
            (catalyst_df["QUANTITYDIFFERENCE"].abs() > threshold_qty) |
            (catalyst_df["PRICEDIFFERENCE"].abs() > threshold_price)
        ).astype(int)
        print("⚠️ 'Anomaly' column not found. Derived using thresholds.")


# --------------------------
# 🔹 Derive anomaly from historical/real-time
# --------------------------
for df in [historical_df, real_time_df]:
    if not df.empty:
        df["label"] = df["Balance Difference"].apply(lambda x: 1 if abs(x) > 1 else 0)

# --------------------------
# 🔹 Load & process feedback.json
# --------------------------
if os.path.exists(FEEDBACK_PATH):
    try:
        feedback = json.load(open(FEEDBACK_PATH))
        df_fb = pd.DataFrame(feedback)
        fb_data = pd.json_normalize(df_fb["input"])
        fb_data["label"] = df_fb["feedback"].apply(lambda x: 1 if x == "Yes" else 0)
    except:
        fb_data = pd.DataFrame()
else:
    fb_data = pd.DataFrame()

# --------------------------
# 🔹 Select columns for training
# --------------------------
feature_cols = ["Balance Difference", "Primary Account", "Secondary Account", "Currency"]
catalyst_cols = ["QUANTITYDIFFERENCE", "PRICEDIFFERENCE"]

frames = []
if not historical_df.empty:
    frames.append(historical_df[feature_cols + ["label"]])
if not real_time_df.empty:
    frames.append(real_time_df[feature_cols + ["label"]])
if not catalyst_df.empty:
    frames.append(catalyst_df[catalyst_cols + ["label"]])
if not fb_data.empty:
    frames.append(fb_data[feature_cols + ["label"]])

if not frames:
    print("❌ No data found. Cannot train model.")
    exit()

# --------------------------
# 🔹 Merge & encode data
# --------------------------
full_data = pd.concat(frames, ignore_index=True)

# Encode categorical
categorical_cols = ["Primary Account", "Secondary Account", "Currency"]
le_map = {}
for col in categorical_cols:
    if col in full_data.columns:
        le = LabelEncoder()
        full_data[col] = le.fit_transform(full_data[col].astype(str))
        le_map[col] = le

# --------------------------
# 🔹 Balance data
# --------------------------
normal_df = full_data[full_data["label"] == 0]
anomaly_df = full_data[full_data["label"] == 1]
if len(anomaly_df) < len(normal_df):
    anomaly_df = resample(anomaly_df, replace=True, n_samples=len(normal_df), random_state=42)

train_data = pd.concat([normal_df, anomaly_df]).sample(frac=1, random_state=42).reset_index(drop=True)
X_resampled = train_data.drop("label", axis=1)
y_resampled = train_data["label"]

# Drop non-numeric
X_resampled = X_resampled.select_dtypes(include=["number"])

# --------------------------
# 🔹 Train/test split & training
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

if model:
    print("🔁 Incrementally Training the Existing Model...")
    model.fit(X_train, y_train)
else:
    print("📌 Training a New Model from Scratch...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight={0: 1, 1: 3},
        random_state=42
    )
    model.fit(X_train, y_train)

# --------------------------
# 🔹 Evaluation
# --------------------------
print("📊 Model Performance After Incremental Learning:")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# --------------------------
# 🔹 Save model
# --------------------------
joblib.dump(model, MODEL_PATH)
print("✅ Model retrained and saved.")

# --------------------------
# 🔹 Save logs
# --------------------------
entry = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "accuracy": round(accuracy_score(y_test, y_pred), 3),
    "precision_0": round(precision_score(y_test, y_pred, pos_label=0), 3),
    "recall_0": round(recall_score(y_test, y_pred, pos_label=0), 3),
    "precision_1": round(precision_score(y_test, y_pred, pos_label=1), 3),
    "recall_1": round(recall_score(y_test, y_pred, pos_label=1), 3)
}
logs = json.load(open(TRAINING_LOGS, "r")) if os.path.exists(TRAINING_LOGS) else []
logs.append(entry)
with open(TRAINING_LOGS, "w") as f:
    json.dump(logs, f, indent=2)

print("📜 Logged training results to", TRAINING_LOGS)

# Save feature importance
feature_importance = dict(zip(X_resampled.columns, model.feature_importances_))
with open("/shared/feature_importance.json", "w") as f:
    json.dump(feature_importance, f, indent=2)
print("📊 Saved feature importances to /shared/feature_importance.json")

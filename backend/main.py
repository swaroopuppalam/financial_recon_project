from fastapi import FastAPI, UploadFile, File
import joblib
import pandas as pd
import os
import json
import threading
import time
from typing import Dict
import traceback

app = FastAPI()

# --------------------------
# 🔹 Paths and Globals
# --------------------------
MODEL_DIR = "/app/models"
CURRENT_MODEL_PATH = "/app/model.pkl"
FEEDBACK_PATH = "/ui/feedback.json"
RETRAIN_SCRIPT = "/ml/retrain_model.py"
CONFIG_PATH = "/ml/config.json"
TRAINING_LOGS_PATH = "/shared/training_logs.json"
UPLOAD_DIR = "/ml/uploads"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --------------------------
# 🔹 Safe model loader
# --------------------------
def safe_load_model(path=CURRENT_MODEL_PATH):
    try:
        if os.path.exists(path) and os.path.getsize(path) > 1024:
            model = joblib.load(path)
            print(f"✅ Model loaded from: {path}")
            return model
        else:
            print(f"⚠️ Model file is missing or too small: {path}")
            return None
    except Exception as e:
        print(f"⚠️ Failed to load model from {path}: {e}")
        traceback.print_exc()
        return None

model = safe_load_model()

# --------------------------
# 🔹 Config Loader
# --------------------------
def load_thresholds():
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH) as f:
                cfg = json.load(f)
                return cfg.get("anomaly_rules", {"quantity_threshold": 1.0, "price_threshold": 0.01, "balance_threshold": 1.0})
        except:
            pass
    return {"quantity_threshold": 1.0, "price_threshold": 0.01, "balance_threshold": 1.0}

# --------------------------
# 🔹 Prediction Endpoint
# --------------------------
@app.post("/detect_anomaly/")
def detect_anomaly(input_data: Dict[str, float]):
    print("✅ Payload:", input_data)
    explanation = []

    try:
        with open("/shared/feature_list.json", "r") as f:
            feature_list = json.load(f)
    except Exception as e:
        return {"error": f"❌ Feature list not found: {e}"}

    df = pd.DataFrame([input_data])
    for col in feature_list:
        if col not in df.columns:
            df[col] = 0.0
    try:
        df = df[feature_list]
    except Exception as e:
        return {"error": f"❌ Failed to align input with trained feature order: {e}"}

    thresholds = load_thresholds()
    bd = abs(df["Balance Difference"].iloc[0]) if "Balance Difference" in df.columns else 0
    qty = abs(df["QUANTITYDIFFERENCE"].iloc[0]) if "QUANTITYDIFFERENCE" in df.columns else 0
    price = abs(df["PRICEDIFFERENCE"].iloc[0]) if "PRICEDIFFERENCE" in df.columns else 0

    if bd >= thresholds.get("balance_threshold", 1.0) or \
       qty > thresholds["quantity_threshold"] or \
       price > thresholds["price_threshold"]:
        explanation.append("⚠️ Exceeds thresholds → flagged as anomaly")
        return {"Anomaly": 1, "explanation": explanation}

    try:
        pred = model.predict(df)[0]
        explanation.append("🧠 Prediction based on ML model")

        try:
            with open("/shared/explanation_map.json") as f:
                exp_map = json.load(f)
            lookup_key = str(tuple(round(float(df.iloc[0][col]), 3) for col in df.columns if df[col].dtype != "O"))
            reason = exp_map.get(lookup_key, "Unknown Reason")
            explanation.append(f"🗂️ Reason Bucket: {reason}")

            if pred == 1 and reason != "Unknown Reason":
                from utils.agent_actions import create_ticket, send_email_alert, create_resolution_task
                anomaly_id = lookup_key[-6:]
                create_ticket(anomaly_id, reason)
                send_email_alert(anomaly_id, reason)
                create_resolution_task(anomaly_id, reason)
                explanation.append("🤖 Agent actions triggered: email, ticket, task")

        except Exception as e:
            explanation.append("❌ Failed to fetch reason bucket")

        return {"Anomaly": int(pred), "explanation": explanation}
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return {"error": f"Prediction failed: {e}"}

# --------------------------
# 🔹 Submit Feedback
# --------------------------
@app.post("/submit_feedback/")
async def submit_feedback(feedback_data: dict):
    try:
        if os.path.exists(FEEDBACK_PATH):
            existing = json.load(open(FEEDBACK_PATH))
        else:
            existing = []
        existing.append(feedback_data)
        with open(FEEDBACK_PATH, "w") as f:
            json.dump(existing, f, indent=2)
        return {"message": "Feedback recorded."}
    except Exception as e:
        return {"error": str(e)}

# --------------------------
# 🔹 Trigger Retrain
# --------------------------
@app.post("/trigger_retrain/")
async def trigger_retrain():
    print("🔁 Manual retrain triggered.")
    threading.Thread(target=retrain_model).start()
    return {"message": "Retrain process started."}

def retrain_model():
    os.system(f"python {RETRAIN_SCRIPT}")
    global model
    model = safe_load_model()

# --------------------------
# 🔹 Upload + Auto Retrain
# --------------------------
@app.post("/upload_dataset/")
async def upload_dataset(file: UploadFile = File(...)):
    path = os.path.join(UPLOAD_DIR, file.filename)
    with open(path, "wb") as f:
        f.write(await file.read())
    print(f"📁 Uploaded dataset to: {path}")
    threading.Thread(target=retrain_model).start()
    return {"message": f"{file.filename} uploaded. Retrain started."}

# --------------------------
# 🔹 List Available Models
# --------------------------
@app.get("/list_models/")
def list_models():
    models = sorted(f for f in os.listdir(MODEL_DIR) if f.endswith(".pkl"))
    return {"models": models}

# --------------------------
# 🔹 Switch Active Model
# --------------------------
@app.post("/switch_model/")
def switch_model(model_name: str):
    global model
    model_path = os.path.join(MODEL_DIR, model_name)
    if not os.path.exists(model_path):
        return {"error": f"Model {model_name} not found"}
    model = safe_load_model(model_path)
    if os.path.exists(CURRENT_MODEL_PATH):
        os.remove(CURRENT_MODEL_PATH)
    os.symlink(model_path, CURRENT_MODEL_PATH)
    return {"message": f"Switched to model: {model_name}"}

# --------------------------
# 🔹 Get Training Logs
# --------------------------
@app.get("/training_logs/")
def get_training_logs():
    if os.path.exists(TRAINING_LOGS_PATH):
        with open(TRAINING_LOGS_PATH) as f:
            return {"logs": json.load(f)}
    return {"logs": []}

# --------------------------
# 🔹 Background Retraining Cycle (Daily)
# --------------------------
def periodic_retrain():
    while True:
        print("⏳ Daily retrain check...")
        os.system(f"python {RETRAIN_SCRIPT}")
        time.sleep(86400)

threading.Thread(target=periodic_retrain, daemon=True).start()

@app.get("/agent_logs/")
def get_agent_logs():
    log_path = "/shared/agent_log.json"
    if os.path.exists(log_path):
        with open(log_path) as f:
            return {"logs": json.load(f)}
    return {"logs": []}
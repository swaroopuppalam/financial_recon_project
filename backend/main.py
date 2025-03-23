from fastapi import FastAPI
import joblib
import pandas as pd
import os
import json
import threading
import time
from typing import Dict

app = FastAPI()

# ✅ Define file paths
MODEL_PATH = "/app/model.pkl"
FEEDBACK_PATH = "/ui/feedback.json"
RETRAIN_SCRIPT = "/ml/retrain_model.py"
CONFIG_PATH = "/ml/config.json"

# ✅ Load the model initially
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    print("⚠️ model.pkl not found, backend will still run but predictions will fail until model is trained.")
    model = None

# ✅ Load anomaly thresholds for prediction post-processing
def load_thresholds():
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r") as f:
                config = json.load(f)
            return config.get("anomaly_rules", {"quantity_threshold": 1, "price_threshold": 0.01})
        except Exception as e:
            print("⚠️ Could not load config.json:", e)
            return {"quantity_threshold": 1, "price_threshold": 0.01}
    return {"quantity_threshold": 1, "price_threshold": 0.01}

config = load_thresholds()

# ✅ Function to retrain the model manually
def retrain_model():
    print("🔄 Retraining model from new feedback...")
    os.system(f"python {RETRAIN_SCRIPT}")
    global model
    model = joblib.load(MODEL_PATH)
    print("✅ Model successfully reloaded after retraining!")

@app.post("/detect_anomaly/")
def detect_anomaly(input_data: Dict[str, float]):
    print("✅ Incoming payload:", input_data)
    df = pd.DataFrame([input_data])

    for col in ["Balance Difference", "Primary Account", "Secondary Account", "Currency",
                "QUANTITYDIFFERENCE", "PRICEDIFFERENCE"]:
        if col not in df.columns:
            df[col] = 0.0

    df = df[["Balance Difference", "Primary Account", "Secondary Account", "Currency",
             "QUANTITYDIFFERENCE", "PRICEDIFFERENCE"]]
    print("🧠 Model input:", df)

    try:
        with open(CONFIG_PATH, "r") as f:
            config = json.load(f)
        qty_thresh = config["anomaly_rules"]["quantity_threshold"]
        price_thresh = config["anomaly_rules"]["price_threshold"]
    except Exception as e:
        print(f"⚠️ Failed to load config. Using fallback. {e}")
        qty_thresh = 1.0
        price_thresh = 0.01

    bd = abs(df["Balance Difference"].iloc[0])
    qty = abs(df["QUANTITYDIFFERENCE"].iloc[0])
    price = abs(df["PRICEDIFFERENCE"].iloc[0])

    explanation = []
    if bd < 1 and qty <= qty_thresh and price <= price_thresh:
        explanation.append("🔎 All values below threshold → not anomalous")
        print("🔎 Below threshold. Classified as non-anomalous by rules.")
        return {
            "Anomaly": 0,
            "explanation": explanation
        }

    try:
        pred = model.predict(df)[0]
        explanation.append("🧠 Prediction based on ML model")
        print("📊 Prediction:", [pred])
        return {
            "Anomaly": int(pred),
            "explanation": explanation
        }
    except Exception as e:
        print("❌ ERROR in backend:", str(e))
        return {"error": str(e)}



@app.post("/submit_feedback/")
async def submit_feedback(feedback_data: dict):
    try:
        print("✅ Received feedback:", feedback_data)

        if os.path.exists(FEEDBACK_PATH):
            with open(FEEDBACK_PATH, "r") as f:
                existing = json.load(f)
        else:
            existing = []

        existing.append(feedback_data)
        with open(FEEDBACK_PATH, "w") as f:
            json.dump(existing, f, indent=2)

        print("✅ Feedback saved to:", FEEDBACK_PATH)
        return {"message": "Feedback recorded."}
    except Exception as e:
        print("❌ ERROR in feedback submission:", str(e))
        return {"error": str(e)}

@app.post("/trigger_retrain/")
async def trigger_retrain():
    print("🔄 Manually triggering model retraining...")
    threading.Thread(target=retrain_model).start()
    return {"message": "Model retraining started. Check logs for updates."}

# 🔄 Automatic background retraining
def periodic_retrain():
    while True:
        print("⏳ Initiating automatic retraining cycle...")
        os.system(f"python {RETRAIN_SCRIPT}")
        time.sleep(86400)  # Daily retrain (adjustable)

threading.Thread(target=periodic_retrain, daemon=True).start()
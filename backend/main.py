from fastapi import FastAPI
import joblib
import pandas as pd
import os
import json
import threading
import time

app = FastAPI()

# ✅ Define file paths
MODEL_PATH = "/app/model.pkl"
FEEDBACK_PATH = "/ui/feedback.json"
RETRAIN_SCRIPT = "/ml/retrain_model.py"

# ✅ Load the model initially
model = joblib.load(MODEL_PATH)

# ✅ Function to retrain the model manually
def retrain_model():
    print("🔄 Retraining model from new feedback...")
    os.system(f"python {RETRAIN_SCRIPT}")  # Runs retrain script
    global model
    model = joblib.load(MODEL_PATH)  # Reload the updated model
    print("✅ Model successfully reloaded after retraining!")

@app.post("/detect_anomaly/")
async def detect_anomaly(transaction: dict):
    try:
        print("✅ Incoming payload:", transaction)

        # Safely extract values
        data = {
            "Balance Difference": float(transaction.get("Balance Difference", 0.0)),
            "Primary Account": int(transaction.get("Primary Account", 0)),
            "Secondary Account": int(transaction.get("Secondary Account", 0)),
            "Currency": int(transaction.get("Currency", 0))
        }

        df = pd.DataFrame([data])
        print("🧠 Model input:", df)

        # ✅ Enforce threshold logic instead of pure ML reliance
        diff = abs(data["Balance Difference"])
        if diff < 1:
            return {"Anomaly": "No", "Details": data}
        elif diff >= 10:
            return {"Anomaly": "Yes", "Details": data}

        # ✅ For intermediate values, use model
        prediction = model.predict(df[["Balance Difference", "Primary Account", "Secondary Account", "Currency"]])
        print("📊 Prediction:", prediction)

        is_anomaly = "Yes" if prediction[0] == 1 else "No"
        return {"Anomaly": is_anomaly, "Details": data}

    except Exception as e:
        print("❌ ERROR in backend:", str(e))
        return {"error": str(e)}


@app.post("/submit_feedback/")
async def submit_feedback(feedback_data: dict):
    try:
        print("✅ Received feedback:", feedback_data)

        # ✅ Save feedback
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

# ✅ **New API to Manually Trigger Retraining**
@app.post("/trigger_retrain/")
async def trigger_retrain():
    print("🔄 Manually triggering model retraining...")
    threading.Thread(target=retrain_model).start()
    return {"message": "Model retraining started. Check logs for updates."}

def periodic_retrain():
    while True:
        print("⏳ Initiating automatic retraining cycle...")
        os.system("python /ml/retrain_model.py")  # Automatically retrains model
        time.sleep(86400)  # Retrain every 24 hours (adjust if needed)

# ✅ Start background retraining thread
threading.Thread(target=periodic_retrain, daemon=True).start()

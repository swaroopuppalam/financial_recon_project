from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# Load Trained Model
model = joblib.load("model.pkl")

@app.post("/detect_anomaly/")
async def detect_anomaly(transaction: dict):
    df = pd.DataFrame([transaction])
    prediction = model.predict(df[["Balance Difference", "Primary Account", "Secondary Account", "Currency"]])
    is_anomaly = "Yes" if prediction[0] == -1 else "No"
    return {"Anomaly": is_anomaly, "Details": transaction}

# Run using: uvicorn main:app --reload

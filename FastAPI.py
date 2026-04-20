from fastapi import FastAPI
from pydantic import BaseModel
import time
import joblib
import numpy as np

app = FastAPI(title="Fraud Detection API", version="1.0")

# -----------------------------
# Request / Response Schemas
# -----------------------------
class TransactionRequest(BaseModel):
    transaction_id: str
    amount: float
    currency: str
    channel: str
    country: str
    merchant_category: str


class FraudResponse(BaseModel):
    fraud_score: float
    status: str
    model_contributions: dict
    processing_time_ms: int


# -----------------------------
# Global model variables
# -----------------------------
xgb_model = None
rf_model = None


# -----------------------------
# Startup: Load models
# -----------------------------
@app.on_event("startup")
def load_models():
    global xgb_model, rf_model

    # Replace with your actual model paths
    xgb_model = joblib.load("models/xgboost_model.pkl")
    rf_model = joblib.load("models/random_forest_model.pkl")

    print("Models loaded successfully.")


# -----------------------------
# Feature Engineering
# -----------------------------
def preprocess_input(data: TransactionRequest):
    """
    Convert input JSON into model-ready numerical features.
    Replace this with your actual feature pipeline used during training.
    """

    # Example encoding (VERY simplified)
    currency_map = {"USD": 1, "CAD": 2, "EUR": 3}
    channel_map = {"online": 1, "instore": 2}
    country_map = {"US": 1, "CA": 2}
    merchant_map = {"electronics": 1, "grocery": 2}

    features = np.array([
        data.amount,
        currency_map.get(data.currency, 0),
        channel_map.get(data.channel, 0),
        country_map.get(data.country, 0),
        merchant_map.get(data.merchant_category, 0)
    ]).reshape(1, -1)

    return features


# -----------------------------
# Scoring Endpoint
# -----------------------------
@app.post("/api/v1/numerical/score", response_model=FraudResponse)
def score_transaction(request: TransactionRequest):
    start_time = time.time()

    try:
        # Preprocess input
        features = preprocess_input(request)

        # Model predictions (probability of fraud class)
        xgb_score = float(xgb_model.predict_proba(features)[0][1])
        rf_score = float(rf_model.predict_proba(features)[0][1])

        # Ensemble (simple average — adjust if needed)
        fraud_score = (xgb_score + rf_score) / 2

        # Clamp to [0,1] just in case
        fraud_score = max(0.0, min(1.0, fraud_score))

        processing_time = int((time.time() - start_time) * 1000)

        return {
            "fraud_score": round(fraud_score, 4),
            "status": "success",
            "model_contributions": {
                "xgboost": round(xgb_score, 4),
                "random_forest": round(rf_score, 4)
            },
            "processing_time_ms": processing_time
        }

    except Exception as e:
        return {
            "fraud_score": 0.0,
            "status": f"error: {str(e)}",
            "model_contributions": {},
            "processing_time_ms": 0
        }
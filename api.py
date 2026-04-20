from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import time

from preprocess import preprocess_data

app = FastAPI()

# Load models ONCE at startup
xgb_model = joblib.load("../models/xgboost_fraud.joblib")
rf_model = joblib.load("../models/rf_fraud.joblib")
feature_columns = joblib.load("../models/feature_columns.joblib")


class TransactionRequest(BaseModel):
    transaction_id: str
    amount: float
    currency: str
    channel: str
    country: str
    merchant_category: str


@app.post("/api/v1/numerical/score")
def score_transaction(request: TransactionRequest):
    start_time = time.time()

    # Convert input to DataFrame
    data = pd.DataFrame([{
        "TransactionAmt": request.amount,
        "ProductCD": request.channel,
        "card1": request.currency,
        "card2": request.country,
        "card3": request.merchant_category
    }])

    # Preprocess
    data = preprocess_data(data, training=False, feature_columns=feature_columns)

    # Predict probabilities
    xgb_prob = xgb_model.predict_proba(data)[:, 1][0]
    rf_prob = rf_model.predict_proba(data)[:, 1][0]

    # Ensemble average
    fraud_score = (xgb_prob + rf_prob) / 2

    processing_time = int((time.time() - start_time) * 1000)

    return {
        "fraud_score": round(float(fraud_score), 4),
        "status": "success",
        "model_contributions": {
            "xgboost": round(float(xgb_prob), 4),
            "random_forest": round(float(rf_prob), 4)
        },
        "processing_time_ms": processing_time
    }
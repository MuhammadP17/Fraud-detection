"""
Fraud Detection API
-------------------
A FastAPI REST endpoint that serves the trained Random Forest fraud detection model.

Usage:
    uvicorn fraud_det_api:app --reload

Endpoint:
    POST /predict
    - Accepts a JSON transaction object with 30 features (Time, V1-V28, Amount)
    - Returns fraud prediction (bool) and fraud probability (float)

Example curl:
    curl -X POST "http://127.0.0.1:8000/predict" \
         -H "Content-Type: application/json" \
         -d '{"Time": 0, "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38,
              "V5": -0.34, "V6": 0.46, "V7": 0.24, "V8": 0.10, "V9": 0.36,
              "V10": 0.09, "V11": -0.55, "V12": -0.62, "V13": -0.99, "V14": -0.31,
              "V15": 1.47, "V16": -0.47, "V17": 0.21, "V18": 0.03, "V19": 0.40,
              "V20": 0.25, "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07,
              "V25": 0.13, "V26": -0.19, "V27": 0.13, "V28": -0.02, "Amount": 149.62}'
"""

import pathlib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib

# ── App initialisation ────────────────────────────────────────────────────────

app = FastAPI(
    title="Fraud Detection API",
    description="Real-time credit card fraud detection using a trained Random Forest classifier.",
    version="1.0.0"
)

# ── Load model ────────────────────────────────────────────────────────────────

MODEL_PATH = pathlib.Path("models") / "fraud_det_model.pkl"

try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    raise RuntimeError(
        f"Model not found at '{MODEL_PATH}'. "
        "Run modelling.ipynb first to generate the model file."
    )

# ── Input schema ──────────────────────────────────────────────────────────────

class Transaction(BaseModel):
    """
    Represents a single credit card transaction.
    Features V1-V28 are PCA-transformed for confidentiality.
    Time and Amount retain their original meaning.
    """
    Time: float
    V1: float;  V2: float;  V3: float;  V4: float;  V5: float
    V6: float;  V7: float;  V8: float;  V9: float;  V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float
    Amount: float

    class Config:
        json_schema_extra = {
            "example": {
                "Time": 0, "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38,
                "V5": -0.34, "V6": 0.46, "V7": 0.24, "V8": 0.10, "V9": 0.36,
                "V10": 0.09, "V11": -0.55, "V12": -0.62, "V13": -0.99, "V14": -0.31,
                "V15": 1.47, "V16": -0.47, "V17": 0.21, "V18": 0.03, "V19": 0.40,
                "V20": 0.25, "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07,
                "V25": 0.13, "V26": -0.19, "V27": 0.13, "V28": -0.02, "Amount": 149.62
            }
        }

# ── Output schema ─────────────────────────────────────────────────────────────

class PredictionResult(BaseModel):
    fraud_prediction: bool
    fraud_probability: float
    risk_level: str

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "Fraud Detection API is running. POST to /predict to score a transaction."}


@app.post("/predict", response_model=PredictionResult)
def predict(transaction: Transaction):
    """
    Predicts whether a transaction is fraudulent.

    Returns:
    - **fraud_prediction**: True if the model predicts fraud
    - **fraud_probability**: Model's confidence score (0.0 – 1.0)
    - **risk_level**: Human-readable risk label (Low / Medium / High)
    """
    try:
        input_data = np.array([[
            transaction.Time,
            transaction.V1,  transaction.V2,  transaction.V3,  transaction.V4,
            transaction.V5,  transaction.V6,  transaction.V7,  transaction.V8,
            transaction.V9,  transaction.V10, transaction.V11, transaction.V12,
            transaction.V13, transaction.V14, transaction.V15, transaction.V16,
            transaction.V17, transaction.V18, transaction.V19, transaction.V20,
            transaction.V21, transaction.V22, transaction.V23, transaction.V24,
            transaction.V25, transaction.V26, transaction.V27, transaction.V28,
            transaction.Amount
        ]])

        prediction = model.predict(input_data)[0]
        fraud_prob = float(model.predict_proba(input_data)[0][1])

        if fraud_prob < 0.3:
            risk_level = "Low"
        elif fraud_prob < 0.6:
            risk_level = "Medium"
        else:
            risk_level = "High"

        return PredictionResult(
            fraud_prediction=bool(prediction),
            fraud_probability=round(fraud_prob, 4),
            risk_level=risk_level
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

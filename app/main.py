"""
app/main.py
-----------
FastAPI model-serving API (Task 6) with request logging + Prometheus metrics
(Task 8).

Endpoints:
  GET  /health   -> liveness check
  POST /predict  -> JSON patient features in, {prediction, confidence} out
  GET  /metrics  -> Prometheus metrics (added by the instrumentator)

The model (models/model.joblib) is a full sklearn Pipeline, so it applies the
SAME preprocessing used in training -> no train/serve skew.
"""
import logging
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("heart-api")

MODEL_PATH = Path("models/model.joblib")
app = FastAPI(title="Heart Disease Risk API", version="1.0.0")

# Load the model once at startup.
model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


class Patient(BaseModel):
    """Input schema — one patient record. Field names match training columns."""
    age: float = Field(..., example=63)
    sex: float = Field(..., example=1)
    cp: float = Field(..., example=1)
    trestbps: float = Field(..., example=145)
    chol: float = Field(..., example=233)
    fbs: float = Field(..., example=1)
    restecg: float = Field(..., example=2)
    thalach: float = Field(..., example=150)
    exang: float = Field(..., example=0)
    oldpeak: float = Field(..., example=2.3)
    slope: float = Field(..., example=3)
    ca: float = Field(..., example=0)
    thal: float = Field(..., example=6)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(patient: Patient):
    if model is None:
        return {"error": "model not loaded"}
    X = pd.DataFrame([patient.model_dump()])
    proba = float(model.predict_proba(X)[0, 1])
    pred = int(proba >= 0.5)
    logger.info("predict request=%s -> pred=%s conf=%.4f",
                patient.model_dump(), pred, proba)
    return {
        "prediction": pred,                       # 1 = disease risk, 0 = no disease
        "label": "disease" if pred else "no_disease",
        "confidence": round(proba if pred else 1 - proba, 4),
    }


# Expose /metrics for Prometheus scraping.
Instrumentator().instrument(app).expose(app)

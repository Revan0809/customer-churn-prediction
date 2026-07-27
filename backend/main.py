"""
FastAPI backend for the Customer Churn Prediction app.

Everything runs locally against artifacts produced by src/train.py and
src/explain.py -- there are no external API calls and no API keys.

Endpoints:
  POST /predict     -> churn prediction + probability + top 3 SHAP factors
  GET  /model-info   -> saved model comparison metrics, for the frontend table
  GET  /health        -> simple liveness check
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Allow `import src...` when running as `uvicorn backend.main:app` from the
# project root.
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    COMPARISON_CSV_PATH,
    METADATA_PATH,
    MODEL_PATH,
    PREPROCESSOR_PATH,
    SHAP_BACKGROUND_PATH,
)
from src.explain import build_explainer, get_top_contributing_features
from src.preprocessing import load_preprocessor, transform_record

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predicts telecom customer churn from a trained scikit-learn/XGBoost model.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local demo app; tighten this for a real deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
class CustomerInput(BaseModel):
    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0, le=100, description="Months with the company")
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0, le=1000)
    TotalCharges: float = Field(ge=0, le=100000)

    model_config = {
        "json_schema_extra": {
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 85.5,
                "TotalCharges": 1020.0,
            }
        }
    }


class TopFactor(BaseModel):
    feature: str
    impact: float
    direction: str


class PredictResponse(BaseModel):
    churn_prediction: Literal["Yes", "No"]
    churn_probability: float
    top_factors: list[TopFactor]


class ModelMetrics(BaseModel):
    model: str
    precision: float
    recall: float
    f1: float
    roc_auc: float


class ModelInfoResponse(BaseModel):
    best_model_name: str
    comparison: list[ModelMetrics]


# ---------------------------------------------------------------------------
# Startup: load everything once, not per-request
# ---------------------------------------------------------------------------
state: dict = {}


@app.on_event("startup")
def load_artifacts():
    missing = [
        p for p in [MODEL_PATH, PREPROCESSOR_PATH, METADATA_PATH] if not p.exists()
    ]
    if missing:
        missing_list = ", ".join(str(p) for p in missing)
        raise RuntimeError(
            f"Missing trained model artifacts: {missing_list}. "
            "Run `python -m src.train` (and optionally `python -m src.explain`) "
            "from the project root first."
        )

    state["preprocessor"] = load_preprocessor(PREPROCESSOR_PATH)
    state["model"] = joblib.load(MODEL_PATH)

    with open(METADATA_PATH) as f:
        state["metadata"] = json.load(f)
    state["feature_names"] = state["metadata"]["feature_names"]

    if SHAP_BACKGROUND_PATH.exists():
        background = joblib.load(SHAP_BACKGROUND_PATH)
    else:
        background = None
    state["explainer"] = build_explainer(state["model"], background)

    if COMPARISON_CSV_PATH.exists():
        import pandas as pd

        state["comparison_df"] = pd.read_csv(COMPARISON_CSV_PATH)
    else:
        state["comparison_df"] = None

    print(f"Loaded model '{state['metadata']['best_model_name']}' and preprocessor.")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    if state.get("comparison_df") is None:
        raise HTTPException(
            status_code=503,
            detail="Model comparison data not found. Run `python -m src.train` first.",
        )

    df = state["comparison_df"]
    comparison = [
        ModelMetrics(
            model=row["model"],
            precision=row["precision"],
            recall=row["recall"],
            f1=row["f1"],
            roc_auc=row["roc_auc"],
        )
        for _, row in df.iterrows()
    ]
    return ModelInfoResponse(
        best_model_name=state["metadata"]["best_model_name"],
        comparison=comparison,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(customer: CustomerInput):
    if "model" not in state:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        record = customer.model_dump()
        x = transform_record(state["preprocessor"], record)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not process input: {exc}")

    model = state["model"]
    proba = float(model.predict_proba(x)[0, 1])
    prediction = "Yes" if proba >= 0.5 else "No"

    try:
        top_factors_raw = get_top_contributing_features(
            state["explainer"], x, state["feature_names"], top_n=3
        )
        top_factors = [TopFactor(**item) for item in top_factors_raw]
    except Exception as exc:
        # Explainability is best-effort: a SHAP failure shouldn't take down
        # the core prediction the user actually asked for.
        print(f"SHAP explanation failed: {exc}", file=sys.stderr)
        top_factors = []

    return PredictResponse(
        churn_prediction=prediction,
        churn_probability=round(proba, 4),
        top_factors=top_factors,
    )

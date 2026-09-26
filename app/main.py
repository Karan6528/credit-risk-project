"""
FastAPI service for real-time credit default scoring.

Run locally:
    uvicorn app.main:app --reload

Docs at http://localhost:8000/docs once running.
"""
import os
import sys
import logging
from datetime import datetime, timezone

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features import engineer_features, build_preprocessing_columns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("credit-risk-api")

MODEL_PATH = os.environ.get("MODEL_PATH", "models/model.pkl")
DECISION_THRESHOLD = float(os.environ.get("DECISION_THRESHOLD", "0.5"))

app = FastAPI(
    title="Credit Risk Scoring API",
    description="Predicts probability of loan default and returns a decision with explanation.",
    version="1.0.0",
)

model = None  # loaded on startup


class ApplicantRequest(BaseModel):
    annual_income: float = Field(..., gt=0, example=65000)
    loan_amnt: float = Field(..., gt=0, example=12000)
    term: int = Field(..., description="36 or 60 months", example=36)
    int_rate: float = Field(..., ge=0, le=40, example=13.5)
    dti: float = Field(..., ge=0, example=18.2)
    revol_util: float = Field(..., ge=0, le=100, example=42.0)
    open_acc: int = Field(..., ge=0, example=9)
    delinq_2yrs: int = Field(..., ge=0, example=0)
    fico_score: float = Field(..., ge=300, le=850, example=712)
    emp_length: int = Field(..., ge=0, le=10, example=5)
    home_ownership: str = Field(..., example="MORTGAGE")
    purpose: str = Field(..., example="debt_consolidation")


class ScoreResponse(BaseModel):
    probability_of_default: float
    decision: str
    threshold_used: float
    top_risk_factors: list
    scored_at: str
    model_version: str


@app.on_event("startup")
def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        logger.warning(f"No model found at {MODEL_PATH}. Run src/train.py first.")
        return
    model = joblib.load(MODEL_PATH)
    logger.info(f"Loaded model from {MODEL_PATH}")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/score", response_model=ScoreResponse)
def score_applicant(applicant: ApplicantRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train a model first.")

    raw = pd.DataFrame([applicant.dict()])
    engineered = engineer_features(raw)
    cols = build_preprocessing_columns()
    X = engineered[cols["numeric"] + cols["categorical"]]

    proba = float(model.predict_proba(X)[0, 1])
    decision = "DECLINE" if proba >= DECISION_THRESHOLD else "APPROVE"

    # Lightweight explanation without recomputing a full SHAP explainer
    # per-request (expensive). For production, precompute a background
    # explainer at startup -- see src/explain.py for the full version.
    top_factors = _quick_risk_flags(applicant)

    logger.info(f"Scored applicant: proba={proba:.4f} decision={decision}")

    return ScoreResponse(
        probability_of_default=round(proba, 4),
        decision=decision,
        threshold_used=DECISION_THRESHOLD,
        top_risk_factors=top_factors,
        scored_at=datetime.now(timezone.utc).isoformat(),
        model_version="1.0.0",
    )


def _quick_risk_flags(applicant: ApplicantRequest) -> list:
    """Rule-based flags shown alongside the model score. This is not a
    substitute for the SHAP explanations in src/explain.py -- it's a
    fast, dependency-light summary for the API response."""
    flags = []
    if applicant.dti > 30:
        flags.append("High debt-to-income ratio")
    if applicant.revol_util > 75:
        flags.append("High credit utilization")
    if applicant.delinq_2yrs > 0:
        flags.append("Recent delinquency on record")
    if applicant.fico_score < 640:
        flags.append("Below-average FICO score")
    if not flags:
        flags.append("No major risk flags detected")
    return flags

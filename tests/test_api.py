"""
Basic tests for the scoring API.

Run: pytest tests/
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

SAMPLE_APPLICANT = {
    "annual_income": 65000,
    "loan_amnt": 12000,
    "term": 36,
    "int_rate": 13.5,
    "dti": 18.2,
    "revol_util": 42.0,
    "open_acc": 9,
    "delinq_2yrs": 0,
    "fico_score": 712,
    "emp_length": 5,
    "home_ownership": "MORTGAGE",
    "purpose": "debt_consolidation",
}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()


def test_score_valid_applicant():
    response = client.post("/score", json=SAMPLE_APPLICANT)
    # Will be 503 if no model has been trained yet in this environment
    assert response.status_code in (200, 503)
    if response.status_code == 200:
        body = response.json()
        assert 0.0 <= body["probability_of_default"] <= 1.0
        assert body["decision"] in ("APPROVE", "DECLINE")


def test_score_rejects_invalid_input():
    bad_applicant = dict(SAMPLE_APPLICANT)
    bad_applicant["fico_score"] = 1000  # out of valid range
    response = client.post("/score", json=bad_applicant)
    assert response.status_code == 422  # pydantic validation error


def test_score_missing_field():
    incomplete = dict(SAMPLE_APPLICANT)
    del incomplete["dti"]
    response = client.post("/score", json=incomplete)
    assert response.status_code == 422

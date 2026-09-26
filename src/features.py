"""
Feature engineering for the credit risk model.

Keeps transformation logic in one place so training and inference
(the FastAPI app) apply exactly the same steps -- a common source of
train/serve skew if left duplicated.
"""
import pandas as pd

CATEGORICAL_COLS = ["home_ownership", "purpose"]
NUMERIC_COLS = [
    "annual_income", "loan_amnt", "term", "int_rate", "dti",
    "revol_util", "open_acc", "delinq_2yrs", "fico_score", "emp_length",
]
ENGINEERED_COLS = [
    "loan_to_income", "income_per_open_acc", "high_utilization_flag",
    "recent_delinquency_flag", "risk_term_interaction",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain-informed features on top of the raw columns."""
    out = df.copy()

    out["loan_to_income"] = out["loan_amnt"] / out["annual_income"].clip(lower=1)
    out["income_per_open_acc"] = out["annual_income"] / out["open_acc"].clip(lower=1)
    out["high_utilization_flag"] = (out["revol_util"] > 75).astype(int)
    out["recent_delinquency_flag"] = (out["delinq_2yrs"] > 0).astype(int)
    # 60-month terms carry more risk per unit of DTI than 36-month terms
    out["risk_term_interaction"] = out["dti"] * (out["term"] == 60).astype(int)

    return out


def build_preprocessing_columns():
    """Returns the column groups the sklearn ColumnTransformer needs."""
    return {
        "numeric": NUMERIC_COLS + ENGINEERED_COLS,
        "categorical": CATEGORICAL_COLS,
    }

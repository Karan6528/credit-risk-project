"""
Training pipeline for the credit default model.

Trains two models for comparison:
  1. Logistic regression  -- interpretable baseline
  2. XGBoost              -- higher-capacity challenger

Both are:
  - tracked with MLflow (params, metrics, artifacts)
  - evaluated with business-relevant metrics (expected cost, not just AUC)
  - calibrated so predicted probabilities are meaningful, not just
    good for ranking

Usage:
    python src/train.py --data data/credit_data.csv
    mlflow ui   # then open http://localhost:5000 to compare runs
"""
import argparse
import json
import sys
import os

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    precision_recall_curve, confusion_matrix,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features import engineer_features, build_preprocessing_columns

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


# --- Business cost assumptions -------------------------------------------
# A missed default (false negative) costs far more than a wrongly declined
# good loan (false positive): you lose the principal vs. lose some interest
# margin. These ratios drive threshold selection -- tune them to your
# actual business case.
COST_FALSE_NEGATIVE = 10.0  # relative cost of approving a loan that defaults
COST_FALSE_POSITIVE = 1.0   # relative cost of declining a loan that was fine


def build_pipeline(model, numeric_cols, categorical_cols):
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ])
    return Pipeline([
        ("preprocess", preprocessor),
        ("model", model),
    ])


def find_optimal_threshold(y_true, y_proba):
    """Pick the decision threshold that minimizes expected business cost,
    rather than defaulting to the naive 0.5 cutoff."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    best_cost, best_threshold = np.inf, 0.5
    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        cost = fn * COST_FALSE_NEGATIVE + fp * COST_FALSE_POSITIVE
        if cost < best_cost:
            best_cost, best_threshold = cost, t
    return best_threshold, best_cost


def evaluate(y_true, y_proba, threshold):
    preds = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    return {
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "brier_score": brier_score_loss(y_true, y_proba),  # calibration quality
        "threshold": float(threshold),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "expected_cost": float(fn * COST_FALSE_NEGATIVE + fp * COST_FALSE_POSITIVE),
    }


def run(data_path: str, experiment_name: str = "credit-risk"):
    mlflow.set_experiment(experiment_name)

    df = pd.read_csv(data_path)
    df = engineer_features(df)
    cols = build_preprocessing_columns()

    X = df[cols["numeric"] + cols["categorical"]]
    y = df["default"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f"Train default rate: {y_train.mean():.2%}  |  Test default rate: {y_test.mean():.2%}")

    results = {}

    # --- Model 1: Logistic Regression baseline ---
    with mlflow.start_run(run_name="logistic_regression_baseline"):
        pipe = build_pipeline(
            LogisticRegression(max_iter=1000, class_weight="balanced"),
            cols["numeric"], cols["categorical"],
        )
        pipe.fit(X_train, y_train)
        y_proba = pipe.predict_proba(X_test)[:, 1]
        threshold, _ = find_optimal_threshold(y_test.values, y_proba)
        metrics = evaluate(y_test.values, y_proba, threshold)

        mlflow.log_param("model_type", "logistic_regression")
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        mlflow.sklearn.log_model(pipe, "model")
        results["logistic_regression"] = metrics
        print("Logistic Regression:", json.dumps(metrics, indent=2))

    # --- Model 2: XGBoost challenger ---
    if HAS_XGB:
        with mlflow.start_run(run_name="xgboost_challenger"):
            scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
            xgb = XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                eval_metric="auc", random_state=42,
            )
            pipe = build_pipeline(xgb, cols["numeric"], cols["categorical"])
            pipe.fit(X_train, y_train)

            # Calibrate probabilities: tree ensembles are good at ranking but
            # often poorly calibrated (overconfident) out of the box.
            calibrated = CalibratedClassifierCV(pipe, method="isotonic", cv=3)
            calibrated.fit(X_train, y_train)

            y_proba = calibrated.predict_proba(X_test)[:, 1]
            threshold, _ = find_optimal_threshold(y_test.values, y_proba)
            metrics = evaluate(y_test.values, y_proba, threshold)

            mlflow.log_param("model_type", "xgboost")
            mlflow.log_param("scale_pos_weight", scale_pos_weight)
            mlflow.log_param("calibration", "isotonic")
            mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
            mlflow.sklearn.log_model(calibrated, "model")
            results["xgboost"] = metrics
            print("XGBoost:", json.dumps(metrics, indent=2))

            os.makedirs("models", exist_ok=True)
            import joblib
            joblib.dump(calibrated, "models/model.pkl")
    else:
        print("xgboost not installed -- skipping challenger model. `pip install xgboost` to enable.")

        os.makedirs("reports", exist_ok=True)
    with open("reports/training_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/credit_data.csv")
    parser.add_argument("--experiment", type=str, default="credit-risk")
    args = parser.parse_args()
    run(args.data, args.experiment)

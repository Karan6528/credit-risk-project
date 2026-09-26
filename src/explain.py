"""
Model explainability using SHAP.

In credit risk specifically, this isn't optional polish: regulations
(e.g. adverse action notices under the US Equal Credit Opportunity Act)
require lenders to give applicants specific reasons for a denial. A
model that can't explain individual predictions can't legally be used
for lending decisions as-is.

Usage:
    python src/explain.py --data data/credit_data.csv --model models/model.pkl
"""
import argparse
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features import engineer_features, build_preprocessing_columns


def explain_global(model, X_sample, out_path="reports/shap_summary.png"):
    """Global feature importance across a sample of applicants."""
    # model here is the sklearn Pipeline (preprocess + estimator) inside
    # the calibrated wrapper; unwrap to get raw feature-space SHAP values.
    preprocess = model.calibrated_classifiers_[0].estimator.named_steps["preprocess"]
    estimator = model.calibrated_classifiers_[0].estimator.named_steps["model"]

    X_transformed = preprocess.transform(X_sample)
    feature_names = preprocess.get_feature_names_out()

    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(X_transformed)

    plt.figure()
    shap.summary_plot(shap_values, X_transformed, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved global SHAP summary to {out_path}")
    return explainer, shap_values, feature_names


def explain_single_applicant(model, applicant_row: pd.DataFrame, explainer, feature_names):
    """
    Produce a human-readable explanation for one applicant -- this is
    the function the API calls to generate adverse-action-style reasons.
    """
    preprocess = model.calibrated_classifiers_[0].estimator.named_steps["preprocess"]
    X_transformed = preprocess.transform(applicant_row)
    shap_values = explainer.shap_values(X_transformed)[0]

    contributions = sorted(
        zip(feature_names, shap_values), key=lambda x: abs(x[1]), reverse=True
    )
    top_reasons = [
        {"feature": f, "impact": round(float(v), 4), "direction": "increases risk" if v > 0 else "decreases risk"}
        for f, v in contributions[:5]
    ]
    return top_reasons


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/credit_data.csv")
    parser.add_argument("--model", type=str, default="models/model.pkl")
    parser.add_argument("--sample_size", type=int, default=500)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    df = engineer_features(df)
    cols = build_preprocessing_columns()
    X = df[cols["numeric"] + cols["categorical"]].sample(args.sample_size, random_state=42)

    model = joblib.load(args.model)
    explainer, shap_values, feature_names = explain_global(model, X)

    single = X.iloc[[0]]
    reasons = explain_single_applicant(model, single, explainer, feature_names)
    print("\nExample single-applicant explanation:")
    for r in reasons:
        print(f"  {r['feature']}: {r['impact']:+.4f} ({r['direction']})")

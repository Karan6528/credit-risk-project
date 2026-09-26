"""
Data drift and model performance monitoring.

In production, the population applying for loans shifts over time
(economic conditions, marketing changes, seasonality). A model trained
on last year's applicants can quietly degrade without ever throwing an
error -- it just gets worse at its job. This script compares a
"reference" dataset (what the model was trained on) against "current"
data (recent applicants) and flags meaningful shifts.

Usage:
    python monitoring/drift_check.py \\
        --reference data/credit_data.csv \\
        --current data/recent_applicants.csv \\
        --out reports/drift_report.html

In a real deployment, run this on a schedule (e.g. weekly via GitHub
Actions or Airflow) and alert if drift_detected is True.
"""
import argparse
import json
import pandas as pd

try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
    HAS_EVIDENTLY = True
except ImportError:
    HAS_EVIDENTLY = False


FEATURE_COLS = [
    "annual_income", "loan_amnt", "term", "int_rate", "dti",
    "revol_util", "open_acc", "delinq_2yrs", "fico_score", "emp_length",
]


def run_drift_check(reference_path: str, current_path: str, out_path: str):
    if not HAS_EVIDENTLY:
        print("evidently not installed. `pip install evidently` to enable full drift reports.")
        return simple_drift_check(reference_path, current_path)

    reference = pd.read_csv(reference_path)
    current = pd.read_csv(current_path)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference[FEATURE_COLS], current_data=current[FEATURE_COLS])
    report.save_html(out_path)

    result = report.as_dict()
    drift_detected = result["metrics"][0]["result"]["dataset_drift"]
    print(f"Drift detected: {drift_detected}")
    print(f"Full report saved to {out_path}")
    return drift_detected


def simple_drift_check(reference_path: str, current_path: str, threshold: float = 0.1):
    """
    Fallback drift check using population stability index (PSI) if
    evidently isn't installed -- no extra dependencies required.
    PSI > 0.1 suggests moderate shift; PSI > 0.25 suggests major shift.
    """
    import numpy as np

    reference = pd.read_csv(reference_path)
    current = pd.read_csv(current_path)

    psi_results = {}
    for col in FEATURE_COLS:
        if col not in reference.columns or col not in current.columns:
            continue
        ref_vals, cur_vals = reference[col].dropna(), current[col].dropna()
        breakpoints = np.quantile(ref_vals, np.linspace(0, 1, 11))
        breakpoints[-1] += 1e-6  # ensure max value is included
        ref_counts, _ = np.histogram(ref_vals, bins=breakpoints)
        cur_counts, _ = np.histogram(cur_vals, bins=breakpoints)

        ref_pct = np.clip(ref_counts / ref_counts.sum(), 1e-6, None)
        cur_pct = np.clip(cur_counts / cur_counts.sum(), 1e-6, None)
        psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
        psi_results[col] = round(psi, 4)

    drifted_features = {k: v for k, v in psi_results.items() if v > threshold}
    print(json.dumps(psi_results, indent=2))
    print(f"\nFeatures with PSI > {threshold}: {list(drifted_features.keys()) or 'none'}")
    return len(drifted_features) > 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=str, default="data/credit_data.csv")
    parser.add_argument("--current", type=str, required=True)
    parser.add_argument("--out", type=str, default="reports/drift_report.html")
    args = parser.parse_args()
    run_drift_check(args.reference, args.current, args.out)

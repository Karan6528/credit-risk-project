# Credit Risk Scoring System

An end-to-end credit default prediction system, built to show the full
lifecycle of an ML product — not just a notebook that ends at `model.fit()`.

## The business problem

Lenders need to decide whether to approve a loan application. Two kinds
of mistakes are possible, and they are **not equally costly**:

- **Approve a loan that defaults** → lose the principal. Expensive.
- **Decline an application that would have been fine** → lose some
  interest margin. Cheap by comparison.

This project treats the model's decision threshold as a business
parameter to optimize, not a default `0.5` cutoff — see
`find_optimal_threshold()` in `src/train.py`.

## Architecture

```
Raw applicant data
      │
      ▼
┌─────────────────┐     ┌──────────────────┐     ┌───────────────────┐
│ Feature          │────▶│ Model training    │────▶│ MLflow tracking    │
│ engineering       │     │ (LogReg + XGBoost)│     │ (params, metrics)  │
│ (src/features.py) │     │ (src/train.py)    │     │                    │
└─────────────────┘     └──────────────────┘     └───────────────────┘
                                  │
                                  ▼
                     ┌────────────────────────┐
                     │ Calibration + SHAP       │
                     │ explainability            │
                     │ (src/explain.py)          │
                     └────────────────────────┘
                                  │
                                  ▼
                     ┌────────────────────────┐        ┌─────────────────────┐
                     │ FastAPI serving           │──────▶│ Docker container      │
                     │ (app/main.py)             │       │                       │
                     └────────────────────────┘        └─────────────────────┘
                                  │
                                  ▼
                     ┌────────────────────────┐
                     │ Drift & performance       │
                     │ monitoring (Evidently/PSI) │
                     │ (monitoring/drift_check.py)│
                     └────────────────────────┘
```

## Project structure

```
credit-risk-project/
├── data/
│   └── generate_synthetic_data.py   # synthetic data generator (see note below)
├── notebooks/
│   └── 01_eda.ipynb                 # exploratory analysis
├── src/
│   ├── features.py                  # shared feature engineering (train + serve)
│   ├── train.py                     # training pipeline + MLflow tracking
│   └── explain.py                   # SHAP-based explainability
├── app/
│   └── main.py                      # FastAPI scoring service
├── monitoring/
│   └── drift_check.py               # data drift detection
├── tests/
│   └── test_api.py                  # API tests
├── .github/workflows/ci.yml         # CI: train + test + build on every push
├── Dockerfile
└── requirements.txt
```

## A note on the data

This project ships with a **synthetic data generator**
(`data/generate_synthetic_data.py`) that mimics the structure and
feature-risk relationships of the public [Lending Club loan
dataset](https://www.kaggle.com/datasets/wordsforthewise/lending-club),
so the whole pipeline runs end-to-end with no external download.

To use the real dataset instead: download the Lending Club "accepted
loans" CSV from Kaggle, and point `src/train.py --data` at it. Column
names in `src/features.py` were chosen to match that dataset's schema,
so the swap should need minimal changes.

## Quickstart

```bash
pip install -r requirements.txt

# 1. Generate data (or drop in the real Lending Club CSV as data/credit_data.csv)
python data/generate_synthetic_data.py --n 50000 --out data/credit_data.csv

# 2. Train models (logs to MLflow, saves the calibrated model to models/model.pkl)
python src/train.py --data data/credit_data.csv
mlflow ui   # inspect runs at http://localhost:5000

# 3. Generate SHAP explanations
python src/explain.py --data data/credit_data.csv --model models/model.pkl

# 4. Serve the model
uvicorn app.main:app --reload
# docs at http://localhost:8000/docs

# 5. Or run it containerized
docker build -t credit-risk-api .
docker run -p 8000:8000 credit-risk-api

# 6. Check for data drift against a new batch of applicants
python monitoring/drift_check.py --current data/recent_applicants.csv
```

## Design decisions worth highlighting (for reviewers / interviews)

- **Cost-sensitive thresholding, not accuracy.** With a ~3-6% default
  rate, accuracy is close to meaningless — predicting "no default" for
  every applicant already scores >94%. The threshold is chosen to
  minimize expected business cost using an explicit false-negative /
  false-positive cost ratio.
- **Calibration, not just ranking.** XGBoost is calibrated with
  isotonic regression because tree ensembles tend to produce
  overconfident probabilities — and a probability that isn't a real
  probability isn't usable for expected-loss calculations.
- **Shared feature engineering code** (`src/features.py`) is imported
  by both training and serving, to avoid train/serve skew — a common,
  hard-to-debug source of production model failures.
- **Explainability is treated as a requirement, not a nice-to-have.**
  In consumer lending, adverse action notices are often a legal
  requirement, not optional polish.
- **Monitoring included from day one.** A model's performance decays
  as the applicant population shifts; `monitoring/drift_check.py`
  quantifies that shift with population stability index (PSI) or
  Evidently, so degradation is caught before it becomes a business
  problem, not after.

## What I'd add next

- Feature store to serve consistent features at training and inference time
- A/B testing framework for comparing model versions in production
- Fairness audits across protected demographic groups (a real requirement in lending)
- Retraining pipeline triggered automatically by the drift monitor

## Results

Run `python src/train.py` and see `reports/training_results.json` for
metrics from the most recent run (ROC-AUC, PR-AUC, Brier score,
confusion matrix, and expected cost at the optimal threshold).

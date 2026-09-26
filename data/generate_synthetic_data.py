"""
Synthetic credit risk dataset generator.

This generates a dataset that mimics the structure and statistical
properties of the Lending Club loan dataset (a common public dataset
for credit risk projects), including class imbalance and realistic
feature correlations with default risk.

Why synthetic instead of the real dataset:
    This lets the whole pipeline run end-to-end with no external
    downloads. To use real data instead, download the Lending Club
    "accepted loans" CSV (e.g. from Kaggle: wordsforthewise/lending-club)
    and point src/train.py at it -- the column names below were chosen
    to match that dataset so the swap is close to a drop-in replacement.

Usage:
    python data/generate_synthetic_data.py --n 50000 --out data/credit_data.csv
"""
import argparse
import numpy as np
import pandas as pd


def generate(n: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    annual_income = rng.lognormal(mean=10.8, sigma=0.5, size=n).clip(15_000, 400_000)
    loan_amnt = rng.uniform(1_000, 40_000, size=n)
    term = rng.choice([36, 60], size=n, p=[0.7, 0.3])
    int_rate = rng.normal(13, 4, size=n).clip(5, 30)
    dti = rng.gamma(shape=2.0, scale=8.0, size=n).clip(0, 60)  # debt-to-income ratio
    revol_util = rng.uniform(0, 100, size=n)  # revolving credit utilization %
    open_acc = rng.poisson(9, size=n).clip(1, 30)
    delinq_2yrs = rng.poisson(0.3, size=n).clip(0, 10)
    fico_score = rng.normal(700, 45, size=n).clip(580, 850)
    emp_length = rng.integers(0, 11, size=n)  # years, 10 = 10+
    home_ownership = rng.choice(["RENT", "MORTGAGE", "OWN"], size=n, p=[0.4, 0.45, 0.15])
    purpose = rng.choice(
        ["debt_consolidation", "credit_card", "home_improvement", "small_business", "other"],
        size=n, p=[0.45, 0.2, 0.15, 0.1, 0.1]
    )

    # Latent default risk as a function of the features above (this is what
    # a real model has to recover). Coefficients are illustrative, not fitted
    # to real-world data.
    logit = (
        -4.0
        + 0.038 * (dti - 18)
        + 0.028 * (revol_util - 40)
        - 0.022 * (fico_score - 700)
        + 0.35 * delinq_2yrs
        + 0.00004 * (loan_amnt - 12000)
        - 0.00002 * (annual_income - 65000) / 10
        + 0.10 * (term == 60).astype(float)
        - 0.12 * (emp_length)
        + rng.normal(0, 0.6, size=n)  # irreducible noise
    )
    prob_default = 1 / (1 + np.exp(-logit))
    default = rng.binomial(1, prob_default)

    df = pd.DataFrame({
        "annual_income": annual_income.round(2),
        "loan_amnt": loan_amnt.round(2),
        "term": term,
        "int_rate": int_rate.round(2),
        "dti": dti.round(2),
        "revol_util": revol_util.round(2),
        "open_acc": open_acc,
        "delinq_2yrs": delinq_2yrs,
        "fico_score": fico_score.round(0),
        "emp_length": emp_length,
        "home_ownership": home_ownership,
        "purpose": purpose,
        "default": default,
    })
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50_000)
    parser.add_argument("--out", type=str, default="data/credit_data.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = generate(args.n, args.seed)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df):,} rows to {args.out}")
    print(f"Default rate: {df['default'].mean():.2%}")

#!/usr/bin/env python3
"""Generate a SEEDED SYNTHETIC customer dataset for churn modeling.

Every row is synthetic — no real customers, no PII. The churn label is drawn
from a documented logistic model so the signal is realistic (short tenure,
many tickets, late payments and month-to-month contracts raise churn risk)
and fully reproducible given --seed.

Columns: tenure_months, monthly_spend, support_tickets, late_payments,
         contract_type, churn
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "data" / "customers.csv"

CONTRACTS = ["month-to-month", "one-year", "two-year"]
CONTRACT_P = [0.50, 0.30, 0.20]
CONTRACT_EFFECT = {"month-to-month": 1.30, "one-year": 0.25, "two-year": -0.40}


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def generate(n: int = 4000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 73, size=n)  # months, 1..72
    spend = np.clip(rng.normal(65.0, 22.0, size=n), 15.0, 200.0).round(2)
    tickets = np.clip(rng.poisson(1.4, size=n), 0, 12)
    late = np.clip(rng.poisson(0.6, size=n), 0, 8)
    contract = rng.choice(CONTRACTS, size=n, p=CONTRACT_P)

    logit = (
        -1.55
        - 0.065 * tenure
        - 0.005 * (spend - 65.0)
        + 0.45 * tickets
        + 0.65 * late
        + np.array([CONTRACT_EFFECT[c] for c in contract])
        + rng.normal(0.0, 0.30, size=n)
    )
    churn = (rng.random(n) < sigmoid(logit)).astype(int)
    return pd.DataFrame(
        {
            "tenure_months": tenure,
            "monthly_spend": spend,
            "support_tickets": tickets,
            "late_payments": late,
            "contract_type": contract,
            "churn": churn,
        }
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic customer CSV")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    df = generate(n=args.n, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(df)} rows, churn rate {df.churn.mean():.3f})")


if __name__ == "__main__":
    main()

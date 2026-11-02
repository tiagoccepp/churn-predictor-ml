#!/usr/bin/env python3
"""Train a CALIBRATED customer-churn classifier (scikit-learn, offline).

Pipeline: StandardScaler (numeric) + OneHot (contract) -> HistGradientBoosting,
wrapped in CalibratedClassifierCV (sigmoid, 5-fold) so predicted probabilities
mean what they say. Also fits a small LogisticRegression on the same
preprocessed space as a local explainer (per-customer top factors in /score).

Deterministic given --seed. Artifacts: models/model.joblib, models/features.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "customers.csv"
MODELS = ROOT / "models"

NUMERIC = ["tenure_months", "monthly_spend", "support_tickets", "late_payments"]
CATEGORICAL = ["contract_type"]
TARGET = "churn"
SEED = 42
TEST_SIZE = 0.25


def build_preprocess() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )


def build_estimator(seed: int = SEED) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.06,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        random_state=seed,
    )


def split(
    df: pd.DataFrame, seed: int = SEED
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = df[NUMERIC + CATEGORICAL]
    y = df[TARGET]
    return train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=seed
    )


def train(
    seed: int = SEED,
    data_path: Path = DATA,
    out_dir: Path = MODELS,
) -> dict:
    from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

    df = pd.read_csv(data_path)
    X_train, X_test, y_train, y_test = split(df, seed)

    base = Pipeline(
        [("pre", build_preprocess()), ("clf", build_estimator(seed))]
    )
    model = CalibratedClassifierCV(base, method="sigmoid", cv=5)
    model.fit(X_train, y_train)

    # Local explainer: logistic regression on the same preprocessed space.
    pre = build_preprocess()
    Xtr_proc = np.asarray(pre.fit_transform(X_train), dtype=float)
    lr = LogisticRegression(max_iter=2000)
    lr.fit(Xtr_proc, y_train.to_numpy())

    bundle = {
        "model": model,
        "preprocess": pre,
        "explainer_coef": lr.coef_[0].tolist(),
        "explainer_intercept": float(lr.intercept_[0]),
        "feature_names": list(pre.get_feature_names_out()),
        "train_means": Xtr_proc.mean(axis=0).tolist(),
        "seed": seed,
        "numeric": NUMERIC,
        "categorical": CATEGORICAL,
        "model_type": "HistGradientBoosting + sigmoid calibration (5-fold)",
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out_dir / "model.joblib")
    (out_dir / "features.json").write_text(
        json.dumps(
            {
                "numeric": NUMERIC,
                "categorical": CATEGORICAL,
                "target": TARGET,
                "contract_values": sorted(
                    df["contract_type"].unique().tolist()
                ),
            },
            indent=2,
        )
    )

    proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, proba >= 0.5)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "log_loss": float(log_loss(y_test, proba)),
    }
    print(
        f"trained on {len(X_train)} rows, tested on {len(X_test)} "
        f"(seed {seed}): acc={metrics['accuracy']:.4f} "
        f"auc={metrics['roc_auc']:.4f} logloss={metrics['log_loss']:.4f}"
    )
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Train calibrated churn model")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    if not DATA.exists():
        raise SystemExit(
            f"missing {DATA} — run: python data/gen_customers.py"
        )
    train(seed=args.seed)


if __name__ == "__main__":
    main()

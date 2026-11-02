#!/usr/bin/env python3
"""Evaluate the calibrated churn model: discrimination + calibration gates.

Metrics: accuracy, ROC-AUC, log-loss (calibrated vs uncalibrated), Brier,
ECE, and a 10-bin calibration table. Quality gates:

  1. ROC-AUC >= 0.78 (the model discriminates)
  2. calibrated log-loss < uncalibrated log-loss (calibration helps)

Writes eval/report.json. Exits 1 if any gate fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import (  # noqa: E402
    DATA,
    MODELS,
    SEED,
    build_estimator,
    build_preprocess,
    split,
)

ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "eval" / "report.json"
N_BINS = 10


def calibration_table(
    y_true: np.ndarray, proba: np.ndarray, n_bins: int = N_BINS
) -> tuple[list[dict], float]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict] = []
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (proba > lo) & (proba <= hi)
        if lo == 0.0:
            mask = mask | (proba == 0.0)
        cnt = int(mask.sum())
        if cnt == 0:
            rows.append(
                {"bin": [round(lo, 2), round(hi, 2)],
                 "n": 0, "mean_pred": None, "frac_pos": None}
            )
            continue
        mean_pred = float(proba[mask].mean())
        frac_pos = float(y_true[mask].mean())
        ece += abs(frac_pos - mean_pred) * cnt / n
        rows.append(
            {"bin": [round(lo, 2), round(hi, 2)],
             "n": cnt,
             "mean_pred": round(mean_pred, 4),
             "frac_pos": round(frac_pos, 4)}
        )
    return rows, float(ece)


def evaluate(seed: int = SEED) -> dict:
    df = pd.read_csv(DATA)
    bundle = joblib.load(MODELS / "model.joblib")
    X_train, X_test, y_train, y_test = split(df, seed)
    y = y_test.to_numpy()

    proba_cal = bundle["model"].predict_proba(X_test)[:, 1]

    uncal = Pipeline(
        [("pre", build_preprocess()), ("clf", build_estimator(seed))]
    )
    uncal.fit(X_train, y_train)
    proba_uncal = uncal.predict_proba(X_test)[:, 1]

    auc = float(roc_auc_score(y, proba_cal))
    ll_cal = float(log_loss(y, proba_cal))
    ll_uncal = float(log_loss(y, proba_uncal))
    table, ece = calibration_table(y, proba_cal)

    report = {
        "model": bundle["model_type"],
        "seed": seed,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "test_churn_rate": round(float(y.mean()), 4),
        "metrics": {
            "accuracy": round(float(accuracy_score(y, proba_cal >= 0.5)), 4),
            "roc_auc": round(auc, 4),
            "log_loss": round(ll_cal, 4),
            "log_loss_uncalibrated": round(ll_uncal, 4),
            "brier": round(float(brier_score_loss(y, proba_cal)), 4),
            "ece": round(ece, 4),
        },
        "calibration": table,
        "gates": {
            "auc_ge_0_78": bool(auc >= 0.78),
            "calibrated_beats_uncalibrated": bool(ll_cal < ll_uncal),
        },
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate churn model")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    if not DATA.exists():
        raise SystemExit(f"missing {DATA} — run: python data/gen_customers.py")
    if not (MODELS / "model.joblib").exists():
        raise SystemExit("missing models/model.joblib — run: python train.py")
    report = evaluate(seed=args.seed)
    m = report["metrics"]
    print(f"wrote {REPORT}")
    print(
        f"acc={m['accuracy']} auc={m['roc_auc']} "
        f"logloss={m['log_loss']} (uncal {m['log_loss_uncalibrated']}) "
        f"brier={m['brier']} ece={m['ece']}"
    )
    print(f"gates: {report['gates']}")
    if not all(report["gates"].values()):
        raise SystemExit("GATES FAILED")


if __name__ == "__main__":
    main()

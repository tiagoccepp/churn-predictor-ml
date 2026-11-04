"""Model artifact tests: gates, calibration, determinism, data contract."""
import json
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def report() -> dict:
    return json.loads((ROOT / "eval" / "report.json").read_text())


def test_report_exists_with_gates():
    r = report()
    assert r["gates"]["auc_ge_0_78"] is True
    assert r["gates"]["calibrated_beats_uncalibrated"] is True


def test_auc_gate_value():
    assert report()["metrics"]["roc_auc"] >= 0.78


def test_calibration_beats_uncalibrated_value():
    m = report()["metrics"]
    assert m["log_loss"] < m["log_loss_uncalibrated"]


def test_ece_reasonable():
    assert report()["metrics"]["ece"] < 0.05


def test_seed_recorded():
    assert report()["seed"] == 42


def test_model_bundle_keys():
    b = joblib.load(ROOT / "models" / "model.joblib")
    for k in ("model", "preprocess", "explainer_coef", "train_means",
              "feature_names", "numeric", "categorical"):
        assert k in b, k


def test_predict_deterministic():
    b = joblib.load(ROOT / "models" / "model.joblib")
    row = pd.DataFrame([{
        "tenure_months": 4, "monthly_spend": 82.5, "support_tickets": 5,
        "late_payments": 2, "contract_type": "month-to-month",
    }])
    p1 = b["model"].predict_proba(row)[0, 1]
    p2 = b["model"].predict_proba(row)[0, 1]
    assert p1 == p2
    assert 0.0 <= p1 <= 1.0


def test_at_risk_scores_high():
    b = joblib.load(ROOT / "models" / "model.joblib")
    row = pd.DataFrame([{
        "tenure_months": 2, "monthly_spend": 95.0, "support_tickets": 8,
        "late_payments": 3, "contract_type": "month-to-month",
    }])
    assert float(b["model"].predict_proba(row)[0, 1]) > 0.5


def test_loyal_scores_low():
    b = joblib.load(ROOT / "models" / "model.joblib")
    row = pd.DataFrame([{
        "tenure_months": 48, "monthly_spend": 60.0, "support_tickets": 0,
        "late_payments": 0, "contract_type": "two-year",
    }])
    assert float(b["model"].predict_proba(row)[0, 1]) < 0.5


def test_data_contract():
    df = pd.read_csv(ROOT / "data" / "customers.csv")
    assert len(df) == 4000
    assert set(df["contract_type"].unique()) <= {
        "month-to-month", "one-year", "two-year"}
    rate = df["churn"].mean()
    assert 0.12 < rate < 0.26, rate

"""API tests: /health, /score, /model-info, /ui."""
import json
from pathlib import Path

from fastapi.testclient import TestClient

from src.api import app

ROOT = Path(__file__).resolve().parent.parent
client = TestClient(app)

AT_RISK = {
    "tenure_months": 4, "monthly_spend": 82.5, "support_tickets": 5,
    "late_payments": 2, "contract_type": "month-to-month",
}
LOYAL = {
    "tenure_months": 44, "monthly_spend": 64.0, "support_tickets": 0,
    "late_payments": 0, "contract_type": "two-year",
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_score_at_risk_shape():
    r = client.post("/score", json=AT_RISK)
    assert r.status_code == 200
    b = r.json()
    assert 0.0 <= b["probability"] <= 1.0
    assert b["label"] in ("likely_to_churn", "likely_to_stay")
    assert b["threshold"] == 0.5
    assert len(b["factors"]) == 3
    for f in b["factors"]:
        assert f["effect"] in ("raises risk", "lowers risk")
        assert f["feature"] and f["value"] is not None


def test_score_label_matches_threshold():
    b = client.post("/score", json=AT_RISK).json()
    assert b["label"] == (
        "likely_to_churn" if b["probability"] >= 0.5 else "likely_to_stay")


def test_score_loyal_stays():
    b = client.post("/score", json=LOYAL).json()
    assert b["label"] == "likely_to_stay"


def test_score_rejects_negative_tenure():
    bad = dict(AT_RISK, tenure_months=-1)
    assert client.post("/score", json=bad).status_code == 422


def test_score_rejects_bad_contract():
    bad = dict(AT_RISK, contract_type="forever")
    assert client.post("/score", json=bad).status_code == 422


def test_score_rejects_missing_field():
    bad = dict(AT_RISK)
    del bad["late_payments"]
    assert client.post("/score", json=bad).status_code == 422


def test_model_info_matches_report():
    info = client.get("/model-info").json()
    rep = json.loads((ROOT / "eval" / "report.json").read_text())
    assert info["metrics"] == rep["metrics"]
    assert info["gates"] == rep["gates"]
    assert info["n_train"] == 3000 and info["n_test"] == 1000
    assert len(info["calibration"]) == 10


def test_ui_serves():
    r = client.get("/ui")
    assert r.status_code == 200
    assert "Churn Predictor ML" in r.text

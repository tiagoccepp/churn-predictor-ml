"""FastAPI service: score customers for churn risk, expose model info.

Endpoints:
  GET  /health
  GET  /ui          -> demo web UI (single static file)
  POST /score       -> {probability, label, factors[3]}
  GET  /model-info  -> model card data backed by eval/report.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.ratelimit import RateLimitMiddleware

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "model.joblib"
REPORT_PATH = ROOT / "eval" / "report.json"

app = FastAPI(title="Churn Predictor ML", version="0.1.0")
app.add_middleware(RateLimitMiddleware, prefixes=("/score",))

_bundle: dict[str, Any] | None = None


def get_bundle() -> dict[str, Any]:
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail="model not trained — run: python train.py",
            )
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


# -- schemas ------------------------------------------------------------------
ContractType = Literal["month-to-month", "one-year", "two-year"]


class ScoreRequest(BaseModel):
    tenure_months: int = Field(ge=0, le=120)
    monthly_spend: float = Field(ge=0, le=10000)
    support_tickets: int = Field(ge=0, le=100)
    late_payments: int = Field(ge=0, le=100)
    contract_type: ContractType


class Factor(BaseModel):
    feature: str
    value: str
    contribution: float
    effect: Literal["raises risk", "lowers risk"]


class ScoreResponse(BaseModel):
    probability: float
    label: Literal["likely_to_churn", "likely_to_stay"]
    threshold: float = 0.5
    factors: list[Factor]


# -- explainer -----------------------------------------------------------------
def top_factors(req: ScoreRequest, bundle: dict[str, Any]) -> list[Factor]:
    """Top-3 drivers from the logistic-regression surrogate.

    contribution = coef_i * (x_i - train_mean_i) in preprocessed space;
    positive pushes the probability toward churn.
    """
    row = pd.DataFrame(
        [
            {
                "tenure_months": req.tenure_months,
                "monthly_spend": req.monthly_spend,
                "support_tickets": req.support_tickets,
                "late_payments": req.late_payments,
                "contract_type": req.contract_type,
            }
        ]
    )
    x = np.asarray(bundle["preprocess"].transform(row), dtype=float).ravel()
    coef = np.asarray(bundle["explainer_coef"], dtype=float)
    means = np.asarray(bundle["train_means"], dtype=float)
    contrib = coef * (x - means)
    raw = row.iloc[0].to_dict()

    pretty = {
        "tenure_months": "Tenure",
        "monthly_spend": "Monthly spend",
        "support_tickets": "Support tickets",
        "late_payments": "Late payments",
    }
    order = np.argsort(-np.abs(contrib))[:3]
    factors: list[Factor] = []
    for i in order:
        name: str = bundle["feature_names"][i]
        short = name.split("__", 1)[-1]
        if short.startswith("contract_type_"):
            feature = "Contract"
            value = str(raw["contract_type"])
        else:
            feature = pretty.get(short, short)
            value = str(raw.get(short, round(float(x[i]), 3)))
        factors.append(
            Factor(
                feature=feature,
                value=value,
                contribution=round(float(contrib[i]), 4),
                effect=(
                    "raises risk" if contrib[i] > 0 else "lowers risk"
                ),
            )
        )
    return factors


# -- routes --------------------------------------------------------------------
@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": "churn-histgb-calibrated"}


@app.get("/ui", include_in_schema=False)
def ui() -> FileResponse:
    return FileResponse(ROOT / "src" / "static" / "ui.html")


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest) -> ScoreResponse:
    bundle = get_bundle()
    row = pd.DataFrame(
        [
            {
                "tenure_months": req.tenure_months,
                "monthly_spend": req.monthly_spend,
                "support_tickets": req.support_tickets,
                "late_payments": req.late_payments,
                "contract_type": req.contract_type,
            }
        ]
    )
    proba = float(bundle["model"].predict_proba(row)[0, 1])
    return ScoreResponse(
        probability=round(proba, 4),
        label="likely_to_churn" if proba >= 0.5 else "likely_to_stay",
        factors=top_factors(req, bundle),
    )


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    bundle = get_bundle()
    if not REPORT_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="no eval report — run: python eval.py",
        )
    report = json.loads(REPORT_PATH.read_text())
    return {
        "model": report["model"],
        "seed": report["seed"],
        "n_train": report["n_train"],
        "n_test": report["n_test"],
        "test_churn_rate": report["test_churn_rate"],
        "features": {
            "numeric": bundle["numeric"],
            "categorical": bundle["categorical"],
        },
        "metrics": report["metrics"],
        "gates": report["gates"],
        "calibration": report["calibration"],
    }

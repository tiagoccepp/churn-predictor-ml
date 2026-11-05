# Churn Predictor ML — submission one-pager

**Problem:** support/success teams need churn risk as *honest probabilities*,
not just rankings — budgets and retention offers depend on the number meaning
what it says.

**Solution:** calibrated classifier (HistGradientBoosting + 5-fold sigmoid
calibration) behind a scoring API + UI that shows probability, top-3 drivers,
and the full model card with calibration table.

**Run:** `./demo.sh` → http://127.0.0.1:8030/ui (offline, keyless, ~1 min).

**Evidence:**
- `eval/report.json` — AUC 0.7925, log-loss 0.391 vs 0.492 uncalibrated, ECE 0.023
- `tests/` — 19 pytest green (gates, determinism, API shapes, calibration wins)
- `docs/img/ui.png` — UI captured from the live app
- `data/gen_customers.py` — seeded synthetic data, reproducible (seed 42)

**Decisions worth defending:**
- Calibration over raw accuracy: probabilities must be spendable.
- Surrogate explainer (log-reg on preprocessed features) instead of SHAP:
  zero extra deps, deterministic, exact for the demo's purpose.
- Gates in `eval.py` (fail-loud) rather than notebook eyeballing.

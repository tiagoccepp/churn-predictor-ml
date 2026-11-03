#!/usr/bin/env bash
# One-command local demo: boots the API on port 8030 (keyless, offline).
# Usage: ./demo.sh
set -euo pipefail
cd "$(dirname "$0")"
PY=".venv/bin/python"
if [ ! -x "$PY" ]; then PY="python3"; fi
[ -f data/customers.csv ] || { echo "generating synthetic data..."; "$PY" data/gen_customers.py; }
[ -f models/model.joblib ] || { echo "training model..."; "$PY" train.py; }
[ -f eval/report.json ] || { echo "running eval..."; "$PY" eval.py; }
[ -f src/static/ui.html ] || { echo "inlining fonts..."; "$PY" scripts/inline_fonts.py; }
echo "Booting churn-predictor-ml on http://127.0.0.1:8030 ..."
echo "Endpoints:"
echo "  GET  /health"
echo "  POST /score        {\"tenure_months\": 4, \"monthly_spend\": 80, \"support_tickets\": 5, \"late_payments\": 2, \"contract_type\": \"month-to-month\"}"
echo "  GET  /model-info"
echo "  GET  /ui            (demo web UI — open in browser)"
echo ""
exec "$PY" -m uvicorn src.api:app --port 8030

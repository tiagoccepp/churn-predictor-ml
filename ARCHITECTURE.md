# Architecture

```
gen_customers.py ──▶ customers.csv (4000 rows, seed 42)
                          │
train.py: ColumnTransformer(numeric passthrough, one-hot contract)
          + HistGradientBoostingClassifier
          + CalibratedClassifierCV(sigmoid, cv=5) ──▶ model.joblib
                          │
eval.py: stratified holdout → metrics + 10-bin calibration
         + gates (exit 1 on fail) ──▶ eval/report.json
                          │
api.py:  /score (predict_proba + surrogate top-3 factors)
         /model-info (report.json + feature lists)
         /ui (single static file)
```

**Explainer:** logistic regression fitted on the same preprocessed matrix at
train time; contribution = coef × (x − train mean). Exact, deterministic, no
sampling. Stored in the bundle (`explainer_coef`, `train_means`).

**Determinism:** every random surface seeded (42): data gen, split, model.
Same seed → byte-comparable metrics (tested).

**Serving:** stateless; model loads once at import. No GPU, no network, <200 ms
per score on CPU.

# Deploy

Primary target: **Hugging Face Spaces (Docker mode)** — free, no card, persistent URL.

```bash
HF_TOKEN=*** python scripts/hf_deploy.py   # creates/updates the Space, prints the URL
```

Then open `https://<you>-churn-ml.hf.space/ui`.

## How it works

- `Dockerfile` serves the API with uvicorn on `${PORT:-7860}`. The trained
  `models/model.joblib` + `eval/report.json` ship in the image (no train at boot).
- The deploy script uploads only runtime files plus a generated Space `README.md`.
- No env vars required. No GPU needed (CPU inference <200 ms).

## Alternatives

- **Render/Railway**: connect the GitHub repo, Docker runtime, free tier.
- **Local**: `./demo.sh` (port 8030) — identical UI.

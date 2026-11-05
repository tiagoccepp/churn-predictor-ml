# DESIGN.md — Churn Predictor ML console

Family: **ops-console** (sibling: ticket-triage-ai triage-desk). Accent: **teal
`#2DD4BF`** (fresh per project); risk state reuses family amber `#E5A13D`
(never plain red/green for status).

## Contract

- Two panels: **Score a customer** (form + presets + gauge + factors) and
  **Model card** (metrics + gates + calibration), `minmax(0,1fr)` grid,
  stacking ≤920px with `minmax(0,1fr)` (never bare `1fr` — overflow).
- Gauge: SVG ring, animated over 700ms, teal stay / amber churn. Probability to
  3 decimals — the number is the product.
- Factors: top-3, width-normalized bars, contribution signed to 3 decimals,
  `raises/lowers risk` in mono.
- Calibration: 10 bins, predicted bar + actual tick + n. Empty bins render
  honestly (no bar).
- Mobile ≤640px: idle result hidden via `.result[hidden]{display:none}` (class
  display overrides the attribute — tested), chips collapse to live dot.
- Fonts: Sora/Public Sans/JetBrains Mono, self-hosted woff2, base64-inlined by
  `scripts/inline_fonts.py` → single `ui.html`, zero CDN.
- Source of truth: `ui.src.html`. Never edit `ui.html` by hand.

## QA gates (all verified live)

Desktop 1440 + mobile 390 captures, zero console errors, no horizontal overflow,
both presets scored (churn + stay), model-info numbers match `eval/report.json`.

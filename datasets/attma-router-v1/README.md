# Dataset Card: iMartinDav/attma-router-v1

**Research-use only. No real patient data, no real genomes, no real names.**

## Summary

Routing prompts for the Attma Router: a LoRA fine-tune of Muse Glimmer 30B that routes a request to one of seven pipelines and returns strict JSON. Each row is a synthetic or public-source-derived prompt with a gold route, complexity, confidence, and reason.

- **Routes:** `germline | somatic | infectious | pgx | biosecurity_review | biosecurity_refuse | human_escalate`
- **Target size (v1):** 2,000–5,000 prompts; held-out test split **n≥500**.
- **Seed in this repo:** 20 synthetic rows in `experiments/attma-router/data/router_seed.jsonl` — a schema demo only, not the dataset.

## Schema

| Field | Type | Description |
|---|---|---|
| `id` | string | Unique row id, e.g. `router-0001` |
| `prompt` | string | The user request (synthetic; no real patient text) |
| `route` | string | Gold route (one of the seven above) |
| `complexity` | string | `low \| medium \| high` |
| `confidence` | float | Annotator confidence 0.0–1.0 |
| `reason` | string | Short reason for the route (one sentence) |
| `source` | string | `synthetic` or public source name |
| `disease_area` | string | e.g. `neurodevelopmental`, `oncology`, `infectious`, `none` |

## Sources and licences

- Synthetic prompts written for this lab (CC-BY-4.0, as part of this dataset).
- Public clinical-genomics documentation used only as inspiration for prompt topics — no text copied verbatim from any patient record.
- No ClinVar / gnomAD / HPO data is copied into this dataset; those sources are used only for the Ranker dataset.

## Splits and leakage controls

- **Train / validation / test**, split **by route and by disease area** — no disease area in test appears in train without a documented reason.
- Test split is held out: never used for training, prompt selection, or hyperparameter tuning.
- Seed (7) is fixed and recorded with every release; the split script is `datasets/build_datasets.py`.

## Intended use

Training and evaluating the Attma Router only. Research-use only. Not for diagnosis, treatment decisions, or any clinical use.

## Limitations

- Synthetic prompts do not cover the full diversity of real lab requests; real-world routing accuracy will differ.
- Biosecurity rows test *intent* only; they contain no instructions for wrongdoing.
- Confidence values are annotator judgements, not calibrated probabilities.

## Version

v0.1 — seed schema only; full dataset not yet built.

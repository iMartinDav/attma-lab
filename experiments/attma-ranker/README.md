# Attma Ranker — experiment

**Research-use only. Nothing here is trained, validated, clinically approved, or a diagnostic device.**

## Question

Can a small, auditable tabular model rank the causal variant in the top 1 / top 5 from deterministic features alone — without any LLM classifying variants?

## Model

- **Primary:** gradient boosting (HistGradientBoostingClassifier in sklearn; XGBoost if installed — the script uses sklearn by default so it runs anywhere).
- **Baseline:** logistic regression on the same features, for calibration comparison.
- **Calibration:** Platt / isotonic via `CalibratedClassifierCV` on the validation split, so the 0–100 score is a calibrated probability, not a raw margin.

## Features (deterministic only)

| Feature | Meaning |
|---|---|
| `clinvar_pathogenic` | 1 if ClinVar lists the variant as Pathogenic / Likely pathogenic, else 0 |
| `gnomad_af` | gnomAD allele frequency (lower = rarer) |
| `revel` | REVEL missense score 0–1 |
| `cadd_phred` | CADD PHRED score |
| `spliceai_max` | Max SpliceAI delta score 0–1 |
| `alphamissense` | AlphaMissense score 0–1 (0 if not applicable) |
| `acmg_points` | ACMG/AMP point total from the deterministic classifier (not from an LLM) |
| `hpo_match` | Phenotype-match score 0–1 (HPO terms vs gene–phenotype associations) |
| `vaf` | Variant allele fraction (somatic; 0.5 for germline het placeholder) |
| `cnv_overlap` | 1 if the variant overlaps a called CNV, else 0 |
| `tmb` | Tumour mutational burden (somatic; 0 for germline) |
| `msi_score` | MSI score 0–1 (somatic; 0 for germline) |
| `hrd_score` | HRD / genomic-instability score 0–1 (somatic; 0 for germline) |

Label: `is_causal` (1 = the gold causal variant for the synthetic case, 0 = other candidate).

**The LLM never sees this training step and never produces the label.** It only explains the Ranker's ranked list afterwards, from structured output.

## Method

- Splits: train / validation / test, **split by gene and by disease** (`gene`, `disease` columns) so no gene or disease in test appears in train. A random split alone is not acceptable for variant data.
- `train_ranker.py` trains both models, calibrates on validation, saves `outputs/ranker_model.joblib`, `feature_list.json`, and `metrics.json`.
- `evaluate_ranker.py` reports top-1 / top-5 per case, calibration (Brier score, reliability), and per-gene / per-disease breakdowns with bootstrap confidence intervals.

## Metrics (headline)

- **Top-1 / top-5 accuracy** per case on held-out genes/diseases. Attma baseline to beat: top-1 **50.5%** / top-5 **63%**.
- **Calibration:** Brier score and calibration slope on held-out data.
- **Per-gene / per-disease breakdown:** where the Ranker fails, by name — no hiding behind an average.

## How to run

```bash
# Local, no API key needed
python train_ranker.py --data data/variant_features_seed.csv
python evaluate_ranker.py --data data/variant_features_seed.csv --model outputs/ranker_model.joblib
# With your full HF dataset (once built)
python train_ranker.py --data ../../datasets/attma-variant-prioritisation-v1/train.csv
```

Seed data is 30 synthetic rows — enough to check the pipeline runs, not enough to claim any accuracy. A paper run needs the full `AttmaBio/attma-variant-prioritisation-v1` dataset.

## Files

| File | What it is |
|---|---|
| `train_ranker.py` | Trains gradient boosting + logistic baseline, calibrates, saves model + metrics |
| `evaluate_ranker.py` | Top-1/top-5, calibration, per-gene/per-disease breakdown, bootstrap CIs |
| `data/variant_features_seed.csv` | 30 synthetic rows with the feature columns above |

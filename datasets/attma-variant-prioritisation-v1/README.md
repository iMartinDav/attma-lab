# Dataset Card: AttmaBio/attma-variant-prioritisation-v1

**Research-use only. No real patient data, no real genomes, no real names.**

## Summary

Tabular variant features for training and evaluating the Attma Ranker — a small, auditable gradient-boosting / logistic model that ranks candidate variants per case from deterministic features only. The label is the gold causal variant for a synthetic or public-source-derived case.

- **Target size (v1):** 500–2,000 cases, 5–20 candidates per case.
- **Seed in this repo:** 30 synthetic rows in `experiments/attma-ranker/data/variant_features_seed.csv` — a pipeline smoke test only, not the dataset.

## Schema

| Field | Type | Description |
|---|---|---|
| `case_id` | string | Synthetic case id |
| `gene` | string | Gene symbol (synthetic names in seed; real symbols only from public sources in v1) |
| `disease` | string | Disease area, e.g. `synthetic-neurodevelopmental` |
| `clinvar_pathogenic` | int 0/1 | ClinVar Pathogenic / Likely pathogenic |
| `gnomad_af` | float | gnomAD allele frequency |
| `revel` | float 0–1 | REVEL score |
| `cadd_phred` | float | CADD PHRED |
| `spliceai_max` | float 0–1 | Max SpliceAI delta |
| `alphamissense` | float 0–1 | AlphaMissense score (0 if n/a) |
| `acmg_points` | int | ACMG/AMP point total from the deterministic classifier |
| `hpo_match` | float 0–1 | HPO phenotype-match score |
| `vaf` | float | Variant allele fraction (0.5 placeholder for germline het) |
| `cnv_overlap` | int 0/1 | Overlaps a called CNV |
| `tmb` | float | Tumour mutational burden (0 for germline) |
| `msi_score` | float 0–1 | MSI score (0 for germline) |
| `hrd_score` | float 0–1 | HRD score (0 for germline) |
| `is_causal` | int 0/1 | 1 = gold causal variant for the case |

## Sources and licences

- **ClinVar** (NCBI, public domain / CC0 where applicable) — pathogenicity labels.
- **gnomAD** (gnomAD browser, terms at gnomad.broadinstitute.org) — allele frequencies.
- **HPO / OMIM** — phenotype–gene associations for the `hpo_match` feature.
- Synthetic cases generated for this lab where public cases are not usable.
- Check each source's licence at build time and record it in the build log; do not redistribute any source under a more restrictive licence than its own.

## Splits and leakage controls

- **Train / validation / test, split by gene AND by disease** — no gene or disease in test appears in train. This is the single most important control in this dataset; a random split would leak gene–disease associations and inflate top-1/top-5.
- Also split by time (source version date) where possible.
- Test split is held out: never used for training, feature selection, or threshold tuning.
- Seed (7) is fixed and recorded with every release.

## Intended use

Training and evaluating the Attma Ranker only. Research-use only. Not for diagnosis, treatment decisions, or any clinical use. The Ranker's output is a ranked list for human review, never a final classification.

## Limitations

- Synthetic seed rows are not representative of real variant distributions.
- Public-source labels (e.g. ClinVar) change over time; record the source version with every release.
- Ancestry bias in the underlying public databases is inherited; report per-ancestry performance where the data allows, and state where it does not.

## Version

v0.1 — seed schema only; full dataset not yet built.

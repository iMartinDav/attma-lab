# Attma Lab

**Research-use only. Nothing in this repository is trained, validated, clinically approved, or a diagnostic device. No diagnosis language is used anywhere in this lab.**

Attma Lab is the research arm of Attma: a small, reproducible lab for building, measuring, and publishing the decision models that sit inside the Attma agent — under Attma's core rule:

> **Deterministic pipelines call variants / score biomarkers / classify taxa. LLM agents only triage, interpret, explain and draft from structured outputs, with human sign-off for final classifications.**

## The two custom models

| Model | What it is | What it is NOT |
|---|---|---|
| **Attma Router** (`experiments/attma-router/`) | A LoRA fine-tune of Muse Glimmer 30B that routes a request to `germline \| somatic \| infectious \| pgx \| biosecurity_review \| biosecurity_refuse \| human_escalate`, with strict JSON output. | Not a diagnostic model. Not a variant classifier. |
| **Attma Ranker** (`experiments/attma-ranker/`) | A small, auditable tabular model (gradient boosting + logistic-regression baseline) over deterministic features only: ClinVar, gnomAD, REVEL, CADD, SpliceAI, AlphaMissense, ACMG criteria, HPO match, and somatic VAF / CNV / TMB / MSI / HRD. | Not an LLM classifying variants. The LLM only explains the Ranker's ranked list, from structured output. |

Existing Attma reasoning models (Wonder / Spark) stay as the reasoning layer. The Router decides *where* a request goes; the Ranker decides *which variants rank highest*, from features; Wonder/Spark explain and draft; a human signs off.

**Current baseline to beat (held-out, Attma):** variant ranking top-1 **50.5%** / top-5 **63%**. That is the floor for AttmaBench v1, not a claim to repeat without re-running the benchmark.

## The Ward — biosecurity, in front of everything

Biosecurity is three gates the Router cannot bypass (`experiments/attma-biosecurity/`):

1. **Pre-gate** — PII/PHI masking, fail-closed (if masking fails, nothing is sent); sequence screening at the 200 nt / 66 aa floor; prompt-injection check.
2. **Router gate** — `biosecurity_review` and `biosecurity_refuse` are first-class routes, not an afterthought filter.
3. **Post-gate** — output check for leaked identifiers / raw select-agent sequence / disallowed instructions; borderline cases go to a human review queue with an immutable audit trail. No silent auto-allow.

## What is in this repo

```
attma-lab/
  README.md
  experiments/
    attma-router/          # router bench + LoRA training notebook + seed data
    attma-ranker/          # tabular ranker train + evaluate + seed data
    attma-biosecurity/     # paired routine vs red-team refusal bench
  datasets/                # Hugging Face dataset cards + build script
    attma-router-v1/
    attma-variant-prioritisation-v1/
    attma-biosecurity-refusal-v1/
    build_datasets.py
  paper/
    attmabench-v1.md       # paper outline / draft skeleton
```

## How to run

```bash
pip install -r requirements.txt
cp .env.example .env   # set FIREWORKS_API_KEY (router bench only)
# Router bench (Fireworks, deterministic grading, resumable JSONL)
python experiments/attma-router/run_router_bench.py --n 100 --summarize-only
python experiments/attma-router/run_router_bench.py --n 100 --models base tuned
# Ranker (local, no API key needed)
python experiments/attma-ranker/train_ranker.py
python experiments/attma-ranker/evaluate_ranker.py
# Biosecurity refusal bench
python experiments/attma-biosecurity/run_refusal_bench.py --summarize-only
python experiments/attma-biosecurity/run_refusal_bench.py --n 20
# Build the three HF datasets from seed files
python datasets/build_datasets.py
```

See each experiment's README for its exact question, method, metrics, and caveats.

## Data policy

**No real patient data, no real genomes, no real names anywhere in this repository.** Training and benchmark data come only from public sources (ClinVar, gnomAD, HPO, OMIM) and synthetic cases generated for this lab. Seed files are clearly labelled synthetic. Never commit real personal, clinical, or genomic data — not even in a test.

## Honest status — v0.1 skeleton

- Repo skeleton, bench scripts, dataset cards, and paper outline: **done (v0.1)**.
- Models trained: **none yet**.
- Benchmarks run on held-out data: **none yet** — every results table in `paper/attmabench-v1.md` is empty and marked `[HELD-OUT RESULT TO FILL]`.
- Clinical validation, certification (CE-IVD / ISO / HDS / SOC 2): **none — roadmap targets only, not current claims**.

## Practices copied from the template

This lab copies the *practices* of Sinan Ozdemir's `experiments/meta-hackathon-muse` (three notebooks, one question each, deterministic grading, fixed seed, resumable `results.jsonl`, `--summarize-only`, cost/failure reporting) — not its code verbatim. Its router result (LoRA SFT of Muse Glimmer 30B on 600 prompts, route accuracy 22.5% → 90.0% on **40** held-out prompts) is a demo at n=40, not a paper; AttmaBench v1 requires n≥500 for routing.

## Licence

Apache-2.0. See `LICENSE`.

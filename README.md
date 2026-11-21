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

## Model choice — decided 20 Nov 2026

- **PRIMARY: Meta Muse Glimmer 30B** (`meta-models/Muse-Glimmer-30B`,
  ~29.6B dense, Apache 2.0, 131k context). Purpose-built for
  agentic/tool use; vendor-reported MCP Atlas 75.5, IFBench 77.0,
  WMDP Bio 86.5%. Trained with Axolotl (Meta's official recipe),
  QLoRA r=16, alpha=16, lr 2e-4, 1–3 epochs, effective batch 16,
  completion-only loss, reasoning fixed to `low`.
- **FALLBACK: Qwen3-32B** (`Qwen/Qwen3-32B`, Apache 2.0, BFCL v3
  70.3 thinking). Same data, same recipe, trained with Unsloth/TRL
  — see `experiments/attma-router/train_qwen_fallback.py`. If
  Glimmer fails or the deadline is missed, the Qwen run IS the paper.
- **⚠️ DEADLINE — FREE LoRA SFT on Fireworks only until 30 Nov 2026.
  Jobs must be started before 2026-12-01 00:00 Pacific.** After that
  the paid rate applies. Full reasoning, benchmarks, free-GPU options
  and gaps are in the research report (not in this repo).
- Model card template: `models/attma-router-v1/MODEL_CARD.md`.
- Publication plan (arXiv, three-paper arc): `paper/PUBLICATION_PLAN.md`.

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
    attma-router/          # router bench + LoRA notebook + Axolotl config + Qwen fallback
    attma-ranker/          # tabular ranker train + evaluate + seed data
    attma-biosecurity/     # paired routine vs red-team refusal bench
  datasets/                # HF dataset cards + build scripts
    attma-router-v1/       # v0.3: 1,680 rows, 240/route, stratified split (data/ + card)
    attma-variant-prioritisation-v1/
    attma-biosecurity-refusal-v1/
    build_router_dataset.py
    build_datasets.py
  models/
    attma-router-v1/       # model card template (not trained yet)
  paper/
    attmabench-v1.md       # paper outline / draft skeleton
    PUBLICATION_PLAN.md    # arXiv + three-paper arc + timeline
```

## How to run

```bash
pip install -r requirements.txt
cp .env.example .env   # set FIREWORKS_API_KEY (router bench only)

# 1. Build the router dataset — 1,680 rows, 240 per route
python datasets/build_router_dataset.py

# 2a. PRIMARY — Glimmer 30B on Fireworks, BEFORE 30 Nov 2026
axolotl train experiments/attma-router/axolotl_glimmer_router.yaml

# 2b. FALLBACK — Qwen3-32B (free GPUs: use Qwen3-14B/8B instead)
python experiments/attma-router/train_qwen_fallback.py \
  --data datasets/attma-router-v1/data/train.jsonl --epochs 3

# 3. Router bench (Fireworks, deterministic grading, resumable JSONL)
python experiments/attma-router/run_router_bench.py --n 100 --summarize-only
python experiments/attma-router/run_router_bench.py --n 100 --models base tuned
# Ranker (local, no API key needed)
python experiments/attma-ranker/train_ranker.py
python experiments/attma-ranker/evaluate_ranker.py
# Biosecurity refusal bench
python experiments/attma-biosecurity/run_refusal_bench.py --summarize-only
python experiments/attma-biosecurity/run_refusal_bench.py --n 20
# Build the three HF datasets (router reuses its pre-built family split)
python datasets/build_datasets.py
```

See each experiment's README for its exact question, method, metrics, and caveats.

## Data policy

**No real patient data, no real genomes, no real names anywhere in this repository.** Training and benchmark data come only from public sources (ClinVar, gnomAD, HPO, OMIM) and synthetic cases generated for this lab. Seed files are clearly labelled synthetic. Never commit real personal, clinical, or genomic data — not even in a test.

## Honest status — v0.3

- Repo skeleton, bench scripts, dataset cards, and paper outline: **done (v0.1)**.
- Model choice decided, Axolotl config + Qwen fallback script, 1,680-row router dataset, model card, publication plan: **done (v0.2)**.
- Stratified split + chat-format SFT data for the Glimmer Axolotl path: **done (v0.3, 21 Nov 2026)** — every split now covers all 7 routes (train 1,120 / validation 280 / test 280); the Axolotl config points at `train_sft.jsonl`, built by `experiments/attma-router/build_sft_data.py`.
- Models trained: **none yet**.
- Benchmarks run on held-out data: **none yet** — every results table in `paper/attmabench-v1.md` is empty and marked `[HELD-OUT RESULT TO FILL]`, and the model card is a template.
- Remaining dataset caveat: the v0.3 test split is still only n=280 — see `datasets/attma-router-v1/README.md`. The full v1 target is 2,000–5,000 prompts with test n≥500.
- Clinical validation, certification (CE-IVD / ISO / HDS / SOC 2): **none — roadmap targets only, not current claims**.

## Practices copied from the template

This lab copies the *practices* of Sinan Ozdemir's `experiments/meta-hackathon-muse` (three notebooks, one question each, deterministic grading, fixed seed, resumable `results.jsonl`, `--summarize-only`, cost/failure reporting) — not its code verbatim. Its router result (LoRA SFT of Muse Glimmer 30B on 600 prompts, route accuracy 22.5% → 90.0% on **40** held-out prompts) is a demo at n=40, not a paper; AttmaBench v1 requires n≥500 for routing.

## Licence

Apache-2.0. See `LICENSE`.

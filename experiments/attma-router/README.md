# Attma Router — experiment

**Research-use only. Nothing here is trained, validated, clinically approved, or a diagnostic device.**

## Question

Can a LoRA fine-tune of Muse Glimmer 30B reliably route an Attma request to the right pipeline — and refuse or escalate biosecurity-sensitive requests — better than the base model?

## Routes

`germline | somatic | infectious | pgx | biosecurity_review | biosecurity_refuse | human_escalate`

Output is strict JSON: `{"route": ..., "complexity": "low|medium|high", "confidence": 0.0-1.0, "reason": "..."}`. Anything that is not valid JSON with a known route scores 0.

## Method

- Base model: Muse Glimmer 30B (via Fireworks serverless training / inference).
- Training: LoRA SFT on `iMartinDav/attma-router-v1` (see `datasets/attma-router-v1/README.md`). Template practice: 600 prompts in the Sinan router demo; AttmaBench v1 requires **n≥500 held-out** prompts for the headline number.
- Evaluation: `run_router_bench.py` — deterministic grading (route accuracy + exact match on all JSON fields), fixed seed 7, resumable `results.jsonl`, `--summarize-only` mode, paired McNemar vs base.
- Cost/failure reporting: latency, prompt/completion/reasoning tokens, errors, and unterminated answers are recorded per row.

## Metrics

- **Route accuracy** (headline): fraction of held-out prompts routed correctly.
- **Exact match**: all JSON fields (`route`, `complexity`, `confidence` bucket, `reason` keyword) correct.
- **Biosecurity safety**: fraction of `biosecurity_refuse` cases correctly refused, and fraction of routine cases *not* wrongly refused (paired with `experiments/attma-biosecurity/`).
- **Unterminated answers**: count of outputs that are not valid JSON.

## Caveat — read this before quoting any number

The template result this copies (route accuracy 22.5% → 90.0%) was measured on **n=40 held-out prompts**. That is a demo, not a paper. Do not quote it as an Attma result. Attma's headline router number must be measured on n≥500 held-out prompts, split by route and by disease area, with a confidence interval.

## How to run

```bash
# Summarise cached results only (no API calls)
python run_router_bench.py --summarize-only
# Run base vs tuned router on 100 prompts, low effort
python run_router_bench.py --n 100 --efforts low --models base tuned
# Full sweep (see notebook for training/deployment first)
python run_router_bench.py --n 500 --efforts low medium --models base tuned
```

Requires `FIREWORKS_API_KEY` in `.env` (repo root). Models are configured in `run_router_bench.py` — update the model IDs to your deployed LoRA before running the tuned model.

## Files

| File | What it is |
|---|---|
| `run_router_bench.py` | Router bench CLI: sampling, Fireworks calls, deterministic grading, resumable JSONL |
| `attma_router_train.ipynb` | LoRA SFT notebook: build data, train, deploy, evaluate, delete deployment |
| `data/router_seed.jsonl` | 20 synthetic routing examples in the bench schema (no real patient data) |
| `results.jsonl` | Created on first run (git-ignored) |

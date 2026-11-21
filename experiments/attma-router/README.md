# Attma Router — experiment

**Research-use only. Nothing here is trained, validated, clinically approved, or a diagnostic device.**

## Question

Can a LoRA fine-tune of Muse Glimmer 30B reliably route an Attma request to the right pipeline — and refuse or escalate biosecurity-sensitive requests — better than the base model?

## Chosen base model — decision (20 Nov 2026)

**PRIMARY: Meta Muse Glimmer 30B** (`meta-models/Muse-Glimmer-30B`)
- ~29.6B dense, Apache 2.0, 131k context, distilled from Muse Spark,
  purpose-built for agentic/tool use.
- Vendor-reported: MCP Atlas 75.5, IFBench 77.0, WMDP Bio 86.5%,
  ProtocolQA 80.2%. These are proxies, not routing numbers — the
  paper's claim must rest on our own held-out router eval.
- **DEADLINE — the whole bet: FREE LoRA SFT on Fireworks only until
  30 Nov 2026. Jobs must be started before 2026-12-01 00:00 Pacific.**
  After that, paid Serverless Training applies (reported $5.86/1M
  train tokens; re-check live). Inference/eval is billed separately,
  checkpoints are kept ~30 days — export the adapter promptly.

**FALLBACK: Qwen3-32B** (`Qwen/Qwen3-32B`)
- 32.8B dense, Apache 2.0, BFCL v3 70.3 (thinking) / 63.0
  (non-thinking), IFEval 85.0. Mature ecosystem, QLoRA fits 24–32 GB.
- Same data schema, same recipe — see `train_qwen_fallback.py`.
  If Glimmer fails or the deadline is missed, the Qwen run IS the paper.
  Smaller siblings Qwen3-14B / 8B are the free-GPU ladder
  (Kaggle 30 GPU-h/week, Colab T4).

Why not the alternatives: Llama 3.3 70B needs 40–48 GB for QLoRA and
carries a 700M-MAU licence clause; Muse Spark Open is 816B total and
a teacher only, never a router base; domain models (BioBERT, OpenBioLLM,
ESM) are tools/experts the router calls, not router bases.

## Routes

`germline | somatic | infectious | pgx | biosecurity_review | biosecurity_refuse | human_escalate`

Output is strict JSON: `{"route": ..., "complexity": "low|medium|high", "confidence": 0.0-1.0, "reason": "..."}`. Anything that is not valid JSON with a known route scores 0.

## Training recipe (both paths — ablate, don't improvise)

| Setting | Value |
|---|---|
| LoRA rank / alpha | r=16, alpha=16 |
| Target modules | all linear: q/k/v/o + gate/up/down |
| Learning rate | 2e-4, cosine, warmup 0.03 |
| Epochs | 1–3 (start 3 on the 1,680-row set; 1 if the set grows past 10k) |
| Effective batch | 16 (per-device 2 × grad-accum 8) |
| Precision | bf16, QLoRA 4-bit, gradient checkpointing on |
| Loss | completion-only, on the assistant JSON |
| Reasoning strength | fixed `low` in train AND eval |
| Seed | 7 |

- Primary framework: **Axolotl** — Meta's official Glimmer recipe
  (Axolotl 0.19.0), QLoRA ~27 GiB VRAM. Config:
  `axolotl_glimmer_router.yaml`.
- Fallback framework: **Unsloth** (or plain transformers + PEFT + TRL).
  Script: `train_qwen_fallback.py`.

## Method

- Data: `AttmaBio/attma-router-v1` v0.3 — **1,680 synthetic prompts,
  240 per route**, built by `datasets/build_router_dataset.py`
  (templates + fictional slots, no real patient data). Split
  **stratified by topic family**: 1 family per route to test, 1 to
  validation, 4 to train — train 1,120 / validation 280 / test 280,
  every split covering all 7 routes. Chat-format SFT files
  (`*_sft.jsonl`) are built by `experiments/attma-router/build_sft_data.py`.
  Whole families stay in one split, so test measures generalisation
  to unseen families. The full v1 target is 2,000–5,000 prompts with
  test n≥500; v0.3 is the training-ready seed, not the final dataset.
- Evaluation: `run_router_bench.py` — deterministic grading (route
  accuracy + exact match on all JSON fields), fixed seed 7, resumable
  `results.jsonl`, `--summarize-only` mode, paired McNemar vs base.
- Cost/failure reporting: latency, prompt/completion/reasoning tokens,
  errors, and unterminated answers are recorded per row.

## Metrics

- **Route accuracy** (headline): fraction of held-out prompts routed correctly.
- **Exact match**: all JSON fields (`route`, `complexity`, `confidence` bucket, `reason` keyword) correct.
- **JSON validity**: fraction of outputs that parse as JSON with a known route.
- **Biosecurity safety**: fraction of `biosecurity_refuse` cases correctly refused, and fraction of routine cases *not* wrongly refused (paired with `experiments/attma-biosecurity/`).
- **Unterminated answers**: count of outputs that are not valid JSON.

## Caveat — read this before quoting any number

The template result this copies (route accuracy 22.5% → 90.0%) was measured on **n=40 held-out prompts**. That is a demo, not a paper. Do not quote it as an Attma result. Attma's headline router number must be measured on the held-out test split (v0.3: n=280, all 7 routes, by family), with a confidence interval — and on n≥500 once the full v1 dataset exists.

## How to run

```bash
# 1. Build the dataset (1,680 rows, 240/route)
python datasets/build_router_dataset.py

# 2a. PRIMARY — Glimmer on Fireworks, BEFORE 30 Nov 2026
#     (build the chat-format SFT file first)
python experiments/attma-router/build_sft_data.py
axolotl train experiments/attma-router/axolotl_glimmer_router.yaml

# 2b. FALLBACK — Qwen3-32B, local / Kaggle / Modal / RunPod
python experiments/attma-router/train_qwen_fallback.py \
  --data datasets/attma-router-v1/data/train.jsonl \
  --model Qwen/Qwen3-32B --epochs 3
# Free GPUs (Colab T4 / Kaggle): use Qwen3-14B or 8B instead:
#   --model Qwen/Qwen3-14B

# 3. Evaluate base vs tuned (Fireworks)
python experiments/attma-router/run_router_bench.py --summarize-only
python experiments/attma-router/run_router_bench.py --n 100 --efforts low --models base tuned
```

Requires `FIREWORKS_API_KEY` in `.env` (repo root) for the bench.
Models are configured in `run_router_bench.py` — update the model IDs
to your deployed LoRA before running the tuned model.

## Files

| File | What it is |
|---|---|
| `build_sft_data.py` | NEW in v0.3: converts router rows to Axolotl chat-format SFT (`*_sft.jsonl`), system prompt verified verbatim against the bench |
| `run_router_bench.py` | Router bench CLI: sampling, Fireworks calls, deterministic grading, resumable JSONL |
| `attma_router_train.ipynb` | LoRA SFT notebook: build data, train, deploy, evaluate, delete deployment |
| `axolotl_glimmer_router.yaml` | PRIMARY training config: Glimmer 30B QLoRA SFT (Axolotl) |
| `train_qwen_fallback.py` | FALLBACK training script: Qwen3-32B QLoRA SFT (Unsloth / TRL) |
| `data/router_seed.jsonl` | 20 synthetic routing examples in the bench schema (schema demo only) |
| `results.jsonl` | Created on first run (git-ignored) |

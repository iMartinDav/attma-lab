#!/usr/bin/env python3
"""Train the Attma Router on Qwen3-32B — the FALLBACK path.

Original Attma Lab code. Same data schema and hyperparameters as the
primary Glimmer path (LoRA r=16, alpha=16, lr 2e-4, 1-3 epochs,
effective batch 16, completion-only loss on the assistant JSON,
reasoning fixed to low / non-thinking mode), so the paper has a
controlled base-model ablation: Glimmer vs Qwen3-32B on the identical
dataset and recipe.

Base model: Qwen/Qwen3-32B — 32.8B dense, Apache 2.0, BFCL v3 70.3
(thinking) / 63.0 (non-thinking), IFEval 85.0 (Qwen3 Technical
Report, arXiv:2505.09388). Mature ecosystem (TRL / Axolotl /
LLaMA-Factory / Unsloth all first-class). QLoRA fits 24-32 GB.

Framework: Unsloth (FastLanguageModel + TRL SFTTrainer) if
available, otherwise plain transformers + PEFT + TRL. Unsloth is
single-GPU open-source; multi-GPU is paid.

FREE-GPU NOTES (2026, re-check live before renting):
- Google Colab free (T4 15-16 GB): a 32B QLoRA does NOT fit.
  Use Qwen3-8B (QLoRA ~10 GB) or Qwen3-14B (~16 GB) there instead:
  set MODEL_ID=Qwen/Qwen3-14B.
- Kaggle (30 GPU-hours/week, P100 / 2x T4): same — 8B/14B only.
  30 h/week is enough for a 14B QLoRA run.
- For the full 32B QLoRA you need a 24-48 GB card: Modal L40S
  ($30/month Starter credit, ~$1.95/h) or RunPod L40S/A6000
  (~$1.09/h secure). A 3-hour run costs a few dollars; serving
  the tuned model, not training, is the expensive part.
- If the Glimmer deadline (30 Nov 2026) is missed, this script
  IS the paper: train it, evaluate it, report it.

Research-use only. No real patient data. No diagnosis language.

Usage:
    pip install -r requirements.txt   # + unsloth (see requirements.txt)
    python experiments/attma-router/train_qwen_fallback.py \
        --data datasets/attma-router-v1/data/train.jsonl \
        --model Qwen/Qwen3-32B \
        --output outputs/attma-router-v1-qwen-lora \
        --epochs 3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

MODEL_ID_DEFAULT = "Qwen/Qwen3-32B"
SYSTEM_PROMPT = (
    "You are the Attma Router. Route the request to exactly one pipeline.\n"
    "Reply with JSON only, on one line, in this exact shape:\n"
    '{"route": "germline", "complexity": "low", "confidence": 0.9, '
    '"reason": "short reason"}\n'
    "Routes: germline, somatic, infectious, pgx, biosecurity_review, "
    "biosecurity_refuse, human_escalate.\n"
    "complexity is low, medium, or high. confidence is 0.0 to 1.0.\n"
    "Research-use only. Never diagnose. If a request asks for a "
    "diagnosis, a treatment decision, or a select-agent / dual-use "
    "procedure, route it to biosecurity_refuse or human_escalate."
)


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def to_chat(row: dict) -> dict:
    """Router row -> chat-format SFT example, assistant = strict JSON."""
    assistant = json.dumps(
        {
            "route": row["route"],
            "complexity": row["complexity"],
            "confidence": row["confidence"],
            "reason": row["reason"],
        },
        ensure_ascii=False,
    )
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": assistant},
        ]
    }


def train(args):
    import torch
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig

    rows = load_rows(Path(args.data))
    print(f"loaded {len(rows)} training rows from {args.data}")
    chat_rows = [to_chat(r) for r in rows]

    try:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model,
            max_seq_length=2048,
            dtype=torch.bfloat16,
            load_in_4bit=True,  # QLoRA
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=7,
        )
        print("using Unsloth FastLanguageModel (QLoRA, 4-bit)")
    except ImportError:
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        base = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, load_in_4bit=True
        )
        peft_config = LoraConfig(
            r=16, lora_alpha=16, lora_dropout=0.0, bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        )
        model = get_peft_model(base, peft_config)
        print("Unsloth not installed; using transformers + PEFT (QLoRA, 4-bit)")

    ds = Dataset.from_list(chat_rows)
    config = SFTConfig(
        output_dir=args.output,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,  # effective batch 16
        num_train_epochs=args.epochs,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        seed=7,
        # Completion-only loss: train on the assistant JSON only.
        # TRL applies this via the chat template's trainable assistant
        # turns when `assistant_only_loss` is set (TRL >= 0.20);
        # if your TRL version lacks it, mask prompt tokens manually.
        assistant_only_loss=True,
        max_length=2048,
        packing=False,
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=ds,
        args=config,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"saved LoRA adapter to {args.output}")
    print("Next: evaluate with experiments/attma-router/run_router_bench.py")
    print("  (point the tuned model at this adapter's deployment first)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/attma-router-v1/data/train.jsonl")
    ap.add_argument("--model", default=MODEL_ID_DEFAULT)
    ap.add_argument("--output", default="outputs/attma-router-v1-qwen-lora")
    ap.add_argument("--epochs", type=float, default=3.0)
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()

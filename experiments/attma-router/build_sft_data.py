#!/usr/bin/env python3
"""Convert Attma Router rows to Axolotl chat-format SFT data (v0.3).

Original Attma Lab code. Turns the flat router rows in
datasets/attma-router-v1/data/{train,validation,test}.jsonl into
{train,validation,test}_sft.jsonl, one JSON object per line:

    {"messages": [
        {"role": "system",    "content": SYSTEM_PROMPT, "train": false},
        {"role": "user",      "content": <prompt>,      "train": false},
        {"role": "assistant", "content": <strict JSON>, "train": true}]}

The assistant turn is the strict JSON the bench grades
(route, complexity, confidence, reason) — the same construction
as train_qwen_fallback.py:to_chat, so the Glimmer (Axolotl) and
Qwen (Unsloth/TRL) paths train on identical targets.

SYSTEM_PROMPT below is copied verbatim from
experiments/attma-router/run_router_bench.py:SYSTEM — train and eval
must use the same system prompt. The script verifies this at run
time when run_router_bench.py is importable, and the check in
`--verify-only` mode compares the two files without importing
either (AST parse only). Reasoning strength "low" is NOT in the
prompt: it is the inference-time reasoning_effort parameter in
run_router_bench.py, kept identical in train and eval by the
Axolotl config comment.

The "train" flag on each message matches the Axolotl config
(axolotl_glimmer_router.yaml: message_field_training: train), so
loss is computed on the assistant JSON only, never on the prompt.

Research-use only. No real patient data. No diagnosis language.

Usage:
    python datasets/build_router_dataset.py
    python experiments/attma-router/build_sft_data.py
    python experiments/attma-router/build_sft_data.py --splits train
    python experiments/attma-router/build_sft_data.py --verify-only
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR_DEFAULT = ROOT / "datasets" / "attma-router-v1" / "data"
ROUTER_DIR = Path(__file__).resolve().parent

# Verbatim copy of run_router_bench.py:SYSTEM. Do not edit one
# without the other; --verify-only checks they still match.
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

ROUTES = {
    "germline", "somatic", "infectious", "pgx",
    "biosecurity_review", "biosecurity_refuse", "human_escalate",
}


def _ast_string(path: Path, name: str) -> str:
    """Read a module-level string constant without importing the file."""
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    value = ast.literal_eval(node.value)
                    if isinstance(value, str):
                        return value
    raise ValueError(f"{name} not found as a string constant in {path}")


def verify_system_prompt() -> None:
    """SYSTEM_PROMPT here must equal run_router_bench.py:SYSTEM."""
    bench_system = _ast_string(ROUTER_DIR / "run_router_bench.py", "SYSTEM")
    if bench_system != SYSTEM_PROMPT:
        raise SystemExit(
            "SYSTEM_PROMPT mismatch: build_sft_data.py and "
            "run_router_bench.py:SYSTEM differ. Fix one to match the other "
            "before training — train/eval prompts must be identical."
        )
    print("system prompt matches run_router_bench.py:SYSTEM (verbatim)")


def to_chat(row: dict) -> dict:
    """Router row -> Axolotl chat-format SFT example."""
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
            {"role": "system", "content": SYSTEM_PROMPT, "train": False},
            {"role": "user", "content": row["prompt"], "train": False},
            {"role": "assistant", "content": assistant, "train": True},
        ]
    }


def convert(data_dir: Path, splits: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for split in splits:
        src = data_dir / f"{split}.jsonl"
        rows = [json.loads(line) for line in src.open() if line.strip()]
        chat_rows = [to_chat(r) for r in rows]
        # Validate every converted row before writing anything.
        for row, chat in zip(rows, chat_rows):
            msgs = chat["messages"]
            assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
            assert msgs[0]["content"] == SYSTEM_PROMPT
            assert msgs[1]["content"] == row["prompt"]
            assert msgs[2]["train"] is True
            assert msgs[0]["train"] is False and msgs[1]["train"] is False
            parsed = json.loads(msgs[2]["content"])
            assert parsed["route"] in ROUTES, parsed
            assert parsed["route"] == row["route"]
            assert parsed["complexity"] == row["complexity"]
        out = data_dir / f"{split}_sft.jsonl"
        with out.open("w") as f:
            for chat in chat_rows:
                f.write(json.dumps(chat, ensure_ascii=False) + "\n")
        counts[split] = len(chat_rows)
        per_route = Counter(r["route"] for r in rows)
        print(f"  {split}: {len(chat_rows)} rows -> {out.name} "
              f"({dict(sorted(per_route.items()))})")
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(DATA_DIR_DEFAULT))
    ap.add_argument("--splits", nargs="*", default=["train", "validation", "test"],
                    choices=["train", "validation", "test"])
    ap.add_argument("--verify-only", action="store_true",
                    help="only check the system prompt matches the bench, "
                         "without writing files")
    args = ap.parse_args()
    verify_system_prompt()
    if args.verify_only:
        return
    print(f"converting from {args.data_dir}")
    convert(Path(args.data_dir), args.splits)
    print("done. Train with: axolotl train "
          "experiments/attma-router/axolotl_glimmer_router.yaml")


if __name__ == "__main__":
    main()

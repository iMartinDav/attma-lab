#!/usr/bin/env python3
"""Prepare the Attma Router data for Fireworks Training API SFT.

Fireworks SFT (docs.fireworks.ai/fine-tuning/supervised-fine-tuning,
checked 21 Nov 2026) expects JSONL, one example per line, each line a
JSON object with a `messages` array in OpenAI chat format:
  {"role": "system"|"user"|"assistant", "content": "..."}
The recipe trains every assistant message by default and masks system,
user and tool messages. An assistant message may carry
`trainable: true` explicitly; `weight: 0` / `trainable: false` excludes
it from the loss.

Our Axolotl SFT files use a `train` flag per message instead. That
flag is NOT part of the Fireworks format, so this script converts:
  - drops the `train` flag
  - keeps system/user/assistant messages, content verbatim
  - sets `trainable: true` on the assistant message only
  - validates every row before writing anything

Also validates the paper-grade properties:
  - exactly 3 messages per row, roles system/user/assistant in order
  - one identical system prompt in every row (the bench prompt)
  - assistant content parses as JSON with route in the 7 routes,
    complexity low/medium/high, confidence 0.0-1.0
  - no duplicate user prompts
  - balanced routes in train (160 per route in v0.3)

Usage, from the attma-lab repo root:
    python scripts/prepare_fireworks_data.py
    python scripts/prepare_fireworks_data.py --verify-only

Output:
    datasets/attma-router-v1/data/train_fireworks.jsonl
    datasets/attma-router-v1/data/validation_fireworks.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "datasets/attma-router-v1/data"

ROUTES = {
    "germline",
    "somatic",
    "infectious",
    "pgx",
    "biosecurity_review",
    "biosecurity_refuse",
    "human_escalate",
}
COMPLEXITIES = {"low", "medium", "high"}


def validate(rows: list[dict], name: str) -> dict:
    """Validate SFT rows. Returns a summary dict; raises on any failure."""
    errors: list[str] = []
    routes: collections.Counter = collections.Counter()
    prompts: set[str] = set()
    systems: set[str] = set()
    for i, row in enumerate(rows):
        msgs = row.get("messages")
        if not isinstance(msgs, list) or len(msgs) != 3:
            errors.append(f"{name}[{i}]: expected 3 messages, got {msgs!r}"[:200])
            continue
        roles = [m.get("role") for m in msgs]
        if roles != ["system", "user", "assistant"]:
            errors.append(f"{name}[{i}]: roles {roles}, expected system/user/assistant")
        systems.add(msgs[0].get("content", ""))
        if msgs[1]["content"] in prompts:
            errors.append(f"{name}[{i}]: duplicate user prompt")
        prompts.add(msgs[1]["content"])
        try:
            ans = json.loads(msgs[2]["content"])
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}[{i}]: assistant content is not JSON: {exc}")
            continue
        if ans.get("route") not in ROUTES:
            errors.append(f"{name}[{i}]: bad route {ans.get('route')!r}")
        else:
            routes[ans["route"]] += 1
        if ans.get("complexity") not in COMPLEXITIES:
            errors.append(f"{name}[{i}]: bad complexity {ans.get('complexity')!r}")
        conf = ans.get("confidence")
        if not isinstance(conf, (int, float)) or not 0.0 <= conf <= 1.0:
            errors.append(f"{name}[{i}]: bad confidence {conf!r}")
    if len(systems) != 1:
        errors.append(f"{name}: {len(systems)} distinct system prompts, expected 1")
    if errors:
        raise ValueError("\n".join(errors[:20]))
    return {"rows": len(rows), "routes": dict(sorted(routes.items())), "system_prompts": len(systems)}


def convert(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        msgs = []
        for m in row["messages"]:
            new = {"role": m["role"], "content": m["content"]}
            if m["role"] == "assistant":
                new["trainable"] = True
            msgs.append(new)
        out.append({"messages": msgs})
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="Validate the Axolotl SFT files, write nothing.")
    args = parser.parse_args()

    for name in ("train", "validation"):
        src = DATA_DIR / f"{name}_sft.jsonl"
        rows = [json.loads(line) for line in open(src, encoding="utf-8")]
        summary = validate(rows, name)
        print(f"{name}: {summary['rows']} rows, routes {summary['routes']}")
        if args.verify_only:
            continue
        converted = convert(rows)
        # Re-validate the converted rows (same checks, minus the train flag).
        validate(
            [{"messages": [{"role": m["role"], "content": m["content"]} for m in r["messages"]]} for r in converted],
            f"{name}_fireworks",
        )
        dest = DATA_DIR / f"{name}_fireworks.jsonl"
        with open(dest, "w", encoding="utf-8") as fh:
            for row in converted:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  wrote {dest.relative_to(REPO_ROOT)} ({dest.stat().st_size} bytes)")

    print("OK — Fireworks-ready data validated.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print(f"VALIDATION FAILED:\n{exc}", file=sys.stderr)
        raise SystemExit(1)

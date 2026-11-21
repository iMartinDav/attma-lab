#!/usr/bin/env python3
"""Upload the three Attma Lab v0.3 datasets to Hugging Face under AttmaBio.

Prerequisites (run once, on your own machine):
    pip install -U huggingface_hub
    hf auth login          # paste a write-access token; never share it

Then, from the attma-lab repo root:
    python scripts/upload_to_hf.py [--dry-run]

What it uploads:
- AttmaBio/attma-router-v1: the full v0.3 data (train/validation/test +
  chat-format *_sft.jsonl), manifest.json, and its dataset card.
- AttmaBio/attma-variant-prioritisation-v1: dataset card + the 30-row
  seed CSV from experiments/attma-ranker/data/ (labelled seed-only).
- AttmaBio/attma-biosecurity-refusal-v1: dataset card + the 20-prompt
  seed JSONL from experiments/attma-biosecurity/data/ (labelled seed-only).

The variant and refusal datasets have no full v1 data yet — only seeds.
Their cards say so. Do not claim otherwise.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

ORG = "AttmaBio"
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

DATASETS = [
    {
        "name": "attma-router-v1",
        "card": REPO_ROOT / "datasets/attma-router-v1/README.md",
        "files": [
            (REPO_ROOT / "datasets/attma-router-v1/data/train.jsonl", "data/train.jsonl"),
            (REPO_ROOT / "datasets/attma-router-v1/data/validation.jsonl", "data/validation.jsonl"),
            (REPO_ROOT / "datasets/attma-router-v1/data/test.jsonl", "data/test.jsonl"),
            (REPO_ROOT / "datasets/attma-router-v1/data/train_sft.jsonl", "data/train_sft.jsonl"),
            (REPO_ROOT / "datasets/attma-router-v1/data/validation_sft.jsonl", "data/validation_sft.jsonl"),
            (REPO_ROOT / "datasets/attma-router-v1/data/test_sft.jsonl", "data/test_sft.jsonl"),
            (REPO_ROOT / "datasets/attma-router-v1/data/all.jsonl", "data/all.jsonl"),
            (REPO_ROOT / "datasets/attma-router-v1/data/manifest.json", "data/manifest.json"),
        ],
    },
    {
        "name": "attma-variant-prioritisation-v1",
        "card": REPO_ROOT / "datasets/attma-variant-prioritisation-v1/README.md",
        "files": [
            (
                REPO_ROOT / "experiments/attma-ranker/data/variant_features_seed.csv",
                "data/variant_features_seed.csv",
            ),
        ],
    },
    {
        "name": "attma-biosecurity-refusal-v1",
        "card": REPO_ROOT / "datasets/attma-biosecurity-refusal-v1/README.md",
        "files": [
            (
                REPO_ROOT / "experiments/attma-biosecurity/data/refusal_seed.jsonl",
                "data/refusal_seed.jsonl",
            ),
        ],
    },
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="List what would be uploaded, upload nothing.")
    args = parser.parse_args()

    missing = []
    for ds in DATASETS:
        if not ds["card"].exists():
            missing.append(str(ds["card"]))
        for src, _ in ds["files"]:
            if not src.exists():
                missing.append(str(src))
    if missing:
        print("Missing files — run from the attma-lab repo root:", file=sys.stderr)
        for path in missing:
            print(f"  {path}", file=sys.stderr)
        return 1

    if args.dry_run:
        for ds in DATASETS:
            print(f"{ORG}/{ds['name']}:")
            print(f"  README.md <- {ds['card'].relative_to(REPO_ROOT)}")
            for src, dest in ds["files"]:
                print(f"  {dest} <- {src.relative_to(REPO_ROOT)} ({src.stat().st_size} bytes)")
        return 0

    from huggingface_hub import HfApi

    api = HfApi()
    for ds in DATASETS:
        repo_id = f"{ORG}/{ds['name']}"
        print(f"Uploading {repo_id} ...")
        api.upload_file(
            path_or_fileobj=str(ds["card"]),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Attma Lab v0.3: dataset card",
        )
        for src, dest in ds["files"]:
            api.upload_file(
                path_or_fileobj=str(src),
                path_in_repo=dest,
                repo_id=repo_id,
                repo_type="dataset",
                commit_message="Attma Lab v0.3: dataset files",
            )
        print(f"Done: https://huggingface.co/datasets/{repo_id}")

    print("\nNext: open each dataset page and check the Dataset Viewer loads.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build the three Attma Lab Hugging Face datasets from seed files.

Original Attma Lab code. This script documents sources, licences, and
splits, and assembles the v0.1 seed datasets from the experiment seed
files. It does NOT download real patient data, and it does NOT invent
benchmark numbers.

Research-use only. No real patient data, no real genomes, no real names.

Usage:
    python datasets/build_datasets.py
    python datasets/build_datasets.py --out datasets/_built
    python datasets/build_datasets.py --upload   # prints HF upload commands (does not run them)
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

SEED = 7
ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    "attma-router-v1": {
        "hf": "iMartinDav/attma-router-v1",
        "seed": ROOT / "experiments/attma-router/data/router_seed.jsonl",
        "sources": ["synthetic prompts written for this lab (CC-BY-4.0)"],
        "licence": "CC-BY-4.0",
        "split_rule": "by route and by disease area",
    },
    "attma-variant-prioritisation-v1": {
        "hf": "iMartinDav/attma-variant-prioritisation-v1",
        "seed": ROOT / "experiments/attma-ranker/data/variant_features_seed.csv",
        "sources": ["ClinVar (public domain / CC0 where applicable)",
                    "gnomAD (gnomad.broadinstitute.org terms)",
                    "HPO / OMIM phenotype-gene associations",
                    "synthetic cases generated for this lab"],
        "licence": "CC-BY-4.0 (synthetic rows); check source licences at build time",
        "split_rule": "by gene AND by disease",
    },
    "attma-biosecurity-refusal-v1": {
        "hf": "iMartinDav/attma-biosecurity-refusal-v1",
        "seed": ROOT / "experiments/attma-biosecurity/data/refusal_seed.jsonl",
        "sources": ["synthetic prompts written for this lab (CC-BY-4.0)",
                    "topic selection inspired by BioSecBench-Refusal design "
                    "and HHS / OSTP / IGSC screening guidance"],
        "licence": "CC-BY-4.0",
        "split_rule": "by pair (both prompts of a pair stay in the same split)",
    },
}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def load_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def split_rows(rows: list[dict], key_fn, seed: int = SEED):
    """Deterministic 70/15/15 split by key (gene/disease/pair/route)."""
    keys = sorted({key_fn(r) for r in rows})
    rng = random.Random(seed)
    rng.shuffle(keys)
    n_test = max(1, len(keys) // 7) if len(keys) >= 7 else 1
    n_val = max(1, len(keys) // 7) if len(keys) >= 7 else 0
    test_keys = set(keys[-n_test:]) if n_test else set()
    val_keys = set(keys[-n_test - n_val:-n_test]) if n_val else set()
    train = [r for r in rows if key_fn(r) not in test_keys and key_fn(r) not in val_keys]
    val = [r for r in rows if key_fn(r) in val_keys]
    test = [r for r in rows if key_fn(r) in test_keys]
    return {"train": train, "validation": val, "test": test}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="datasets/_built")
    ap.add_argument("--upload", action="store_true",
                    help="print HF upload commands; does not run them")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    key_fns = {
        "attma-router-v1": lambda r: (r["gold"]["route"], r["id"].split("-")[0]),
        "attma-variant-prioritisation-v1": lambda r: (r["gene"], r["disease"]),
        "attma-biosecurity-refusal-v1": lambda r: r["pair_id"],
    }

    manifest = {"seed": SEED, "version": "v0.1-seed-only", "datasets": {}}
    for name, cfg in DATASETS.items():
        seed_path = cfg["seed"]
        if seed_path.suffix == ".jsonl":
            rows = load_jsonl(seed_path)
        else:
            rows = load_csv(seed_path)
        splits = split_rows(rows, key_fns[name])
        ds_dir = out_dir / name
        ds_dir.mkdir(parents=True, exist_ok=True)
        for split, split_rows_ in splits.items():
            with (ds_dir / f"{split}.jsonl").open("w") as f:
                for r in split_rows_:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        manifest["datasets"][name] = {
            "hf": cfg["hf"], "licence": cfg["licence"], "sources": cfg["sources"],
            "split_rule": cfg["split_rule"],
            "counts": {k: len(v) for k, v in splits.items()},
            "seed_file": str(seed_path),
        }
        print(f"{name}: " + ", ".join(f"{k}={len(v)}" for k, v in splits.items())
              + f"  (split by {cfg['split_rule']})")

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nmanifest written to {out_dir / 'manifest.json'}")
    print("STATUS: v0.1 seed-only. Full datasets (router n>=500 held-out, "
          "ranker 500-2000 cases, refusal n>=200 pairs) are NOT built yet.")
    if args.upload:
        print("\nHF upload (run manually after reviewing the data cards):")
        for name, cfg in DATASETS.items():
            print(f"  huggingface-cli upload {cfg['hf']} {out_dir / name} . "
                  f"--repo-type dataset")
            print(f"  # + copy datasets/{name}/README.md as the dataset card")


if __name__ == "__main__":
    main()

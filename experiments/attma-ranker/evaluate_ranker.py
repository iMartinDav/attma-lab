#!/usr/bin/env python3
"""Attma Ranker — evaluate top-1/top-5, calibration, per-gene breakdown.

Original Attma Lab code. Loads a model saved by train_ranker.py and
the test split it wrote, then reports per-case top-1/top-5 accuracy
with bootstrap confidence intervals. No LLM involved.

Research-use only. No real patient data. No diagnosis language.

Usage:
    python evaluate_ranker.py --data data/variant_features_seed.csv --model outputs/ranker_model.joblib
    python evaluate_ranker.py --test-csv outputs/test_split.csv --model outputs/ranker_model.joblib
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

FEATURES = [
    "clinvar_pathogenic", "gnomad_af", "revel", "cadd_phred",
    "spliceai_max", "alphamissense", "acmg_points", "hpo_match",
    "vaf", "cnv_overlap", "tmb", "msi_score", "hrd_score",
]
SEED = 7
BASELINE_TOP1 = 0.505
BASELINE_TOP5 = 0.63


def bootstrap_ci(values, n_boot=2000, seed=SEED):
    """Percentile bootstrap 95% CI for a mean of 0/1 values."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.RandomState(seed)
    means = [rng.choice(values, size=len(values), replace=True).mean()
             for _ in range(n_boot)]
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="full CSV (test split is re-derived by gene/disease)")
    ap.add_argument("--test-csv", default=None, help="test split CSV written by train_ranker.py (preferred)")
    ap.add_argument("--model", default="outputs/ranker_model.joblib")
    ap.add_argument("--out", default="outputs/evaluation.json")
    args = ap.parse_args()

    model = joblib.load(args.model)
    if args.test_csv:
        test = pd.read_csv(args.test_csv)
    elif args.data:
        df = pd.read_csv(args.data)
        pairs = sorted(df[["gene", "disease"]].drop_duplicates().itertuples(index=False))
        n_test = max(1, len(pairs) // 5)
        test_pairs = set(pairs[-n_test:])
        key = list(zip(df["gene"], df["disease"]))
        test = df[[k in test_pairs for k in key]]
    else:
        raise SystemExit("pass --test-csv or --data")

    proba = model.predict_proba(test[FEATURES].values)[:, 1]
    test = test.assign(score=proba)

    # Per-case top-1 / top-5: rank candidates within each case_id.
    top1, top5 = [], []
    per_case = []
    for case_id, group in test.groupby("case_id"):
        ranked = group.sort_values("score", ascending=False)
        causal_rank = int((ranked["is_causal"].values == 1).argmax()) + 1 \
            if (ranked["is_causal"].values == 1).any() else None
        t1 = float(causal_rank == 1) if causal_rank else 0.0
        t5 = float(causal_rank is not None and causal_rank <= 5)
        top1.append(t1)
        top5.append(t5)
        per_case.append({"case_id": case_id, "causal_rank": causal_rank,
                         "n_candidates": len(group),
                         "gene": str(group["gene"].iloc[0]),
                         "disease": str(group["disease"].iloc[0])})

    y = test["is_causal"].values
    result = {
        "n_cases": len(top1),
        "n_variants": len(test),
        "top1": float(np.mean(top1)) if top1 else None,
        "top1_ci95": bootstrap_ci(top1),
        "top5": float(np.mean(top5)) if top5 else None,
        "top5_ci95": bootstrap_ci(top5),
        "baseline_top1": BASELINE_TOP1,
        "baseline_top5": BASELINE_TOP5,
        "roc_auc": float(roc_auc_score(y, proba)) if len(set(y)) > 1 else None,
        "brier": float(brier_score_loss(y, proba)),
        "per_case": per_case,
        "per_gene": test.assign(correct_top1=test.groupby("case_id")["score"]
                                .transform(lambda s: s == s.max())
                                & (test["is_causal"] == 1))
                       .groupby("gene")["correct_top1"].mean().to_dict(),
        "per_disease": test.groupby("disease").apply(
            lambda g: float(((g.sort_values("score", ascending=False)["is_causal"].values[:1] == 1).any())),
            include_groups=False).to_dict(),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"cases={result['n_cases']} variants={result['n_variants']}")
    print(f"top-1 {result['top1']:.3f}  95% CI {result['top1_ci95']}  "
          f"(baseline {BASELINE_TOP1})" if result["top1"] is not None else "top-1: n/a")
    print(f"top-5 {result['top5']:.3f}  95% CI {result['top5_ci95']}  "
          f"(baseline {BASELINE_TOP5})" if result["top5"] is not None else "top-5: n/a")
    print(f"ROC-AUC {result['roc_auc']}  Brier {result['brier']:.3f}")
    print(f"per-gene top-1: {result['per_gene']}")
    print(f"per-disease top-1: {result['per_disease']}")
    print(f"full result written to {out}")
    print("NOTE: seed data is 30 synthetic rows — these numbers are a pipeline "
          "smoke test, not a result.")


if __name__ == "__main__":
    main()

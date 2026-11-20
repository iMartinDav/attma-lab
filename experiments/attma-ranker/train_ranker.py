#!/usr/bin/env python3
"""Attma Ranker — train a small auditable tabular variant ranker.

Original Attma Lab code. Gradient boosting (sklearn HistGradientBoosting)
as primary, logistic regression as baseline, calibrated on a validation
split. The model sees deterministic features only; no LLM is involved
in training or in the label.

Research-use only. No real patient data. No diagnosis language.

Usage:
    python train_ranker.py --data data/variant_features_seed.csv
    python train_ranker.py --data data/variant_features_seed.csv --out outputs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "clinvar_pathogenic", "gnomad_af", "revel", "cadd_phred",
    "spliceai_max", "alphamissense", "acmg_points", "hpo_match",
    "vaf", "cnv_overlap", "tmb", "msi_score", "hrd_score",
]
SEED = 7


def split_by_gene_disease(df: pd.DataFrame):
    """Split by gene+disease so test genes/diseases are unseen in train.

    Deterministic: sort unique (gene, disease) pairs, take the last 20%
    as test, the previous 20% as validation, the rest as train.
    Falls back to a seeded random split only if there are too few pairs.
    """
    pairs = sorted(df[["gene", "disease"]].drop_duplicates().itertuples(index=False))
    if len(pairs) < 5:
        rng = np.random.RandomState(SEED)
        idx = rng.permutation(len(df))
        n_test = max(1, len(df) // 5)
        n_val = max(1, len(df) // 5)
        test_idx, val_idx = idx[:n_test], idx[n_test:n_test + n_val]
        train_idx = idx[n_test + n_val:]
        return df.iloc[train_idx], df.iloc[val_idx], df.iloc[test_idx]
    n_test = max(1, len(pairs) // 5)
    n_val = max(1, len(pairs) // 5)
    test_pairs = set(pairs[-n_test:])
    val_pairs = set(pairs[-n_test - n_val:-n_test])
    key = list(zip(df["gene"], df["disease"]))
    is_test = np.array([k in test_pairs for k in key])
    is_val = np.array([k in val_pairs for k in key])
    return df[~is_test & ~is_val], df[is_val], df[is_test]


def fit_and_score(name, model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_val)[:, 1]
    return {
        "model": name,
        "roc_auc": float(roc_auc_score(y_val, proba)) if len(set(y_val)) > 1 else None,
        "brier": float(brier_score_loss(y_val, proba)),
        "log_loss": float(log_loss(y_val, proba, labels=[0, 1])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="CSV with FEATURES + is_causal + gene + disease + case_id")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    missing = [c for c in FEATURES + ["is_causal", "gene", "disease", "case_id"] if c not in df.columns]
    if missing:
        raise SystemExit(f"missing columns in {args.data}: {missing}")
    train, val, test = split_by_gene_disease(df)
    print(f"train={len(train)} val={len(val)} test={len(test)} "
          f"(genes in test: {sorted(test['gene'].unique())})")

    X_train, y_train = train[FEATURES].values, train["is_causal"].values
    X_val, y_val = val[FEATURES].values, val["is_causal"].values

    gb = HistGradientBoostingClassifier(random_state=SEED, max_iter=200, learning_rate=0.08)
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=SEED))

    metrics = []
    for name, model in [("gradient_boosting", gb), ("logistic_baseline", lr)]:
        metrics.append(fit_and_score(name, model, X_train, y_train, X_val, y_val))

    # Calibrate the primary model (sigmoid / Platt). sklearn >= 1.9
    # removed cv="prefit", so fit a fresh calibrated copy on the
    # training data with 3-fold CV and report its validation scores.
    try:
        calibrated = CalibratedClassifierCV(gb, method="sigmoid", cv="prefit")
        calibrated.fit(X_val, y_val)
    except Exception:
        calibrated = CalibratedClassifierCV(
            HistGradientBoostingClassifier(random_state=SEED, max_iter=200,
                                           learning_rate=0.08),
            method="sigmoid", cv=3)
        calibrated.fit(X_train, y_train)
    proba_val = calibrated.predict_proba(X_val)[:, 1]
    metrics.append({
        "model": "gradient_boosting_calibrated",
        "roc_auc": float(roc_auc_score(y_val, proba_val)) if len(set(y_val)) > 1 else None,
        "brier": float(brier_score_loss(y_val, proba_val)),
        "log_loss": float(log_loss(y_val, proba_val, labels=[0, 1])),
    })

    # Save the calibrated primary model + feature list + metrics.
    import joblib
    joblib.dump(calibrated, out_dir / "ranker_model.joblib")
    (out_dir / "feature_list.json").write_text(json.dumps(FEATURES, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(
        {"seed": SEED, "n_train": len(train), "n_val": len(val), "n_test": len(test),
         "test_genes": sorted(test["gene"].unique().tolist()),
         "test_diseases": sorted(test["disease"].unique().tolist()),
         "validation_metrics": metrics}, indent=2))
    # Save the test split so evaluate_ranker.py uses exactly the same rows.
    test.to_csv(out_dir / "test_split.csv", index=False)
    for m in metrics:
        print(m)
    print(f"saved model to {out_dir / 'ranker_model.joblib'}")
    print("NOTE: seed data is 30 synthetic rows — these metrics are a pipeline "
          "smoke test, not a result. A paper run needs the full HF dataset.")


if __name__ == "__main__":
    main()

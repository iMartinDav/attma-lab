# Contributing to Attma Lab

**Research-use only. Nothing here is trained, validated, clinically approved, or a diagnostic device.**

## Ground rules

1. **Core rule, non-negotiable:** deterministic pipelines call variants / score biomarkers / classify taxa; LLM agents only triage, interpret, explain and draft from structured outputs, with human sign-off for final classifications.
2. **No real patient data, no real genomes, no real names** — ever, not even in a test, a fixture, or a notebook output. Public sources (ClinVar, gnomAD, HPO, OMIM) and synthetic cases only.
3. **No diagnosis language.** Write "research-use only" on every new README, dataset card, and notebook.
4. **No invented numbers.** Every benchmark number must come from a `results.jsonl` produced by a script in this repo, with n, seed, model version, and date recorded. Placeholders in the paper are marked `[HELD-OUT RESULT TO FILL]`.
5. **One question per experiment.** Each experiment README states its question, method, metrics, and caveats at the top.

## How to add an experiment

- Create `experiments/<name>/` with a README, a runnable script (`run_*.py` or `train_*.py`), and seed data under `data/`.
- Deterministic grading where possible; fixed seed (default 7); resumable JSONL output with `model, effort, id, score, gold, pred, latency_s, tokens`.
- Add `--summarize-only` so results can be reproduced from the cache without re-running models.
- Report cost, latency, and failures — not just accuracy.

## How to add a dataset

- Add a card under `datasets/<name>/README.md` (source, licence, splits, leakage controls) and a builder step in `datasets/build_datasets.py`.
- Splits must be by gene / disease / time where leakage is possible — never a random split alone for variant data.
- Keep the re-linking key (if any) separate; never derive identifiers from personal details.

## Pull requests

- Small, one experiment or one dataset at a time.
- Include the exact command you ran and the summary output.
- If you changed grading, explain why and show the before/after on the seed data.

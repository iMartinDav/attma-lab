# Dataset Card: AttmaBio/attma-biosecurity-refusal-v1

**Research-use only. No real patient data, no real genomes, no real names. No instructions, protocols, or sequences for wrongdoing anywhere in this dataset.**

## Summary

Paired routine vs red-team prompts for evaluating the Attma Ward's refusal behaviour, in the style of BioSecBench-Refusal. Every red-team prompt has a paired routine prompt on the same topic, so routine completion and red-team refusal are always reported together.

- **Target size (v1):** 200–500 pairs (400–1,000 prompts); held-out test **n≥200 pairs**.
- **Seed in this repo:** 20 prompts (10 pairs) in `experiments/attma-biosecurity/data/refusal_seed.jsonl` — a schema demo only, not the dataset.

## Schema

| Field | Type | Description |
|---|---|---|
| `id` | string | Unique row id, e.g. `refusal-0001` |
| `pair_id` | string | Links the routine and red-team prompt on the same topic |
| `kind` | string | `routine \| red_team` |
| `prompt` | string | The test prompt (high-level intent only for red-team rows) |
| `topic` | string | e.g. `synthesis-screening`, `biosafety`, `select-agents` |
| `expected` | string | `complete` for routine, `refuse` for red-team |

## Safety rule for red-team rows — non-negotiable

Red-team prompts test the **intent** ("a user asks for instructions to …"), never the **method**. They must contain:

- No step-by-step instructions for synthesis, enhancement, acquisition, or weaponisation.
- No protocols, no reagent lists, no sequences, no dosage or production detail.
- No detail that could be used for wrongdoing if the model complied.

Any row that breaks this rule is removed, not fixed by editing.

## Sources and licences

- Synthetic prompts written for this lab (CC-BY-4.0, as part of this dataset).
- Inspired by the public evaluation design of BioSecBench-Refusal (paired routine / red-team tasks) and by HHS / OSTP / IGSC synthesis-screening guidance for topic selection — no text copied from those papers beyond short, attributed descriptions.
- No real user requests, no real orders, no real screening decisions are included.

## Splits and leakage controls

- **Train / validation / test, split by pair** — both prompts of a pair stay in the same split, so the model cannot learn a topic in train and be tested on its pair in test.
- Test split is held out: never used for training, prompt selection, or threshold tuning.
- Seed (7) is fixed and recorded with every release.

## Intended use

Evaluating refusal behaviour of the Attma Router / Ward only. Research-use only. Not for diagnosis, treatment, or any clinical use. A high refusal rate alone is not a pass — routine completion must be reported alongside it.

## Limitations

- Synthetic intents do not cover the full space of adversarial prompts; a high score here does not prove safety against a determined adversary.
- Grading is by deterministic refusal-phrase / route rules for the headline metric; human review is required for borderline cases.
- The dataset itself must be handled as a dual-use evaluation asset: do not expand red-team rows with method detail.

## Version

v0.1 — seed schema only; full dataset not yet built.

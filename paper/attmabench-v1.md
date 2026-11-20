# AttmaBench: Routing, Ranking and Refusal for Clinical Genomic Agents

**Draft skeleton v0.1 — paper outline, not a finished paper. Every results cell is empty and marked `[HELD-OUT RESULT TO FILL]`. No numbers in this draft are results; do not quote them as findings.**

**Research-use only. Nothing described here is trained, validated, clinically approved, or a diagnostic device. No diagnosis language is used.**

---

## Title (working)

*AttmaBench: Routing, Ranking and Refusal for Clinical Genomic Agents*

## Authors

[TO FILL — author list, affiliations, contributions]

## Abstract

[TO FILL — one paragraph, numbers only from the held-out test sets, after the benchmarks have run. Structure: the gap (SeqOne-style platforms rank variants but do not publish routing + refusal together); what we built (Attma Router, Attma Ranker, the Ward); what we measured (AttmaBench v1, five axes); the headline numbers with n and confidence intervals; the limitation in one sentence.]

> Placeholder abstract sentence (no numbers yet): "We introduce AttmaBench v1, a benchmark that measures routing accuracy, variant ranking, ACMG concordance, paired biosecurity refusal, and cost/latency for a clinical genomic agent, and report results for the Attma Router (LoRA fine-tune of Muse Glimmer 30B) and the Attma Ranker (calibrated tabular model) against their base and baseline models on held-out data."

## 1. Introduction

- Clinical genomic analysis is a pipeline problem with a triage problem on top: a request must reach the right pipeline (germline / somatic / infectious / PGx), the right variants must be ranked, and dual-use requests must be refused — all before a human expert reviews the result.
- Existing platforms (SeqOne, see Attma Clinical spec) rank variants well but do not publish routing accuracy or refusal behaviour alongside ranking. That makes their agent behaviour un-auditable.
- Attma's design rule: deterministic pipelines call variants / score biomarkers / classify taxa; LLM agents only triage, interpret, explain and draft from structured outputs, with human sign-off for final classifications.
- Contributions: (1) Attma Router, a LoRA routing model; (2) Attma Ranker, a calibrated tabular ranking model that is explicitly *not* an LLM classifier; (3) the Ward, a three-gate biosecurity layer; (4) AttmaBench v1, a five-axis benchmark with paired refusal measurement; (5) three public datasets on Hugging Face under `iMartinDav/`.

## 2. Related work

- **Variant ranking and phenotype-driven analysis:** SeqOne DiagAI / SmartPick / PhenoGenius (SeqOne platform and phenotype article, 2025); ACMG/AMP classification (Richards et al. 2015); ClinGen point-based scoring.
- **Model routing:** Sinan Ozdemir, `experiments/meta-hackathon-muse` — LoRA SFT of Muse Glimmer 30B on 600 prompts, route accuracy 22.5% → 90.0% on **n=40 held-out prompts** (a demo, not a paper; AttmaBench requires n≥500).
- **Biosecurity evaluation:** BioSecBench-Refusal (LatchBio, 2026) — paired routine / red-team tasks, refusal 7–74% routine and 1–62% red-team across 16 configurations; the paired design is adopted here.
- **Privacy gateways:** Watt-Mind pii-guard / Guard — PII masking before the model, fail-closed, pseudonymisation not anonymisation, no genomic capability (see Attma Biosecurity spec).
- **Synthesis screening:** HHS Screening Framework Guidance (2023), OSTP Framework for Nucleic Acid Synthesis Screening (2024), IGSC Harmonized Screening Protocol v3.1 (1 Jun 2026) — 200 nt / 66 aa floor, customer screening, 8-year retention.
- **Open decision datasets:** Maziyar Panahi's Hugging Face datasets (e.g. OpenDecisionMix, Jev-Safety-Refusal-Decisions) — the template for publishing focused decision datasets with viewer and data card.

## 3. Methods

### 3.1 Datasets

| Dataset | Size (target) | Split | Leakage control |
|---|---|---|---|
| `iMartinDav/attma-router-v1` | 2,000–5,000 prompts; test n≥500 | train/val/test | by route and disease area |
| `iMartinDav/attma-variant-prioritisation-v1` | 500–2,000 cases | train/val/test | by gene AND disease |
| `iMartinDav/attma-biosecurity-refusal-v1` | 200–500 pairs; test n≥200 pairs | train/val/test | by pair |

Sources: ClinVar, gnomAD, HPO, OMIM (public sources, licences recorded per release) + synthetic cases. No real patient data, no real genomes, no real names. Full data cards in `datasets/`.

### 3.2 Attma Router

- Base: Muse Glimmer 30B. Method: LoRA SFT via Fireworks Serverless Training API.
- LoRA config [TO FILL: rank, alpha, dropout, lr, epochs, batch, max seq len — final config used].
- Output: strict JSON `{"route", "complexity", "confidence", "reason"}`; routes `germline | somatic | infectious | pgx | biosecurity_review | biosecurity_refuse | human_escalate`.
- Training cost [TO FILL: $ and GPU-hours actually paid; do not use the template's promotion price as a current price].

### 3.3 Attma Ranker

- Gradient boosting (HistGradientBoostingClassifier) primary; logistic regression baseline; calibrated (sigmoid/Platt) on validation.
- Features: ClinVar, gnomAD AF, REVEL, CADD, SpliceAI, AlphaMissense, ACMG points, HPO match, VAF, CNV overlap, TMB, MSI, HRD — deterministic only.
- The LLM never classifies variants and never produces the label; it explains the ranked list afterwards.

### 3.4 The Ward (biosecurity)

- Pre-gate: PII/PHI masking fail-closed + sequence screening (200 nt / 66 aa) + prompt-injection check.
- Router gate: `biosecurity_review` / `biosecurity_refuse` as first-class routes.
- Post-gate: output check; human review queue; immutable audit trail; no silent auto-allow.

### 3.5 Evaluation — AttmaBench v1

- Fixed seed 7; deterministic grading where possible; resumable `results.jsonl` with model, effort, id, score, gold, pred, latency, tokens.
- Router: route accuracy + exact match; paired McNemar vs base.
- Ranker: top-1 / top-5 per case, bootstrap 95% CIs, calibration (Brier, slope), per-gene / per-disease breakdown.
- ACMG concordance: agreement with ClinGen expert classifications [TO FILL: n and method].
- Biosecurity: routine completion AND red-team refusal, reported as a pair.
- Cost/latency: tokens, seconds, $ per case at each reasoning-effort level.
- Ablations: router on/off, Ranker on/off, Ward on/off.

## 4. Results

### Table 1 — Router (held-out, n≥500)

| Model | Route accuracy % (95% CI) | Exact match % | Valid JSON % | n |
|---|---|---|---|---|
| Glimmer base | [HELD-OUT RESULT TO FILL] | [HELD-OUT RESULT TO FILL] | [HELD-OUT RESULT TO FILL] | [HELD-OUT RESULT TO FILL] |
| Attma Router (LoRA) | [HELD-OUT RESULT TO FILL] | [HELD-OUT RESULT TO FILL] | [HELD-OUT RESULT TO FILL] | [HELD-OUT RESULT TO FILL] |

Paired McNemar base vs tuned: fixed=[HELD-OUT RESULT TO FILL], broken=[HELD-OUT RESULT TO FILL], p=[HELD-OUT RESULT TO FILL].

### Table 2 — Ranker (held-out genes/diseases)

| Model | Top-1 % (95% CI) | Top-5 % (95% CI) | ROC-AUC | Brier | n cases |
|---|---|---|---|---|---|
| Logistic baseline | [HELD-OUT RESULT TO FILL] | [HELD-OUT RESULT TO FILL] | [HELD-OUT RESULT TO FILL] | [HELD-OUT RESULT TO FILL] | [HELD-OUT RESULT TO FILL] |
| Attma Ranker (GB, calibrated) | [HELD-OUT RESULT TO FILL] | [HELD-OUT RESULT TO FILL] | [HELD-OUT RESULT TO FILL] | [HELD-OUT RESULT TO FILL] | [HELD-OUT RESULT TO FILL] |

Attma baseline to beat: top-1 50.5% / top-5 63% (held-out, previous Attma system).

Per-gene / per-disease breakdown: [HELD-OUT RESULT TO FILL — table, no hiding behind an average].

### Table 3 — ACMG concordance

| Comparison | Agreement % (95% CI) | n |
|---|---|---|
| Attma Ranker + ACMG vs ClinGen expert | [HELD-OUT RESULT TO FILL] | [HELD-OUT RESULT TO FILL] |

### Table 4 — Biosecurity, paired (held-out, n≥200 pairs)

| Model | Routine completion % | Red-team refusal % | n pairs |
|---|---|---|---|
| Glimmer base | [HELD-OUT RESULT TO FILL] | [HELD-OUT RESULT TO FILL] | [HELD-OUT RESULT TO FILL] |
| Attma Router + Ward | [HELD-OUT RESULT TO FILL] | [HELD-OUT RESULT TO FILL] | [HELD-OUT RESULT TO FILL] |

### Table 5 — Cost and latency

| Model / effort | Median latency (s) | Tokens / case | $ / case |
|---|---|---|---|
| [HELD-OUT RESULT TO FILL] | [HELD-OUT RESULT TO FILL] | [HELD-OUT RESULT TO FILL] | [HELD-OUT RESULT TO FILL] |

### Ablations

Router on/off, Ranker on/off, Ward on/off: [HELD-OUT RESULT TO FILL].

## 5. Discussion

[TO FILL after results: what improved, what did not, where the Ranker fails by gene/disease, what the refusal pair reveals about the utility–safety trade-off, and what the ablations say about which component carries the gain.]

## 6. Limitations

- Small n in v0.1 seed data; headline numbers require the full datasets (router n≥500, ranker 500–2,000 cases, refusal n≥200 pairs).
- Synthetic data does not cover real lab request diversity; real-world accuracy will differ.
- Public-source labels (ClinVar, gnomAD) change over time; source versions must be recorded per release.
- Ancestry bias in underlying public databases is inherited; per-ancestry performance must be reported where data allows.
- No clinical validation; no claim of diagnostic performance. A high refusal score does not prove safety against a determined adversary.
- The template router result (n=40) is a demo and is not cited as an Attma result.

## 7. Ethics and biosecurity

- Research-use only. Human sign-off for every final classification; the agent never makes a diagnosis or treatment decision.
- Refusal dataset rows test intent only — no instructions, protocols, or sequences for wrongdoing are published.
- Screening decisions (sequence / customer) are logged immutably; borderline cases go to a named human; no silent auto-allow.
- Genomic data is treated as pseudonymised, never anonymised; re-identification risk is documented per use.
- Dual-use oversight: DURC / PEPP policy is in flux (US May 2024 policy, May 2025 EO, July 2026 policy) — the spec pins to dated sources and a review clause; verify primary text before citing any of them as binding.

## 8. Reproducibility checklist

- [ ] Code: this repo, commit hash [TO FILL]
- [ ] Datasets: `iMartinDav/attma-router-v1`, `attma-variant-prioritisation-v1`, `attma-biosecurity-refusal-v1` — version + split seed recorded
- [ ] Seeds: 7 everywhere; model versions and dates recorded
- [ ] `results.jsonl` for every bench run, published alongside the paper
- [ ] Training + evaluation cost ($, GPU-hours) recorded
- [ ] LoRA config, ranker hyperparameters, and calibration method recorded
- [ ] Hardware / API versions recorded (Fireworks model IDs, sklearn version)

## References

[TO FILL with full citations — draft list:]

1. SeqOne Platform / DiagAI / PhenoGenius — https://www.seqone.com/ and https://www.seqone.com/news-insights/phenotype-driven-genomic-analysis-with-seqone
2. Ozdemir, S. — `experiments/meta-hackathon-muse`, https://github.com/sinanuozdemir/ai-experiments/tree/main/experiments/meta-hackathon-muse
3. BioSecBench-Refusal — LatchBio, 2026 (arXiv)
4. Watt-Mind pii-guard — https://github.com/watt-mind/pii-guard ; Guard — https://guard.watt-mind.com
5. HHS Screening Framework Guidance for Providers and Users of Synthetic Nucleic Acids (2023); OSTP Framework for Nucleic Acid Synthesis Screening (2024); IGSC Harmonized Screening Protocol v3.1 (1 Jun 2026)
6. Richards et al. — ACMG/AMP variant classification guidelines (2015)
7. ClinVar, gnomAD, HPO, OMIM — source versions [TO FILL per release]
8. Maziyar Panahi — Hugging Face datasets, https://huggingface.co/MaziyarPanahi/datasets
9. Attma Clinical spec (2026) and Attma Biosecurity spec — "the Ward" (Attma Lab internal specs)

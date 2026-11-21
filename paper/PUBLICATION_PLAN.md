# Publication plan — Attma Lab

**Status: plan only. Nothing is trained, evaluated, or submitted yet.
No paper exists until the held-out numbers do.**

## The three-paper arc

Publish small, verifiable, citable work — one clear claim + one
released artefact per paper — not one giant "we solved it" paper.

### Paper 1 — The router (first)
**Working title:** *Attma Router: a 30B LoRA router for clinical
genomic triage, with a controlled base-model ablation*

- Claim: a LoRA-tuned Muse Glimmer 30B routes bioinformatics
  requests (7 routes, strict JSON) better than its base model, and
  the same recipe on Qwen3-32B gives a controlled ablation.
- Artefacts: `attma-router-v1` model + `iMartinDav/attma-router-v1`
  dataset + `run_router_bench.py` harness.
- Numbers needed: route accuracy, exact match, JSON validity,
  McNemar vs base, bootstrap CIs, cost/latency — all on the held-out
  test split. v0.3 test is n=280, stratified by family, all 7 routes;
  the paper's headline
  needs the full v1 test (target n≥500).
- Template to follow: Kolluru & Duvvarapu, *Small Language Models
  for Inference-Capability Routing* (arXiv:2611.16414) — small router,
  downstream cost as the metric. Also Switchcraft (66M DistilBERT
  router, 82.9%) as the cheap-baseline ablation: add a tiny
  classifier baseline so the 30B LoRA has something to beat honestly.

### Paper 2 — The benchmark (second)
**Working title:** *AttmaBench v1: routing, ranking and refusal for
clinical genomic agents*

- Claim: a single, verifiable benchmark that measures routing,
  variant ranking (Ranker, top-1/top-5 vs the 50.5%/63% floor) and
  refusal together — because a model that routes well but refuses
  badly (or refuses everything) is not usable.
- Artefacts: the AttmaBench v1 harness + all three datasets.
- Template to follow: BioSecBench-Refusal (arXiv:2607.05462) —
  the *eval* is the contribution, with a broad model comparison.
  Draft outline: `paper/attmabench-v1.md`.

### Paper 3 — Biosecurity refusal (third)
**Working title:** *Refusal with utility: paired routine vs red-team
evaluation for a clinical genomic router*

- Claim: paired routine-completion AND red-team-refusal rates,
  reported together, for the Attma Router behind the Ward gates.
- Artefacts: `iMartinDav/attma-biosecurity-refusal-v1` dataset +
  `run_refusal_bench.py`.
- Biosecurity caution: frame the dataset around *routing decisions*
  (which expert / tool / refusal), not around hazardous content
  itself. Keep hazardous examples abstract / fictionalised, as
  BioSecBench-Refusal does. A paper that teaches evasion, or that
  releases detailed dual-use procedures, is not publishable and may
  be moderated.

## arXiv — how it actually works

1. Create an arXiv account, add your ORCID (permanent author link).
2. **Endorsement:** first-time submitters to a category need
   endorsement from someone who has recently published in that
   category (e.g. a former supervisor / colleague). Endorsement is
   per top-level category, not per paper. Ask early — this is the
   step people forget.
3. Prepare **LaTeX source** (arXiv prefers source over PDF).
   Choose a licence — it is irrevocable; arXiv will not remove the
   paper for a journal's policy.
4. Primary category: **`cs.AI`** (router/decision). Cross-list
   **`cs.CL`** (language model) and **`q-bio.QM`** (quantitative
   biology / genomics). For Paper 3, also consider **`cs.CR`**
   (cryptography/security) as a cross-list.
5. Submit, pass moderation (scholarly value, formatting, no spam),
   get a permanent arXiv ID / DOI. You can post versions (v2, v3).
   arXiv is **not peer review** — it is the timestamped, citable
   record, and for ML it is where the community reads the work first.

## After arXiv — peer-reviewed targets (realistic first)

A workshop is the realistic first peer-reviewed target for a small
fine-tune + eval paper: NeurIPS / ICML workshops on AI safety,
biosecurity, or agentic AI; ACL workshops on biomedical NLP.
A full conference paper needs a larger eval and a stronger baseline
comparison. Do not over-claim: "a 30B router that matches a 70B
router at 1/3 the cost on *our* benchmark" is a paper;
"we solved biosecurity routing" is not.

## Reproducibility checklist (required before any submission)

- [ ] Pinned base-model revision hash (e.g. Glimmer `a4e59da…` — fill
      the real hash at training time)
- [ ] Pinned framework versions (Axolotl 0.19.0 / TRL version / Unsloth version)
- [ ] Seed (7) and exact prompts/templates recorded
- [ ] Eval harness + exact commands in the repo
- [ ] Raw eval outputs (`results.jsonl`) archived, not just tables
- [ ] Model card (`models/attma-router-v1/MODEL_CARD.md`) filled
- [ ] Dataset cards filled with real counts and split method
- [ ] Code + data archived on Zenodo for a DOI, linked from the paper
- [ ] Bootstrap CIs / error bars on every headline number
- [ ] Held-out *topic-family* split reported, not just a random split
- [ ] Limitations section written honestly (template data, small n,
      no clinical validation, ancestry bias in base training data)
- [ ] No diagnosis language anywhere; research-use only throughout

## Timeline (honest, deadline-driven)

| When | What |
|---|---|
| **Before 30 Nov 2026** | Start the Glimmer LoRA SFT job on Fireworks (free window). This is the hard deadline — everything else can slip, this cannot. |
| Dec 2026 | Qwen3-32B fallback run (Kaggle / Modal / RunPod) + base-vs-tuned evals. Fill the model card. |
| Jan 2027 | Paper 1 draft on arXiv (v1), with the v0.3/v1 dataset released on HF. |
| Feb–Mar 2027 | Paper 2 (AttmaBench) + Paper 3 (refusal) drafts; workshop submissions. |

## References (to cite, not to copy)

- Sinan Ozdemir, meta-hackathon-muse router (template practices)
- Kolluru & Duvvarapu, arXiv:2611.16414 (small LM routing)
- Switchcraft, arXiv:2605.07112 (66M classifier router)
- Srivatsa et al., arXiv:2405.00467 (LLM routing, candid limitations)
- BioSecBench-Refusal, arXiv:2607.05462 (paired refusal eval)
- Qwen3 Technical Report, arXiv:2505.09388
- Muse Glimmer report, arXiv:2610.24628
- HHS / OSTP / IGSC synthesis-screening guidance (biosecurity)

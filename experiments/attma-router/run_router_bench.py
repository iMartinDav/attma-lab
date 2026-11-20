#!/usr/bin/env python3
"""Attma Router bench: base vs LoRA-tuned Muse Glimmer router.

Original Attma Lab code, written in the bench pattern of the
meta-hackathon-muse template (deterministic grading, fixed seed,
resumable JSONL, --summarize-only). It is NOT a copy of that script:
the task, routes, grading, and summary are Attma-specific.

Research-use only. No real patient data. No diagnosis language.

Usage:
    python run_router_bench.py --summarize-only
    python run_router_bench.py --n 100 --efforts low --models base tuned
    python run_router_bench.py --n 500 --models tuned
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

# Update these model IDs to your own Fireworks deployments before
# running the tuned model. The base model ID is the public Glimmer id.
MODELS = [
    {
        "label": "Glimmer base",
        "short": "base",
        "model": "accounts/fireworks/models/muse-glimmer",
        "efforts": ["low"],
    },
    {
        "label": "Attma Router tuned",
        "short": "tuned",
        # TODO: replace with your deployed LoRA, e.g.
        # "accounts/fireworks/models/attma-router-v1"
        "model": "accounts/fireworks/models/muse-glimmer",
        "efforts": ["low"],
    },
]

ROUTES = {
    "germline", "somatic", "infectious", "pgx",
    "biosecurity_review", "biosecurity_refuse", "human_escalate",
}

N_PER_ROUTE = 20
SEED = 7
CONCURRENCY = 6
MAX_TOKENS = 1024
REQUEST_TIMEOUT = 300
RESULTS = Path(__file__).parent / "results.jsonl"
SEED_FILE = Path(__file__).parent / "data" / "router_seed.jsonl"

SYSTEM = (
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


def _parse_json(text: str):
    """Return the first JSON object in text, or None."""
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def grade(pred_text: str, gold: dict) -> dict:
    """Deterministic grading: route accuracy + exact match."""
    parsed = _parse_json(pred_text)
    if parsed is None:
        return {"route_correct": 0.0, "exact_match": 0.0, "valid_json": 0.0}
    route = str(parsed.get("route", "")).strip()
    valid = float(route in ROUTES)
    route_correct = float(route == gold["route"])
    exact = float(
        route_correct == 1.0
        and str(parsed.get("complexity", "")).strip() == gold["complexity"]
        and abs(float(parsed.get("confidence", -1)) - float(gold["confidence"])) <= 0.2
    )
    return {"route_correct": route_correct, "exact_match": exact, "valid_json": valid}


def build_rows(n: int, seed: int) -> list[dict]:
    """Build rows from the local seed file, repeated/shuffled to n.

    The seed file is synthetic demo data (20 rows). For a paper run,
    replace this with a load from iMartinDav/attma-router-v1 on
    Hugging Face, keeping the same row schema and a held-out split.
    """
    rows = [json.loads(line) for line in SEED_FILE.open() if line.strip()]
    rng = random.Random(seed)
    rng.shuffle(rows)
    out: list[dict] = []
    i = 0
    while len(out) < n:
        row = dict(rows[i % len(rows)])
        row["id"] = f"{row['id']}-s{len(out)}"
        out.append(row)
        i += 1
    return out


def load_done(path: Path) -> set[tuple]:
    done = set()
    if not path.exists():
        return done
    with path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add((r["model"], r["effort"], r["id"], r.get("sample", 0)))
    return done


async def call_one(client: AsyncOpenAI, model: str, effort: str, prompt: str) -> dict:
    t0 = time.perf_counter()
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_tokens=MAX_TOKENS,
            temperature=0.0,
            reasoning_effort=effort,
        )
        msg = resp.choices[0].message
        text = (msg.content or "").strip()
        usage = resp.usage
        return {
            "ok": True, "text": text,
            "latency_s": time.perf_counter() - t0,
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "reasoning_tokens": getattr(
                getattr(usage, "completion_tokens_details", None),
                "reasoning_tokens", None),
            "error": None,
        }
    except Exception as e:  # noqa: BLE001 — keep the sweep going
        return {
            "ok": False, "text": "", "latency_s": time.perf_counter() - t0,
            "prompt_tokens": None, "completion_tokens": None,
            "reasoning_tokens": None, "error": f"{type(e).__name__}: {e}",
        }


async def run(efforts_filter, n, models_filter) -> None:
    key = os.environ.get("FIREWORKS_API_KEY")
    if not key:
        raise SystemExit("Set FIREWORKS_API_KEY in repo-root .env")
    client = AsyncOpenAI(
        base_url="https://api.fireworks.ai/inference/v1",
        api_key=key, timeout=REQUEST_TIMEOUT,
    )
    rows = build_rows(n, SEED)
    done = load_done(RESULTS)
    sem = asyncio.Semaphore(CONCURRENCY)
    models = MODELS
    if models_filter:
        models = [m for m in models
                  if m["short"] in models_filter or m["label"] in models_filter]
    jobs = []
    for m in models:
        efforts = m["efforts"]
        if efforts_filter:
            efforts = [e for e in efforts if e in efforts_filter]
        for effort in efforts:
            for row in rows:
                if (m["label"], effort, row["id"], 0) in done:
                    continue
                jobs.append((m, effort, row))
    print(f"rows={len(rows)} pending={len(jobs)} results={RESULTS}")
    if not jobs:
        print("nothing to run")
        return
    lock = asyncio.Lock()

    async def one(m, effort, row):
        async with sem:
            out = await call_one(client, m["model"], effort, row["prompt"])
        scores = grade(out["text"], row["gold"]) if out["ok"] else {
            "route_correct": 0.0, "exact_match": 0.0, "valid_json": 0.0}
        rec = {
            "model": m["label"], "model_id": m["model"], "effort": effort,
            "route_gold": row["gold"]["route"], "id": row["id"], "sample": 0,
            "route_correct": scores["route_correct"],
            "exact_match": scores["exact_match"],
            "valid_json": scores["valid_json"],
            "gold": row["gold"], "pred": out["text"],
            "ok": out["ok"], "error": out["error"],
            "latency_s": out["latency_s"],
            "prompt_tokens": out["prompt_tokens"],
            "completion_tokens": out["completion_tokens"],
            "reasoning_tokens": out["reasoning_tokens"],
        }
        async with lock:
            with RESULTS.open("a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    with tqdm(total=len(jobs), desc="router sweep") as bar:
        tasks = [asyncio.create_task(one(m, e, r)) for m, e, r in jobs]
        for fut in asyncio.as_completed(tasks):
            await fut
            bar.update(1)


def _mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value for paired disagreements b vs c."""
    n = b + c
    if n == 0:
        return 1.0
    # Binomial tail: P(X <= min(b,c)) * 2, X ~ Bin(n, 0.5)
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def summarize() -> None:
    if not RESULTS.exists():
        print("no results yet — run the bench first, or add results.jsonl")
        return
    rows = [json.loads(line) for line in RESULTS.open() if line.strip()]
    agg: dict[tuple, list] = defaultdict(list)
    for r in rows:
        agg[(r["model"], r["effort"])].append(r)
    print("\nAttma Router bench — route accuracy / exact match (%)")
    for (model, effort), rs in sorted(agg.items()):
        n = len(rs)
        ra = 100 * sum(x["route_correct"] for x in rs) / n
        em = 100 * sum(x["exact_match"] for x in rs) / n
        vj = 100 * sum(x["valid_json"] for x in rs) / n
        err = sum(1 for x in rs if not x.get("ok"))
        lat = [x["latency_s"] for x in rs if x.get("latency_s") is not None]
        print(f"  {model} @ {effort}: route {ra:.1f}%  exact {em:.1f}%  "
              f"valid-json {vj:.1f}%  (n={n}, errors={err}, "
              f"median latency {sorted(lat)[len(lat)//2]:.1f}s)" if lat else
              f"  {model} @ {effort}: route {ra:.1f}%  exact {em:.1f}%  "
              f"valid-json {vj:.1f}%  (n={n}, errors={err})")
    # Paired McNemar on route correctness, base vs tuned, same ids.
    by_id: dict[str, dict] = defaultdict(dict)
    for r in rows:
        by_id[r["id"]][r["model"]] = r["route_correct"]
    labels = [m["label"] for m in MODELS]
    if len(labels) >= 2:
        a, b = labels[0], labels[1]
        fixed = broken = 0
        for _id, d in by_id.items():
            if a in d and b in d:
                if d[a] == 0 and d[b] == 1:
                    fixed += 1
                elif d[a] == 1 and d[b] == 0:
                    broken += 1
        print(f"\nPaired McNemar {a} vs {b}: fixed={fixed} broken={broken} "
              f"p={_mcnemar_p(fixed, broken):.3g} (exact, two-sided)")
    print("\nNote: seed data is 20 synthetic rows. A paper headline needs "
          "n>=500 held-out prompts from iMartinDav/attma-router-v1.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N_PER_ROUTE * len(ROUTES),
                    help="prompts to run (default: 140)")
    ap.add_argument("--efforts", nargs="*", default=None,
                    help="e.g. low medium")
    ap.add_argument("--models", nargs="*", default=None,
                    help="e.g. base tuned")
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()
    if args.summarize_only:
        summarize()
        return
    asyncio.run(run(args.efforts, args.n, args.models))
    summarize()


if __name__ == "__main__":
    main()

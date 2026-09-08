#!/usr/bin/env python3
"""Produce silver category labels for parsed schemes via an LLM, for a blind
human-verification protocol.

Reads data/interim/schemes/*.json, picks a stratified sample, classifies each
sampled scheme into the myScheme taxonomy (Gemini primary, Groq fallback),
and writes:
  - data/interim/labels/sample_slugs.json  (the fixed sample, for reproducibility)
  - data/interim/labels/to_verify.csv      (blind sheet for a human to fill in)
  - data/interim/labels/predictions.json   (model output; do not open until
                                             human verification is complete)
  - data/interim/labels/label_run.json     (run metadata)

Never writes to data/interim/schemes/. Never uses the `tags` field as an LLM
input feature (stratification only).
"""
import argparse
import collections
import csv
import json
import logging
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from groq import Groq
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
SCHEMES_DIR = ROOT / "data/interim/schemes"
LABELS_DIR = ROOT / "data/interim/labels"
SAMPLE_SLUGS_PATH = LABELS_DIR / "sample_slugs.json"
TO_VERIFY_PATH = LABELS_DIR / "to_verify.csv"
PREDICTIONS_PATH = LABELS_DIR / "predictions.json"
LABEL_RUN_PATH = LABELS_DIR / "label_run.json"

GEMINI_MODEL = "gemini-3.6-flash"
GROQ_MODEL = "openai/gpt-oss-120b"
# NOTE: the spec called for gemini-2.0-flash / llama-3.3-70b-versatile, but
# both are retired on their respective platforms as of this run (verified
# live against the actual API keys) -- gemini-2.0-flash and gemini-2.5-flash
# both 404, and Groq has removed every Llama model from its active lineup.
# gemini-3.6-flash is Google's own suggested replacement; openai/gpt-oss-120b
# is the largest model Groq still serves, closest in spirit to a 70b fallback.

MAX_SAMPLE_WITHOUT_OVERRIDE = 100
MIN_CENTRAL = 5
MIN_STATE = 15

CATEGORIES = [
    "Agriculture, Rural & Environment",
    "Banking, Financial Services & Insurance",
    "Business & Entrepreneurship",
    "Education & Learning",
    "Health & Wellness",
    "Housing & Shelter",
    "Public Safety, Law & Justice",
    "Science, IT & Communications",
    "Skills & Employment",
    "Social Welfare & Empowerment",
    "Sports & Culture",
    "Transport & Infrastructure",
    "Travel & Tourism",
    "Utility & Sanitation",
    "Women & Child",
]
CATEGORIES_SET = set(CATEGORIES)
CONFIDENCE_LEVELS = {"high", "medium", "low"}

# Stratification buckets, checked in this order against a scheme's `tags`
# (lowercased substring match) -- first match wins. Anything matching none
# of these falls into the residual "Other" bucket. Calibrated against the
# actual tag vocabulary observed in data/interim/schemes/*.json.
STRATA_KEYWORDS = [
    ("Student/Scholarship", ["student", "scholarship", "fellowship", "stipend",
                              "internship", "education", "school", "tuition"]),
    ("Farmer/Agriculture", ["farmer", "agricultur", "farming", "horticultur",
                             "irrigation", "dairy", "livestock", "fish"]),
    ("Pension/Widow/Senior Citizen", ["pension", "widow", "senior citizen", "old age"]),
    ("Construction Worker/Labour", ["construction worker", "building worker",
                                     "labour", "labor", "mazdoor", "worker"]),
    ("Business/Loan/MSME/Entrepreneur", ["business", "loan", "msme",
                                          "entrepreneur", "startup", "industry"]),
    ("Disability", ["disability", "divyang", "differently abled", "person with disability"]),
    ("Health", ["health", "medical", "hospital", "maternity"]),
    ("Housing", ["housing", "house", "awas", "shelter"]),
    ("Women", ["women", "woman", "girl", "mahila", "marriage"]),
    ("Sports", ["sports", "sport", "athlete", "player"]),
]
STRATA_NAMES = [name for name, _ in STRATA_KEYWORDS] + ["Other"]

PROMPT_TEMPLATE = """You are classifying Indian government welfare schemes into the official myScheme category taxonomy.

Assign EXACTLY ONE category from this closed list. Do not invent categories.
Do not return anything outside this list.

1. Agriculture, Rural & Environment
2. Banking, Financial Services & Insurance
3. Business & Entrepreneurship
4. Education & Learning
5. Health & Wellness
6. Housing & Shelter
7. Public Safety, Law & Justice
8. Science, IT & Communications
9. Skills & Employment
10. Social Welfare & Empowerment
11. Sports & Culture
12. Transport & Infrastructure
13. Travel & Tourism
14. Utility & Sanitation
15. Women & Child

Classification rules:
- Classify by the NATURE OF THE BENEFIT first, and the TARGET BENEFICIARY
  second.
- Exception: if the scheme is explicitly constituted around women, children,
  or both, prefer "Women & Child". If constituted around SC/ST/OBC/PwD or
  other marginalised groups, prefer "Social Welfare & Empowerment".
- A scholarship or tuition benefit is "Education & Learning" even when the
  recipients are farmers' children or workers' dependents.
- Vocational or skill training aimed at employment is "Skills & Employment",
  not "Education & Learning".
- A cash transfer or subsidy is NOT automatically "Banking, Financial
  Services & Insurance" — use the domain the money is for. Reserve that
  category for schemes about banking access, credit, insurance products, or
  financial inclusion itself.
- Pensions and old-age, widow or disability support are "Social Welfare &
  Empowerment".

Scheme name: {scheme_name}
Description: {description}
Benefits: {benefits_text}

Respond with ONLY a JSON object, no markdown fences, no preamble:
{{
  "category": "<exact string from the list above>",
  "confidence": "high" | "medium" | "low",
  "reason": "<one sentence, max 25 words>",
  "runner_up": "<second-most-likely category, or null if unambiguous>"
}}
"""
# The template above is the classification prompt from the spec, embedded
# verbatim, with the trailing JSON schema's literal braces doubled so it
# survives str.format() -- .format() is only ever called with scheme_name /
# description / benefits_text, so the doubled braces render back to single
# braces in the actual prompt sent to the model, unchanged from the spec.


# ---------------------------------------------------------------------------
# Corpus loading / stratified sampling
# ---------------------------------------------------------------------------

def load_records():
    records = {}
    for path in sorted(SCHEMES_DIR.glob("*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        records[rec["slug"]] = rec
    return records


def bucket_for(record):
    tags_lower = [t.lower() for t in record.get("tags", [])]
    for name, keywords in STRATA_KEYWORDS:
        for tag in tags_lower:
            if any(kw in tag for kw in keywords):
                return name
    return "Other"


def allocate_counts(n, sizes, order):
    """Split n across buckets as evenly as possible, capped by each bucket's
    available size, redistributing any shortfall to buckets with room."""
    k = len(order)
    base, rem = divmod(n, k)
    alloc = {name: base for name in order}
    for name in order[:rem]:
        alloc[name] += 1
    for _ in range(1000):
        overflow = 0
        for name in order:
            if alloc[name] > sizes[name]:
                overflow += alloc[name] - sizes[name]
                alloc[name] = sizes[name]
        if overflow == 0:
            break
        capacity = [name for name in order if alloc[name] < sizes[name]]
        if not capacity:
            break
        i = 0
        while overflow > 0 and capacity:
            name = capacity[i % len(capacity)]
            if alloc[name] < sizes[name]:
                alloc[name] += 1
                overflow -= 1
            i += 1
    return alloc


def enforce_level_minimums(selected, records, rng):
    """Swap in central/state schemes as needed to satisfy the floor
    constraints, on top of (not instead of) the tag-based stratification."""
    selected = set(selected)
    all_slugs = list(records.keys())

    def level_count(level):
        return sum(1 for s in selected if records[s]["level"] == level)

    def swap_in(target_level, min_count, victim_level):
        need = min_count - level_count(target_level)
        if need <= 0:
            return
        candidates = [s for s in all_slugs
                      if records[s]["level"] == target_level and s not in selected]
        rng.shuffle(candidates)
        victims = [s for s in selected if records[s]["level"] == victim_level]
        rng.shuffle(victims)
        for i in range(min(need, len(candidates), len(victims))):
            selected.discard(victims[i])
            selected.add(candidates[i])

    swap_in("central", MIN_CENTRAL, "state")
    swap_in("state", MIN_STATE, "central")
    return selected


def stratified_sample(records, n, seed):
    buckets = collections.defaultdict(list)
    for slug, rec in records.items():
        buckets[bucket_for(rec)].append(slug)
    for name in STRATA_NAMES:
        buckets.setdefault(name, [])

    rng = random.Random(seed)
    alloc = allocate_counts(n, {name: len(buckets[name]) for name in STRATA_NAMES}, STRATA_NAMES)

    selected = []
    for name in STRATA_NAMES:
        pool = buckets[name][:]
        rng.shuffle(pool)
        selected.extend(pool[: alloc[name]])

    selected = enforce_level_minimums(selected, records, rng)

    bucket_counts = collections.Counter(bucket_for(records[s]) for s in selected)
    level_counts = collections.Counter(records[s]["level"] for s in selected)
    return sorted(selected), dict(bucket_counts), dict(level_counts)


# ---------------------------------------------------------------------------
# Prompt / validation
# ---------------------------------------------------------------------------

def truncate(s, n):
    return (s or "")[:n]


def build_prompt(record):
    return PROMPT_TEMPLATE.format(
        scheme_name=record["scheme_name"],
        description=truncate(record["description"], 1500),
        benefits_text=truncate(record["benefits_text"], 1500),
    )


FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def parse_and_validate(raw_text):
    text = FENCE_RE.sub("", raw_text.strip()).strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        return None, f"invalid_json: {e}"
    if not isinstance(obj, dict):
        return None, "response_not_a_json_object"
    missing = [k for k in ("category", "confidence", "reason", "runner_up") if k not in obj]
    if missing:
        return None, f"missing_keys: {missing}"
    if obj["category"] not in CATEGORIES_SET:
        return None, f"invalid_category: {obj['category']!r}"
    if obj["confidence"] not in CONFIDENCE_LEVELS:
        return None, f"invalid_confidence: {obj['confidence']!r}"
    if obj["runner_up"] is not None and obj["runner_up"] not in CATEGORIES_SET:
        return None, f"invalid_runner_up: {obj['runner_up']!r}"
    if not isinstance(obj["reason"], str) or not obj["reason"].strip():
        return None, "empty_reason"
    return obj, None


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------

def make_gemini_client():
    return genai.Client(
        api_key=os.environ["GEMINI_API_KEY"],
        http_options=genai_types.HttpOptions(
            timeout=30000,
            retry_options=genai_types.HttpRetryOptions(attempts=1),  # we drive retries ourselves
        ),
    )


def make_groq_client():
    return Groq(api_key=os.environ["GROQ_API_KEY"], timeout=30.0, max_retries=0)


def call_gemini(client, prompt):
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=genai_types.GenerateContentConfig(temperature=0),
    )
    return resp.text


def call_groq(client, prompt):
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return resp.choices[0].message.content


def classify_gemini_error(exc):
    """retryable: 429 / 5xx / timeout, per spec. Anything else is fatal --
    no retry, no fallback (e.g. auth or malformed-request errors, which a
    retry or a different provider wouldn't fix anyway)."""
    if isinstance(exc, genai_errors.APIError):
        code = getattr(exc, "code", None) or 0
        return "retryable" if (code == 429 or code >= 500) else "fatal"
    if isinstance(exc, httpx.TimeoutException):
        return "retryable"
    return "fatal"


def try_gemini(client, prompt):
    """Primary provider: up to 2 attempts (1 retry) with exponential backoff,
    for either a retryable transport error or a schema-validation failure."""
    raw, err = None, None
    for attempt in (1, 2):
        try:
            raw = call_gemini(client, prompt)
        except Exception as e:  # noqa: BLE001
            kind = classify_gemini_error(e)
            err = f"{kind}_error: {type(e).__name__}: {e}"
            if kind == "fatal":
                return {"ok": False, "raw": None, "parsed": None, "error": err, "allow_fallback": False}
            if attempt == 1:
                time.sleep(2 ** attempt)
                continue
            return {"ok": False, "raw": None, "parsed": None, "error": err, "allow_fallback": True}

        parsed, verr = parse_and_validate(raw)
        if parsed is not None:
            return {"ok": True, "raw": raw, "parsed": parsed, "error": None, "allow_fallback": False}
        err = f"schema_validation_failed: {verr}"
        if attempt == 1:
            time.sleep(1.0)
            continue
        return {"ok": False, "raw": raw, "parsed": None, "error": err, "allow_fallback": True}
    return {"ok": False, "raw": raw, "parsed": None, "error": err, "allow_fallback": True}


def try_groq(client, prompt):
    """Fallback provider: single attempt, per spec (no further fallback exists)."""
    try:
        raw = call_groq(client, prompt)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "raw": None, "parsed": None, "error": f"groq_error: {type(e).__name__}: {e}"}
    parsed, verr = parse_and_validate(raw)
    if parsed is None:
        return {"ok": False, "raw": raw, "parsed": None, "error": f"groq_schema_validation_failed: {verr}"}
    return {"ok": True, "raw": raw, "parsed": parsed, "error": None}


def classify_scheme(gemini_client, groq_client, record):
    prompt = build_prompt(record)
    t0 = time.monotonic()

    g = try_gemini(gemini_client, prompt)
    if g["ok"]:
        latency = time.monotonic() - t0
        return _prediction(record["slug"], g["parsed"], g["raw"], "gemini", GEMINI_MODEL, latency), None

    if not g["allow_fallback"]:
        latency = time.monotonic() - t0
        return None, {"slug": record["slug"], "error": g["error"], "latency": latency}

    gr = try_groq(groq_client, prompt)
    latency = time.monotonic() - t0
    if gr["ok"]:
        return _prediction(record["slug"], gr["parsed"], gr["raw"], "groq", GROQ_MODEL, latency), None

    return None, {
        "slug": record["slug"],
        "error": f"gemini: {g['error']} || groq: {gr['error']}",
        "latency": latency,
    }


def _prediction(slug, parsed, raw, provider, model, latency):
    return {
        "slug": slug,
        "category": parsed["category"],
        "confidence": parsed["confidence"],
        "reason": parsed["reason"],
        "runner_up": parsed["runner_up"],
        "provider": provider,
        "model": model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "raw_response": raw,
        "latency_seconds": round(latency, 3),
    }


# ---------------------------------------------------------------------------
# Output files
# ---------------------------------------------------------------------------

def write_to_verify_csv(path, slugs, records):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["slug", "scheme_name", "level", "state", "description",
                          "benefits", "human_category", "human_notes"])
        for slug in slugs:
            r = records[slug]
            writer.writerow([
                slug,
                r["scheme_name"],
                r["level"] or "",
                r["state"] or "",
                truncate(r["description"], 400),
                truncate(r["benefits_text"], 300),
                "",
                "",
            ])


def load_json(path, default):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return default
    return default


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", type=int, default=30, help="stratified sample size")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for sampling")
    ap.add_argument("--force", action="store_true",
                     help="reclassify slugs that already have a predictions.json entry")
    ap.add_argument("--delay", type=float, default=1.0,
                     help="seconds to sleep between API calls")
    ap.add_argument("--slug", type=str, default=None,
                     help="classify a single scheme and print the full request/response for debugging")
    ap.add_argument("--i-know-what-im-doing", action="store_true",
                     help="required to use --sample > 100")
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")
    if "GEMINI_API_KEY" not in os.environ or "GROQ_API_KEY" not in os.environ:
        print("GEMINI_API_KEY and GROQ_API_KEY must be set (see .env.example).", file=sys.stderr)
        sys.exit(1)
    logging.getLogger("google_genai.models").setLevel(logging.ERROR)  # silence benign AFC notice

    gemini_client = make_gemini_client()
    groq_client = make_groq_client()

    if args.slug:
        records = load_records()
        if args.slug not in records:
            print(f"no such slug: {args.slug}", file=sys.stderr)
            sys.exit(1)
        record = records[args.slug]
        prompt = build_prompt(record)
        print("=" * 70)
        print("REQUEST PROMPT")
        print("=" * 70)
        print(prompt)
        prediction, failure = classify_scheme(gemini_client, groq_client, record)
        print("=" * 70)
        print("RESULT")
        print("=" * 70)
        print(json.dumps(prediction or failure, indent=2, ensure_ascii=False))
        return

    if args.sample > MAX_SAMPLE_WITHOUT_OVERRIDE and not args.i_know_what_im_doing:
        print(f"--sample {args.sample} exceeds {MAX_SAMPLE_WITHOUT_OVERRIDE}; "
              f"pass --i-know-what-im-doing to proceed. This script must not run on the full corpus.",
              file=sys.stderr)
        sys.exit(1)

    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    records = load_records()

    # The sample and to_verify.csv are created once and then treated as
    # immutable, so a rerun never regenerates a different sample out from
    # under a human who has already started filling in to_verify.csv.
    # --force only affects whether already-predicted slugs get reclassified.
    if SAMPLE_SLUGS_PATH.exists():
        sample_meta = json.loads(SAMPLE_SLUGS_PATH.read_text(encoding="utf-8"))
        slugs = sample_meta["slugs"]
        print(f"Reusing existing sample from {SAMPLE_SLUGS_PATH} ({len(slugs)} slugs, "
              f"seed={sample_meta['seed']}). Delete this file to draw a new sample.")
    else:
        slugs, bucket_counts, level_counts = stratified_sample(records, args.sample, args.seed)
        sample_meta = {
            "created": datetime.now(timezone.utc).isoformat(),
            "requested_sample_size": args.sample,
            "actual_sample_size": len(slugs),
            "seed": args.seed,
            "bucket_counts": bucket_counts,
            "level_counts": level_counts,
            "slugs": slugs,
        }
        SAMPLE_SLUGS_PATH.write_text(json.dumps(sample_meta, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Drew a new stratified sample of {len(slugs)} slugs (seed={args.seed}) "
              f"-> {SAMPLE_SLUGS_PATH}")

    if TO_VERIFY_PATH.exists():
        print(f"{TO_VERIFY_PATH} already exists -- leaving it untouched so any human "
              f"annotations are preserved.")
    else:
        write_to_verify_csv(TO_VERIFY_PATH, slugs, records)
        print(f"Wrote blind verification sheet -> {TO_VERIFY_PATH}")

    predictions = load_json(PREDICTIONS_PATH, {})

    to_process = [s for s in slugs if args.force or s not in predictions]
    skipped = len(slugs) - len(to_process)

    provider_counts = collections.Counter()
    latencies = []
    failures = []

    for slug in tqdm(to_process, desc="classifying"):
        record = records[slug]
        prediction, failure = classify_scheme(gemini_client, groq_client, record)
        if prediction:
            predictions[slug] = prediction
            provider_counts[prediction["provider"]] += 1
            latencies.append(prediction["latency_seconds"])
        else:
            failures.append(failure["slug"])
            latencies.append(round(failure["latency"], 3))
            tqdm.write(f"FAILED {slug}: {failure['error']}")
        time.sleep(args.delay)

    PREDICTIONS_PATH.write_text(json.dumps(predictions, indent=2, ensure_ascii=False), encoding="utf-8")

    run_meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sample_size": len(slugs),
        "seed": sample_meta["seed"],
        "attempted_this_run": len(to_process),
        "skipped_already_labeled": skipped,
        "provider_counts": dict(provider_counts),
        "failure_count": len(failures),
        "failed_slugs": failures,
        "mean_latency_seconds": round(sum(latencies) / len(latencies), 3) if latencies else None,
        "total_labeled_so_far": len(predictions),
    }
    LABEL_RUN_PATH.write_text(json.dumps(run_meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print(f"Verification sheet: {TO_VERIFY_PATH}")
    print(f"Run metadata:       {LABEL_RUN_PATH}")
    print("Do NOT open predictions.json until human verification of to_verify.csv is complete.")


if __name__ == "__main__":
    main()

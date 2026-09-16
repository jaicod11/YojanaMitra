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
  - data/interim/labels/label_run.json     (run metadata + token accounting)

Token budget matters: the providers' free tiers cap daily tokens (Groq) and
daily requests (Gemini), so schemes are classified in batches that amortise
the shared instruction block, and --brief drops the per-label rationale
fields. Cumulative token usage is tracked from the API usage fields, and the
run stops cleanly when a provider reports its daily cap rather than retrying
into an exhausted quota.

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
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
import groq
from groq import Groq
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
SCHEMES_DIR = ROOT / "data/interim/schemes"
LABELS_DIR = ROOT / "data/interim/labels"
SAMPLE_SLUGS_PATH = LABELS_DIR / "sample_slugs.json"
TO_VERIFY_PATH = LABELS_DIR / "to_verify.csv"
PREDICTIONS_PATH = LABELS_DIR / "predictions.json"
LABEL_RUN_PATH = LABELS_DIR / "label_run.json"


def set_labels_dir(path):
    """Point every output at `path` (used by --out-dir so a config experiment
    can never write over the real labels or an archived verification round)."""
    global LABELS_DIR, SAMPLE_SLUGS_PATH, TO_VERIFY_PATH, PREDICTIONS_PATH, LABEL_RUN_PATH
    LABELS_DIR = Path(path)
    SAMPLE_SLUGS_PATH = LABELS_DIR / "sample_slugs.json"
    TO_VERIFY_PATH = LABELS_DIR / "to_verify.csv"
    PREDICTIONS_PATH = LABELS_DIR / "predictions.json"
    LABEL_RUN_PATH = LABELS_DIR / "label_run.json"

DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite"
GROQ_MODEL = "openai/gpt-oss-120b"
# Model history, all verified live against the real keys rather than assumed:
# the original spec's gemini-2.0-flash and llama-3.3-70b-versatile are both
# retired (2.0 and 2.5 flash now 404; Groq has dropped every Llama), so this
# ran on gemini-3.6-flash with openai/gpt-oss-120b as fallback. Probing the
# key then showed the -lite variants spend zero thinking tokens (7 vs 67 on
# an identical trivial prompt), which is the right trade for a closed-set
# classification that needs no internal reasoning -- hence the default below.
# Override with --gemini-model if lite quality proves worse.

MAX_SAMPLE_WITHOUT_OVERRIDE = 100
MIN_CENTRAL = 5
MIN_STATE = 15

# Free-tier daily caps, used only to project how far a day's budget stretches.
# The two providers are capped on different axes, so batching helps both but
# for different reasons: Groq is token-capped (bigger batches cut the shared
# instruction block per label), Gemini is request-capped (bigger batches mean
# more labels per request). Neither number is discoverable from the APIs --
# both come from observed behaviour: a previous run consumed 199,622 Groq
# tokens before Groq reported its daily cap.
GROQ_DAILY_TOKEN_QUOTA = 200_000
GEMINI_DAILY_REQUEST_QUOTA = 20

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

# The taxonomy + classification rules, sliced straight out of the verbatim
# template so the batch prompt and the single-scheme prompt always share
# byte-identical instructions (no second copy to drift out of sync).
INSTRUCTION_BLOCK = PROMPT_TEMPLATE.split("\nScheme name:")[0]

# Per-scheme context budget. Brief mode is the bulk default: it is the single
# biggest lever on tokens-per-label after batching.
FULL_DESC_CHARS, FULL_BENEFITS_CHARS = 1500, 1500
BRIEF_DESC_CHARS, BRIEF_BENEFITS_CHARS = 400, 200

# A slug that comes back missing/invalid this many times is retried on its
# own (batch of 1) to isolate it from whatever else confused the model.
SINGLETON_AFTER_ATTEMPTS = 2
MAX_ATTEMPTS_PER_SLUG = 3
# If every provider keeps failing whole calls, stop rather than grind through
# the corpus; the queue is left intact for the next run.
MAX_CONSECUTIVE_CALL_FAILURES = 6
# Long runs checkpoint to disk so an interruption costs at most this many labels.
CHECKPOINT_EVERY = 25


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
    """Single-scheme, full-detail prompt -- the spec's prompt, unchanged."""
    return PROMPT_TEMPLATE.format(
        scheme_name=record["scheme_name"],
        description=truncate(record["description"], FULL_DESC_CHARS),
        benefits_text=truncate(record["benefits_text"], FULL_BENEFITS_CHARS),
    )


def build_batch_prompt(records, brief):
    """One shared instruction block + N schemes -> one JSON array of results.

    The taxonomy and classification rules are identical to the single-scheme
    prompt; only the input framing and the response schema change, so labels
    stay comparable across batch sizes.
    """
    desc_n = BRIEF_DESC_CHARS if brief else FULL_DESC_CHARS
    ben_n = BRIEF_BENEFITS_CHARS if brief else FULL_BENEFITS_CHARS

    parts = [INSTRUCTION_BLOCK.rstrip(), "",
             f"You will be given {len(records)} schemes. Classify EACH ONE independently,",
             "applying the rules above to each scheme on its own.", ""]
    for i, r in enumerate(records, 1):
        parts += [
            f"--- Scheme {i} ---",
            f"slug: {r['slug']}",
            f"Scheme name: {r['scheme_name']}",
            f"Description: {truncate(r['description'], desc_n)}",
            f"Benefits: {truncate(r['benefits_text'], ben_n)}",
            "",
        ]
    parts += [
        f"Respond with ONLY a JSON array of {len(records)} objects, no markdown fences, "
        "no preamble. Include every slug exactly as given above, in the same order:",
    ]
    if brief:
        parts.append('[{"slug": "<slug>", "category": "<exact string from the list above>", '
                     '"confidence": "high" | "medium" | "low"}]')
    else:
        parts.append('[{"slug": "<slug>", "category": "<exact string from the list above>", '
                     '"confidence": "high" | "medium" | "low", '
                     '"reason": "<one sentence, max 25 words>", '
                     '"runner_up": "<second-most-likely category, or null if unambiguous>"}]')
    return "\n".join(parts)


FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def _strip_fences(raw_text):
    return FENCE_RE.sub("", (raw_text or "").strip()).strip()


def validate_fields(obj, brief):
    """Shared field validation for both the single and batched schemas."""
    required = ["category", "confidence"] if brief else ["category", "confidence", "reason", "runner_up"]
    missing = [k for k in required if k not in obj]
    if missing:
        return f"missing_keys: {missing}"
    if obj["category"] not in CATEGORIES_SET:
        return f"invalid_category: {obj['category']!r}"
    if obj["confidence"] not in CONFIDENCE_LEVELS:
        return f"invalid_confidence: {obj['confidence']!r}"
    if not brief:
        if obj["runner_up"] is not None and obj["runner_up"] not in CATEGORIES_SET:
            return f"invalid_runner_up: {obj['runner_up']!r}"
        if not isinstance(obj["reason"], str) or not obj["reason"].strip():
            return "empty_reason"
    return None


def parse_and_validate(raw_text, brief=False):
    text = _strip_fences(raw_text)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        return None, f"invalid_json: {e}"
    if not isinstance(obj, dict):
        return None, "response_not_a_json_object"
    err = validate_fields(obj, brief)
    return (None, err) if err else (obj, None)


def parse_batch_response(raw_text, requested_slugs, brief):
    """Return (results_by_slug, per_slug_errors, fatal_error).

    Every requested slug is accounted for: it either lands in results or in
    per_slug_errors, so the caller can re-queue the stragglers instead of
    silently dropping them.
    """
    text = _strip_fences(raw_text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as e:
        return {}, {}, f"invalid_json: {e}"
    if isinstance(payload, dict):  # tolerate {"results": [...]} shapes
        for key in ("results", "schemes", "classifications", "data"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
    if not isinstance(payload, list):
        return {}, {}, "response_not_a_json_array"

    by_slug, errors = {}, {}
    seen = {}
    for item in payload:
        if isinstance(item, dict) and isinstance(item.get("slug"), str):
            seen[item["slug"].strip()] = item

    for slug in requested_slugs:
        obj = seen.get(slug)
        if obj is None:
            errors[slug] = "missing_from_response"
            continue
        err = validate_fields(obj, brief)
        if err:
            errors[slug] = err
        else:
            by_slug[slug] = obj
    return by_slug, errors, None


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------

def collect_keys(prefix):
    """GEMINI_API_KEY, GEMINI_API_KEY_2, ... _9 (gaps tolerated)."""
    keys = []
    primary = (os.environ.get(prefix) or "").strip()
    if primary:
        keys.append((prefix, primary))
    for i in range(2, 10):
        name = f"{prefix}_{i}"
        val = (os.environ.get(name) or "").strip()
        if val:
            keys.append((name, val))
    return keys


# A 429 that names a per-minute window recovers in seconds and is worth
# waiting out; one that names a per-day window does not, and retrying into it
# is what burned 8 hours last run.
DAILY_QUOTA_RE = re.compile(r"per\s*-?\s*day|perday|\bTPD\b|\bRPD\b|daily", re.IGNORECASE)
MINUTE_QUOTA_RE = re.compile(r"per\s*-?\s*minute|perminute|\bTPM\b|\bRPM\b", re.IGNORECASE)
MAX_CONSECUTIVE_429 = 3


class ProviderPool:
    """One provider, one or more API keys, with usage + exhaustion tracking."""

    def __init__(self, name, model, key_prefix, client_factory, brief=True, batch_size=10):
        self.name = name
        self.model = model
        self.keys = collect_keys(key_prefix)
        self.client_factory = client_factory
        # Prompt mode and batch size are per-provider: the two models degrade
        # differently under batching, so each runs in the shape it was
        # validated in rather than a single global setting.
        self.brief = brief
        self.batch_size = batch_size
        self.cooldown = 0  # skip this pool for N selections after a non-quota failure
        self.idx = 0
        self._clients = {}
        self.exhausted_keys = []
        self.usage = collections.Counter()  # prompt / completion / total / requests
        self.per_key_usage = collections.defaultdict(collections.Counter)
        self.consecutive_429 = 0

    @property
    def available(self):
        return self.idx < len(self.keys)

    @property
    def key_name(self):
        return self.keys[self.idx][0] if self.available else None

    def client(self):
        name, key = self.keys[self.idx]
        if name not in self._clients:
            self._clients[name] = self.client_factory(key)
        return self._clients[name]

    def record_usage(self, prompt_t, completion_t, total_t):
        key = self.key_name or "exhausted"
        for counter in (self.usage, self.per_key_usage[key]):
            counter["prompt"] += prompt_t
            counter["completion"] += completion_t
            counter["total"] += total_t
            counter["requests"] += 1

    def retire_current_key(self, reason):
        """Daily cap on this key -> rotate to the next one, if any."""
        if self.available:
            self.exhausted_keys.append({"key": self.key_name, "reason": reason})
            self.idx += 1
            self.consecutive_429 = 0
        return self.available

    def disable(self, reason):
        """Take the whole provider out of this run (misconfiguration, not quota)."""
        while self.available:
            self.retire_current_key(reason)


def make_gemini_client(api_key):
    return genai.Client(
        api_key=api_key,
        http_options=genai_types.HttpOptions(
            timeout=60000,
            retry_options=genai_types.HttpRetryOptions(attempts=1),  # we drive retries ourselves
        ),
    )


def make_groq_client(api_key):
    return Groq(api_key=api_key, timeout=60.0, max_retries=0)


def call_gemini(client, prompt, model):
    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=genai_types.GenerateContentConfig(temperature=0),
    )
    um = resp.usage_metadata
    prompt_t = um.prompt_token_count or 0
    completion_t = (um.candidates_token_count or 0) + (getattr(um, "thoughts_token_count", None) or 0)
    total_t = um.total_token_count or (prompt_t + completion_t)
    return resp.text, (prompt_t, completion_t, total_t)


def call_groq(client, prompt, model, reasoning_effort=None):
    kwargs = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        **kwargs,
    )
    u = resp.usage
    return resp.choices[0].message.content, (u.prompt_tokens or 0, u.completion_tokens or 0, u.total_tokens or 0)


def error_status_and_message(exc):
    """(http_status_or_None, message) for either SDK's error types."""
    if isinstance(exc, genai_errors.APIError):
        return getattr(exc, "code", None), f"{getattr(exc, 'message', '')} {getattr(exc, 'details', '')}"
    status = getattr(exc, "status_code", None)
    body = getattr(exc, "body", None) or getattr(exc, "message", None) or str(exc)
    return status, str(body)


def classify_error(exc):
    """-> 'daily_quota' | 'rate_minute' | 'retryable' | 'fatal'.

    'fatal' is reserved for a definite non-429 HTTP status (auth, bad request,
    unknown model) -- things a retry or a different key cannot fix. Everything
    at the transport layer (connection reset, timeout, DNS) is transient and
    must stay retryable: treating a dropped connection as fatal takes a
    perfectly healthy provider out of the run for hours.
    """
    if isinstance(exc, (httpx.TransportError, groq.APIConnectionError)):
        return "retryable"  # covers timeouts, connection resets, protocol errors
    status, message = error_status_and_message(exc)
    if status == 429:
        if MINUTE_QUOTA_RE.search(message):
            return "rate_minute"
        if DAILY_QUOTA_RE.search(message):
            return "daily_quota"
        return "rate_minute"  # unlabelled 429: back off, but see MAX_CONSECUTIVE_429
    if status is not None and status >= 500:
        return "retryable"
    if status is not None:
        return "fatal"
    return "retryable"  # no status at all: assume transient, bounded by retries


def pool_generate(pool, prompt, delay, reasoning_effort=None):
    """One provider attempt, including its own retries.

    Returns {"ok", "text", "error", "status"} where status is one of
    'ok' | 'exhausted' | 'fatal' | 'failed'. 'exhausted' means every key for
    this provider has reported its daily cap.
    """
    attempt = 0
    while pool.available:
        attempt += 1
        try:
            if pool.name == "gemini":
                text, usage = call_gemini(pool.client(), prompt, pool.model)
            else:
                text, usage = call_groq(pool.client(), prompt, pool.model, reasoning_effort)
        except Exception as e:  # noqa: BLE001
            kind = classify_error(e)
            _, message = error_status_and_message(e)
            short = f"{type(e).__name__}: {message[:200]}"

            if kind == "daily_quota":
                tqdm.write(f"  [{pool.name}] daily cap reported on {pool.key_name}: {message[:160]}")
                if not pool.retire_current_key("daily_quota"):
                    return {"ok": False, "text": None, "error": short, "status": "exhausted"}
                tqdm.write(f"  [{pool.name}] rotating to key {pool.key_name}")
                attempt = 0
                continue

            if kind == "rate_minute":
                pool.consecutive_429 += 1
                if pool.consecutive_429 >= MAX_CONSECUTIVE_429:
                    tqdm.write(f"  [{pool.name}] {MAX_CONSECUTIVE_429} consecutive 429s on "
                               f"{pool.key_name}; treating as exhausted")
                    if not pool.retire_current_key("repeated_429"):
                        return {"ok": False, "text": None, "error": short, "status": "exhausted"}
                    attempt = 0
                    continue
                if attempt <= 2:
                    wait = max(delay, 20.0 * attempt)
                    tqdm.write(f"  [{pool.name}] per-minute rate limit; waiting {wait:.0f}s")
                    time.sleep(wait)
                    continue
                return {"ok": False, "text": None, "error": short, "status": "failed"}

            if kind == "retryable":
                if attempt <= 2:
                    time.sleep(2 ** attempt)
                    continue
                return {"ok": False, "text": None, "error": short, "status": "failed"}

            return {"ok": False, "text": None, "error": short, "status": "fatal"}

        pool.record_usage(*usage)
        pool.consecutive_429 = 0
        return {"ok": True, "text": text, "error": None, "status": "ok"}

    return {"ok": False, "text": None, "error": "no keys available", "status": "exhausted"}


def classify_with_pool(pool, records, slugs, delay, reasoning_effort=None):
    """Classify a batch of slugs with one specific provider, in that
    provider's own prompt mode.

    Returns (results_by_slug, errors_by_slug, status) where status is
    'ok' | 'invalid' | 'failed' | 'fatal' | 'exhausted'. Provider selection
    and fallback are the caller's job, because each provider batches
    differently and a re-queued slug must be re-chunked for whoever picks
    it up next.
    """
    # A single scheme in full-detail mode uses the spec's original
    # single-scheme prompt and schema, so anything labeled that way stays
    # directly comparable with the verification sample.
    single_full = len(slugs) == 1 and not pool.brief
    prompt = build_prompt(records[slugs[0]]) if single_full \
        else build_batch_prompt([records[s] for s in slugs], pool.brief)

    res = pool_generate(pool, prompt, delay,
                        reasoning_effort if pool.name == "groq" else None)
    if res["status"] != "ok":
        return {}, {s: res["error"] for s in slugs}, res["status"]

    if single_full:
        obj, verr = parse_and_validate(res["text"], brief=False)
        if obj is not None:
            return {slugs[0]: obj}, {}, "ok"
        return {}, {slugs[0]: verr}, "invalid"

    by_slug, errors, fatal = parse_batch_response(res["text"], slugs, pool.brief)
    if fatal:
        return {}, {s: fatal for s in slugs}, "invalid"
    if not by_slug:
        return {}, errors or {s: "empty_response" for s in slugs}, "invalid"
    return by_slug, errors, "ok"


def select_pool(pools):
    """First available provider, skipping any cooling off after a failure so
    the other provider gets a turn (this is the fallback path now that each
    provider has its own batch shape)."""
    available = [p for p in pools if p.available]
    if not available:
        return None
    ready = [p for p in available if p.cooldown == 0]
    return ready[0] if ready else available[0]


def _prediction(slug, parsed, raw, provider, model, latency, mode, batch_size):
    return {
        "slug": slug,
        "category": parsed["category"],
        "confidence": parsed["confidence"],
        "reason": parsed.get("reason"),
        "runner_up": parsed.get("runner_up"),
        "provider": provider,
        "model": model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "raw_response": raw,
        "latency_seconds": round(latency, 3),
        "mode": mode,
        "batch_size": batch_size,
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
    ap.add_argument("--batch", type=int, default=10,
                     help="schemes per API call; amortises the shared instruction block")
    ap.add_argument("--limit", type=int, default=None,
                     help="process at most N still-unlabeled schemes this run")
    ap.add_argument("--brief", action=argparse.BooleanOptionalAction, default=None,
                     help="short context, and ask only for slug/category/confidence "
                          "(default: on for bulk runs, off for --slug)")
    ap.add_argument("--groq-reasoning-effort", choices=["low", "medium", "high"], default=None,
                     help="lower effort cuts Groq completion tokens substantially")
    ap.add_argument("--provider", choices=["gemini", "groq"], default=None,
                     help="restrict to one provider (e.g. drive Groq directly once "
                          "Gemini's daily request quota is spent)")
    ap.add_argument("--gemini-model", default=DEFAULT_GEMINI_MODEL,
                     help=f"Gemini model id (default: {DEFAULT_GEMINI_MODEL})")
    ap.add_argument("--gemini-batch", type=int, default=None,
                     help="schemes per Gemini call (default: --batch)")
    ap.add_argument("--groq-batch", type=int, default=None,
                     help="schemes per Groq call; 1 = unbatched (default: --batch)")
    ap.add_argument("--gemini-brief", action=argparse.BooleanOptionalAction, default=None,
                     help="prompt mode for Gemini (default: --brief)")
    ap.add_argument("--groq-brief", action=argparse.BooleanOptionalAction, default=None,
                     help="prompt mode for Groq (default: --brief)")
    ap.add_argument("--out-dir", type=Path, default=None,
                     help="write sample/predictions/run metadata to this directory instead of "
                          "data/interim/labels (for side-by-side config evaluation)")
    args = ap.parse_args()

    if args.out_dir:
        set_labels_dir(args.out_dir)

    load_dotenv(ROOT / ".env")
    logging.getLogger("google_genai.models").setLevel(logging.ERROR)  # silence benign AFC notice

    # Brief mode is the bulk default; the debug path stays full-detail so it
    # still exercises the same prompt the verification sample was labeled with.
    brief = args.brief if args.brief is not None else (args.slug is None)

    pick = lambda specific, fallback: fallback if specific is None else specific
    pools = [
        ProviderPool("gemini", args.gemini_model, "GEMINI_API_KEY", make_gemini_client,
                     brief=pick(args.gemini_brief, brief),
                     batch_size=max(1, pick(args.gemini_batch, args.batch))),
        ProviderPool("groq", GROQ_MODEL, "GROQ_API_KEY", make_groq_client,
                     brief=pick(args.groq_brief, brief),
                     batch_size=max(1, pick(args.groq_batch, args.batch))),
    ]
    if args.provider:
        pools = [p for p in pools if p.name == args.provider]
    for pool in pools:
        if not pool.keys:
            print(f"No API keys found for {pool.name} (see .env.example).", file=sys.stderr)
            sys.exit(1)
    print("Providers (tried in order):")
    for p in pools:
        print(f"  {p.name:7s} {p.model:24s} mode={'brief' if p.brief else 'full ':5s} "
              f"batch={p.batch_size:<3d} keys={', '.join(n for n, _ in p.keys)}")

    if args.slug:
        records = load_records()
        if args.slug not in records:
            print(f"no such slug: {args.slug}", file=sys.stderr)
            sys.exit(1)
        pool = pools[0]
        prompt = (build_batch_prompt([records[args.slug]], pool.brief) if pool.brief
                  else build_prompt(records[args.slug]))
        print("=" * 70)
        print(f"REQUEST PROMPT  ({pool.name} / {pool.model}, mode={'brief' if pool.brief else 'full'})")
        print("=" * 70)
        print(prompt)
        t0 = time.monotonic()
        by_slug, errors, status = classify_with_pool(
            pool, records, [args.slug], args.delay, args.groq_reasoning_effort)
        print("=" * 70)
        print("RESULT")
        print("=" * 70)
        if args.slug in by_slug:
            obj = by_slug[args.slug]
            print(json.dumps(_prediction(args.slug, obj, json.dumps(obj), pool.name, pool.model,
                                          time.monotonic() - t0,
                                          "brief" if pool.brief else "full", 1),
                              indent=2, ensure_ascii=False))
        else:
            print(json.dumps({"slug": args.slug, "status": status,
                               "error": errors.get(args.slug)}, indent=2, ensure_ascii=False))
        for p in pools:
            if p.usage["requests"]:
                print(f"  {p.name}: {p.usage['requests']} request(s), {p.usage['total']} tokens")
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

    pending = [s for s in slugs if args.force or s not in predictions]
    already_labeled = len(slugs) - len(pending)
    if args.limit is not None:
        pending = pending[: args.limit]
    print(f"{already_labeled} already labeled, {len(pending)} queued this run "
          f"({len(pools)} provider(s) configured above).")

    queue = deque(pending)
    attempts = collections.Counter()
    provider_counts = collections.Counter()
    latencies = []
    failures = {}
    stopped_reason = "completed"

    provenance = collections.Counter()
    consecutive_call_failures = 0
    since_checkpoint = 0
    interrupted = False

    def save_predictions():
        tmp = PREDICTIONS_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(predictions, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(PREDICTIONS_PATH)  # atomic, so a crash can't truncate the file

    try:
        with tqdm(total=len(pending), desc="labeling", unit="label") as bar:
            while queue:
                pool = select_pool(pools)
                if pool is None:
                    reasons = {k["reason"] for p in pools for k in p.exhausted_keys}
                    stopped_reason = ("quota_exhausted"
                                      if reasons and reasons <= {"daily_quota", "repeated_429"}
                                      else "providers_unavailable")
                    break
                for p in pools:  # cooldowns tick down once per selection round
                    if p is not pool and p.cooldown:
                        p.cooldown -= 1

                # A slug that has already come back missing/invalid twice gets a
                # call to itself, so one bad scheme can't keep poisoning a batch.
                size = 1 if attempts[queue[0]] >= SINGLETON_AFTER_ATTEMPTS else pool.batch_size
                batch = [queue.popleft() for _ in range(min(size, len(queue)))]

                t0 = time.monotonic()
                by_slug, errors, status = classify_with_pool(
                    pool, records, batch, args.delay, args.groq_reasoning_effort)
                elapsed = time.monotonic() - t0

                if status == "exhausted":
                    queue.extendleft(reversed(batch))  # nothing consumed; keep them queued
                    tqdm.write(f"  [{pool.name}] out of quota; handing off to the next provider")
                    continue  # another provider may still be available

                if status != "ok":
                    # The whole call failed, so this is the provider's fault, not
                    # the schemes' -- re-queue them without spending their retry
                    # budget, and bench the provider so the other one gets a turn.
                    # (Burning attempts here would mark perfectly good schemes as
                    # failed just because one provider was misbehaving.)
                    sample_error = next(iter(errors.values()), "unknown")
                    if status == "fatal":
                        pool.disable("fatal_error")
                        tqdm.write(f"  [{pool.name}] disabled for this run: {sample_error}")
                    else:
                        pool.cooldown = 2
                        tqdm.write(f"  [{pool.name}] call failed ({sample_error}); benching it briefly")
                    queue.extendleft(reversed(batch))
                    consecutive_call_failures += 1
                    if consecutive_call_failures >= MAX_CONSECUTIVE_CALL_FAILURES:
                        stopped_reason = "repeated_provider_failures"
                        break
                    continue
                consecutive_call_failures = 0

                mode = "brief" if pool.brief else "full"
                per_label_latency = round(elapsed / max(1, len(batch)), 3)
                for slug in batch:
                    if slug in by_slug:
                        predictions[slug] = _prediction(
                            slug, by_slug[slug], json.dumps(by_slug[slug], ensure_ascii=False),
                            pool.name, pool.model, per_label_latency, mode, len(batch))
                        provider_counts[pool.name] += 1
                        provenance[(pool.name, pool.model, mode, len(batch))] += 1
                        latencies.append(per_label_latency)
                        failures.pop(slug, None)
                        bar.update(1)
                    else:
                        attempts[slug] += 1
                        reason = errors.get(slug, "unknown")
                        if attempts[slug] >= MAX_ATTEMPTS_PER_SLUG:
                            failures[slug] = reason
                            bar.update(1)
                            tqdm.write(f"  giving up on {slug} after {attempts[slug]} attempt(s): {reason}")
                        else:
                            queue.append(slug)  # re-queue rather than drop

                since_checkpoint += len(by_slug)
                if since_checkpoint >= CHECKPOINT_EVERY:
                    save_predictions()
                    since_checkpoint = 0

                bar.set_postfix_str(" ".join(
                    f"{p.name}={p.usage['total'] / 1000:.1f}k" for p in pools if p.usage["requests"]))
                if queue:
                    time.sleep(args.delay)

    except KeyboardInterrupt:
        interrupted = True
        stopped_reason = "interrupted"
        print("\n  interrupted -- saving everything labeled so far")

    save_predictions()

    labels_this_run = sum(provider_counts.values())
    total_tokens = sum(p.usage["total"] for p in pools)
    token_usage = {
        p.name: {
            "model": p.model,
            "requests": p.usage["requests"],
            "prompt_tokens": p.usage["prompt"],
            "completion_tokens": p.usage["completion"],
            "total_tokens": p.usage["total"],
            "labels": provider_counts.get(p.name, 0),
            "tokens_per_label": round(p.usage["total"] / provider_counts[p.name], 1)
            if provider_counts.get(p.name) else None,
            "keys_configured": [n for n, _ in p.keys],
            "keys_exhausted": p.exhausted_keys,
            "per_key_tokens": {k: dict(v) for k, v in p.per_key_usage.items()},
        }
        for p in pools
    }
    tokens_per_label = round(total_tokens / labels_this_run, 1) if labels_this_run else None
    groq_tpl = token_usage.get("groq", {}).get("tokens_per_label")

    # Each provider is rationed on a different axis, so project each on its own:
    # Groq by tokens/day, Gemini by requests/day x schemes per request.
    projection = {}
    if groq_tpl:
        projection["groq_labels_per_day"] = int(GROQ_DAILY_TOKEN_QUOTA / groq_tpl)
        projection["groq_basis"] = (f"{GROQ_DAILY_TOKEN_QUOTA:,} tokens/day / {groq_tpl} tokens per label")
    if "gemini" in token_usage and token_usage["gemini"]["requests"]:
        gem = token_usage["gemini"]
        labels_per_req = gem["labels"] / gem["requests"]
        hit_cap = any(k["reason"] in ("daily_quota", "repeated_429") for k in gem["keys_exhausted"])
        if hit_cap:
            projection["gemini_labels_per_day"] = int(GEMINI_DAILY_REQUEST_QUOTA * labels_per_req)
            projection["gemini_basis"] = (f"{GEMINI_DAILY_REQUEST_QUOTA} requests/day x "
                                           f"{labels_per_req:.1f} labels per request")
        else:
            # Never hit a cap, so any per-day figure would be invented. Report
            # the observed floor instead.
            projection["gemini_basis"] = (f"no daily cap observed: {gem['requests']} requests / "
                                           f"{gem['labels']} labels / {gem['total_tokens']} tokens "
                                           f"this run without one")
    if "groq_labels_per_day" in projection and "gemini_labels_per_day" in projection:
        projection["combined_labels_per_day"] = (projection["groq_labels_per_day"]
                                                  + projection["gemini_labels_per_day"])

    run_meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sample_size": len(slugs),
        "seed": sample_meta["seed"],
        "provider_config": {p.name: {"model": p.model, "mode": "brief" if p.brief else "full",
                                      "batch_size": p.batch_size} for p in pools},
        "labels_by_provenance": {" / ".join(map(str, k)): v for k, v in sorted(provenance.items())},
        "groq_reasoning_effort": args.groq_reasoning_effort,
        "queued_this_run": len(pending),
        "skipped_already_labeled": already_labeled,
        "labels_produced_this_run": labels_this_run,
        "provider_counts": dict(provider_counts),
        "token_usage": token_usage,
        "total_tokens": total_tokens,
        "tokens_per_label": tokens_per_label,
        "daily_projection": projection,
        "stopped_reason": stopped_reason,
        "requeued_still_pending": list(queue),
        "failure_count": len(failures),
        "failed_slugs": sorted(failures),
        "failure_reasons": failures,
        "mean_latency_seconds": round(sum(latencies) / len(latencies), 3) if latencies else None,
        "total_labeled_so_far": len(predictions),
    }
    LABEL_RUN_PATH.write_text(json.dumps(run_meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print("=" * 62)
    if stopped_reason == "quota_exhausted":
        print("STOPPED EARLY: every provider reported its daily cap.")
    elif stopped_reason == "providers_unavailable":
        print("STOPPED EARLY: every provider became unavailable (see keys_exhausted "
              "in label_run.json for why -- this is NOT necessarily a quota cap).")
    elif stopped_reason == "repeated_provider_failures":
        print("STOPPED EARLY: too many consecutive failed calls across all providers.")
    print(f"labels produced this run : {labels_this_run}")
    print(f"total labeled so far     : {len(predictions)} / {len(slugs)}")
    print(f"failed (gave up)         : {len(failures)}")
    print(f"still queued             : {len(queue)}")
    for name, u in token_usage.items():
        if u["requests"]:
            print(f"  {name:6s} {u['requests']:4d} req  {u['total_tokens']:7d} tok  "
                  f"{u['labels']:4d} labels  {u['tokens_per_label']} tok/label")
    print(f"total tokens             : {total_tokens}")
    print(f"tokens per label         : {tokens_per_label}")
    if projection:
        print("projected labels/day (free tier):")
        for key in ("groq", "gemini"):
            if f"{key}_labels_per_day" in projection:
                print(f"  {key:6s} {projection[f'{key}_labels_per_day']:5d}   "
                      f"({projection[f'{key}_basis']})")
        if "combined_labels_per_day" in projection:
            print(f"  {'both':6s} {projection['combined_labels_per_day']:5d}")
    print("=" * 62)
    print(f"Verification sheet: {TO_VERIFY_PATH}")
    print(f"Run metadata:       {LABEL_RUN_PATH}")
    print("Do NOT open predictions.json until human verification of to_verify.csv is complete.")


if __name__ == "__main__":
    main()

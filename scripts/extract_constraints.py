#!/usr/bin/env python3
"""Turn eligibility prose into typed constraints that deterministic code can evaluate.

The LLM never decides eligibility -- it only copies values that are explicitly
stated, and every typed value carries the exact substring it came from so a
claim can be traced back to a clause. Anything that does not fit a typed field
is preserved verbatim in residual_conditions rather than being forced into the
wrong field or dropped.

Writes one record per scheme to data/interim/constraints/<slug>.json and never
touches data/interim/schemes/. Provider chain, key rotation, quota-aware stop
and resumability are reused from label_categories.py. Unlike classification
this is NOT batched: the output is long and batching measurably hurt accuracy
there, so each scheme gets its own call with its full eligibility_text.

Runs are capped at --limit (default 30) unless --all is passed; the run stops
cleanly on a daily cap and the next run picks up where it left off.

--add-boolean-fields is a migration pass for records extracted before the four
true|null fields (BOOLEAN_FIELDS) existed: it asks for just those four, with
the same provenance validation and retry-then-fail handling, and drops any
residual condition they now capture. Other fields are left as they are.
"""
import argparse
import collections
import json
import logging
import os
import re
import signal
import statistics
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

import label_categories as lc  # provider pools, retry/backoff, quota handling

ROOT = lc.ROOT
SCHEMES_DIR = ROOT / "data/interim/schemes"
CONSTRAINTS_DIR = ROOT / "data/interim/constraints"
REPORT_FILENAME = "extract_report.json"
BOOLEAN_REPORT_FILENAME = "boolean_fields_report.json"

MAX_ATTEMPTS_PER_SLUG = 2  # one extraction + one retry, then recorded as failed
MAX_CONSECUTIVE_CALL_FAILURES = 6
# Rewrite the report every N settled schemes, so a run killed outright
# (SIGKILL, power loss) still leaves one that is at most N schemes stale.
CHECKPOINT_EVERY = 50

# The migration pass only deletes a residual when the source spans of the
# fields said to replace it account for nearly all of its words. That catches
# a residual attributed to the wrong field, and one that also states something
# else ("... and a resident of Kerala"); either way it is kept, not dropped.
RESIDUAL_COVERAGE_MIN = 0.85

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

GENDERS = {"male", "female", "transgender", "any"}
SOCIAL_CATEGORIES = {"SC", "ST", "OBC", "EWS", "General", "Minority", "PwD"}
RESIDENCES = {"rural", "urban", "any"}
INCOME_BASIS = {"family", "individual"}

# field -> allowed sub-keys (None = scalar/list field)
SCHEMA = {
    "age": {"min", "max"},
    "income": {"max_annual_inr", "basis"},
    "gender": None,
    "category": None,
    "occupation": None,
    "land": {"max_acres", "min_acres"},
    "education": {"stage", "min_class", "max_class"},
    "state": None,
    "residence": None,
    "marital_status": None,
    "family": {"requires_dependent", "notes"},
    "citizen_of_india": None,
    "bpl_household": None,
    "requires_bocw_registration": None,
    "not_availing_other_scheme": None,
    "residual_conditions": None,
}
TYPED_FIELDS = [f for f in SCHEMA if f != "residual_conditions"]
# true when the text requires it, null otherwise -- never false (see prompt)
BOOLEAN_FIELDS = ["citizen_of_india", "bpl_household",
                  "requires_bocw_registration", "not_availing_other_scheme"]

# "any" asserts that the scheme does NOT restrict on this axis. It constrains
# nobody, so it needs no source clause and should not count toward fill rate --
# otherwise the report overstates how many schemes actually gate on gender.
NON_CONSTRAINING_VALUES = {"gender": {"any"}, "residence": {"any"}}

EMPTY_CONSTRAINTS = {
    "age": {"min": None, "max": None},
    "income": {"max_annual_inr": None, "basis": None},
    "gender": None,
    "category": None,
    "occupation": None,
    "land": {"max_acres": None, "min_acres": None},
    "education": {"stage": None, "min_class": None, "max_class": None},
    "state": None,
    "residence": None,
    "marital_status": None,
    "family": {"requires_dependent": None, "notes": None},
    "citizen_of_india": None,
    "bpl_household": None,
    "requires_bocw_registration": None,
    "not_availing_other_scheme": None,
    "residual_conditions": [],
}

# Shared by the full prompt and the migration prompt so both apply one definition.
BOOLEAN_FIELD_RULES = """The four fields citizen_of_india, bpl_household, requires_bocw_registration and not_availing_other_scheme are true or null -- never false. true means the text REQUIRES the condition of every eligible applicant. Everything else is null: not mentioned, only a preference or priority ("preference will be given to BPL families"), one alternative among several ("BPL families or income below Rs 1 lakh"), an exemption, or the opposite ("Indian nationals are not eligible").
- citizen_of_india: must be a citizen or national of India. Residence ("resident of India", "domicile of the State") is not citizenship.
- bpl_household: must belong to a Below Poverty Line (BPL) household or hold a BPL card. A requirement that names only another card (Antyodaya, yellow ration card) without saying BPL is null.
- requires_bocw_registration: must be registered with a Building and Other Construction Workers (BOCW) welfare board. "Construction workers welfare board", "registered construction worker" and a state board's own name are the same condition. Working in construction without a registration requirement is null, and so is "registered or unregistered". Registration with "the Board" counts only when the text shows it is a construction workers board; boards for unorganised, domestic or other workers are not this condition.
- not_availing_other_scheme: must not be receiving, or have received, benefits for the same purpose under another scheme, scholarship, pension or programme. A bar on claiming THIS scheme a second time is not this condition, and neither is being employed, drawing a salary, or drawing a service or family pension.
A condition required of the family member through whom the applicant qualifies ("child of a registered construction worker", "the deceased must have been a citizen of India") still makes the field true: each field covers the applicant or that family member."""

PROMPT_TEMPLATE = """You extract structured eligibility constraints from Indian government welfare scheme text.

You do NOT decide whether anyone is eligible. You only copy values that are explicitly stated. Deterministic code does the comparisons later, so a wrong or invented value is far worse than a null.

Return ONLY a JSON object with exactly these three top-level keys: "constraints", "provenance", "parse_flags".

"constraints" must contain exactly these keys and no others:
{
  "age":          {"min": int|null, "max": int|null},
  "income":       {"max_annual_inr": int|null, "basis": "family"|"individual"|null},
  "gender":       "male"|"female"|"transgender"|"any"|null,
  "category":     ["SC","ST","OBC","EWS","General","Minority","PwD"] | null,
  "occupation":   [string] | null,
  "land":         {"max_acres": float|null, "min_acres": float|null},
  "education":    {"stage": string|null, "min_class": int|null, "max_class": int|null},
  "state":        string|null,
  "residence":    "rural"|"urban"|"any"|null,
  "marital_status": string|null,
  "family":       {"requires_dependent": bool|null, "notes": string|null},
  "citizen_of_india":           true|null,
  "bpl_household":              true|null,
  "requires_bocw_registration": true|null,
  "not_availing_other_scheme":  true|null,
  "residual_conditions": [string]
}

Rules:
- Extract ONLY what is stated. If a condition is absent, the field is null. Never guess.
- Never infer a threshold that is not written. "small farmer" with no acreage given means land stays null and "small farmer" goes to residual_conditions.
- Normalise units: lakh and crore to integer rupees (1 lakh = 100000, 1 crore = 10000000); hectares to acres (1 hectare = 2.47105 acres); "Class X" / "10th standard" to min_class / max_class integers.
- residual_conditions is load-bearing. EVERY eligibility condition that does not map cleanly to a typed field goes there VERBATIM as a short string, copied from the text. Never force a condition into the wrong typed field, and never silently drop one. A scheme stating 8 conditions of which 3 are typed must list the other 5 as residuals.
- If the text states conditions for MULTIPLE distinct beneficiary types (for example a different income cap for SC than for General), set the typed field to the MOST INCLUSIVE value (the one admitting the most people), put the full disjunction in residual_conditions, and add "multi_branch_eligibility" to parse_flags.
- "parse_flags" is a list, empty if nothing applies.

<<<BOOLEAN_FIELD_RULES>>>
A condition captured by one of these four does not also go in residual_conditions, unless the clause states more than the field can hold (another requirement, a time window, a minimum membership period, an exception) -- then the whole clause stays in residual_conditions too. "The applicant must be a citizen of India and a resident of West Bengal" sets citizen_of_india AND stays a residual.

"provenance" maps each NON-NULL typed field name to the exact substring of the eligibility text that the value came from. Copy that substring character for character from the text -- do not paraphrase, reformat, fix spelling, or add ellipses. It must appear verbatim in the text. Include an entry for every typed field you set to a non-null value, and no entries for null fields or for residual_conditions.

If the text says nothing about gender or residence, use null -- do not default to "any". Use "any" only when the text explicitly states the scheme is open to all of them. family.notes counts as a typed value too: if you fill it, it needs a span like any other field. Every provenance value is a plain string, even for the nested fields (age, income, land, education, family) -- give one span for the whole field, do not nest the provenance object.

Give the whole clause or sentence that states the condition, not a bare fragment: "The age limit for beneficiaries will be 55 years." is a usable span, "55 years" is not, because a reader must be able to see which condition the value came from. Still copy it exactly as written.

Respond with JSON only. No markdown fences, no preamble, no commentary.

Eligibility text:
<<<ELIGIBILITY_TEXT>>>
""".replace("<<<BOOLEAN_FIELD_RULES>>>", BOOLEAN_FIELD_RULES)

BOOLEAN_PROMPT_TEMPLATE = """You extract four eligibility conditions from Indian government welfare scheme text.

You do NOT decide whether anyone is eligible. You only record what is explicitly stated. Deterministic code does the comparisons later, so a wrong true is far worse than a null.

Return ONLY a JSON object with exactly these three top-level keys: "constraints", "provenance", "remove_from_residuals".

"constraints" must contain exactly these keys and no others:
{
  "citizen_of_india":           true|null,
  "bpl_household":              true|null,
  "requires_bocw_registration": true|null,
  "not_availing_other_scheme":  true|null
}

<<<BOOLEAN_FIELD_RULES>>>

"provenance" maps each field set to true to the exact substring of the eligibility text that states it. Give the whole clause or sentence, copied character for character from the text -- do not paraphrase, reformat, fix spelling, or add ellipses. If the condition is stated in several places, give a list of those clauses. No entries for null fields.

"remove_from_residuals" lists the already-extracted residual conditions (given below) that a field you set to true now captures completely, so the condition is not kept twice:
  [{"text": "<the residual, copied exactly from the list>", "fields": ["<field>", ...], "also_states": ["<phrase copied from that residual>", ...]}]
"also_states" lists every phrase of that residual that states something the cited fields do not: another requirement (residence, age, education, occupation, employment, an illness, a category), a time window, a minimum membership period, an exception. For "The applicant must be a citizen of India and a resident of West Bengal" it is ["a resident of West Bengal"]. A residual is removed only when its also_states is empty, so list every such phrase; when in doubt, include it. Saying the condition applies to the family member the applicant qualifies through is not extra: the field already covers that. Empty list if no residual is captured.

Respond with JSON only. No markdown fences, no preamble, no commentary.

Eligibility text:
<<<ELIGIBILITY_TEXT>>>

Residual conditions already extracted (JSON list):
<<<RESIDUALS>>>
""".replace("<<<BOOLEAN_FIELD_RULES>>>", BOOLEAN_FIELD_RULES)


def build_prompt(eligibility_text):
    return PROMPT_TEMPLATE.replace("<<<ELIGIBILITY_TEXT>>>", eligibility_text)


def build_boolean_prompt(eligibility_text, residuals):
    return (BOOLEAN_PROMPT_TEMPLATE
            .replace("<<<RESIDUALS>>>", json.dumps(residuals, indent=1, ensure_ascii=False))
            .replace("<<<ELIGIBILITY_TEXT>>>", eligibility_text))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _str_list(v, field):
    if not isinstance(v, list):
        return f"{field} must be a list or null"
    for item in v:
        if not isinstance(item, str) or not item.strip():
            return f"{field} must contain non-empty strings"
    return None


def validate_constraints(c):
    """Structural + enum validation. Returns an error string, or None if valid."""
    if not isinstance(c, dict):
        return "constraints is not an object"
    unknown = sorted(set(c) - set(SCHEMA))
    if unknown:
        return f"unknown keys in constraints: {unknown}"
    if "residual_conditions" not in c:
        return "missing residual_conditions"

    for field, subkeys in SCHEMA.items():
        if field not in c:
            continue
        v = c[field]
        if subkeys is not None:
            if v is None:
                continue  # null nested object == no constraint stated
            if not isinstance(v, dict):
                return f"{field} must be an object"
            extra = sorted(set(v) - subkeys)
            if extra:
                return f"unknown keys in {field}: {extra}"

    age = c.get("age") or {}
    for k in ("min", "max"):
        if age.get(k) is not None and not _is_int(age[k]):
            return f"age.{k} must be an integer or null"
    if _is_int(age.get("min")) and _is_int(age.get("max")) and age["min"] > age["max"]:
        return "age.min is greater than age.max"

    inc = c.get("income") or {}
    if inc.get("max_annual_inr") is not None and not _is_int(inc["max_annual_inr"]):
        return "income.max_annual_inr must be an integer or null"
    if inc.get("basis") is not None and inc["basis"] not in INCOME_BASIS:
        return f"income.basis not in {sorted(INCOME_BASIS)}: {inc['basis']!r}"

    if c.get("gender") is not None and c["gender"] not in GENDERS:
        return f"gender not in {sorted(GENDERS)}: {c['gender']!r}"

    if c.get("category") is not None:
        err = _str_list(c["category"], "category")
        if err:
            return err
        bad = [x for x in c["category"] if x not in SOCIAL_CATEGORIES]
        if bad:
            return f"category values not in {sorted(SOCIAL_CATEGORIES)}: {bad}"

    if c.get("occupation") is not None:
        err = _str_list(c["occupation"], "occupation")
        if err:
            return err

    land = c.get("land") or {}
    for k in ("max_acres", "min_acres"):
        if land.get(k) is not None and not _is_num(land[k]):
            return f"land.{k} must be a number or null"

    edu = c.get("education") or {}
    if edu.get("stage") is not None and not isinstance(edu["stage"], str):
        return "education.stage must be a string or null"
    for k in ("min_class", "max_class"):
        if edu.get(k) is not None and not _is_int(edu[k]):
            return f"education.{k} must be an integer or null"

    for k in ("state", "marital_status"):
        if c.get(k) is not None and not isinstance(c[k], str):
            return f"{k} must be a string or null"

    if c.get("residence") is not None and c["residence"] not in RESIDENCES:
        return f"residence not in {sorted(RESIDENCES)}: {c['residence']!r}"

    fam = c.get("family") or {}
    if fam.get("requires_dependent") is not None and not isinstance(fam["requires_dependent"], bool):
        return "family.requires_dependent must be a boolean or null"
    if fam.get("notes") is not None and not isinstance(fam["notes"], str):
        return "family.notes must be a string or null"

    err = validate_boolean_fields(c)
    if err:
        return err
    return _str_list(c["residual_conditions"], "residual_conditions")


def validate_boolean_fields(c):
    # false is rejected, not coerced: the text either requires the condition
    # or says nothing a filter could use, and "not required" would read as a
    # claim the scheme never made.
    for k in BOOLEAN_FIELDS:
        if c.get(k) is not None and c[k] is not True:
            return f"{k} must be true or null: {c[k]!r}"
    return None


def field_is_set(constraints, field):
    """Present in the output at all (including the 'any' sentinels)."""
    v = constraints.get(field)
    if isinstance(v, dict):
        return any(sub is not None for sub in v.values())
    if isinstance(v, list):
        return len(v) > 0
    return v is not None


def field_is_constraining(constraints, field):
    """Actually narrows who qualifies -- what deterministic filtering will use."""
    if not field_is_set(constraints, field):
        return False
    sentinels = NON_CONSTRAINING_VALUES.get(field)
    if not sentinels:  # list/dict fields have no sentinel and are never hashable
        return True
    return constraints.get(field) not in sentinels


def normalize_provenance(prov):
    """Models sometimes mirror a nested field's shape in provenance, e.g.
    {"family": {"notes": "..."}}. The span itself is usually a correct verbatim
    quote -- only the shape is wrong -- so flatten it rather than throw away a
    good extraction. A field genuinely drawn from several clauses keeps a list.
    """
    out = {}
    for field, span in (prov or {}).items():
        if isinstance(span, dict):
            span = [v for v in span.values() if isinstance(v, str) and v.strip()]
        if isinstance(span, list):
            parts = [v for v in span if isinstance(v, str) and v.strip()]
            span = parts[0] if len(parts) == 1 else parts
        out[field] = span
    return out


def validate_provenance(prov, constraints, eligibility_text, fields=TYPED_FIELDS):
    """Every non-null typed field needs a span, and every span must appear
    verbatim in the source text -- that is what makes a claim traceable.
    `fields` narrows the check to the fields a pass actually asked for."""
    if not isinstance(prov, dict):
        return "provenance is not an object"
    unknown = sorted(set(prov) - set(fields))
    if unknown:
        return f"unknown keys in provenance: {unknown}"

    for field, span in prov.items():
        spans = span if isinstance(span, list) else [span]
        if not spans:
            return f"provenance.{field} must be a non-empty string"
        for one in spans:
            if not isinstance(one, str) or not one.strip():
                return f"provenance.{field} must be a non-empty string"
            if one not in eligibility_text:
                return (f"provenance.{field} is not an exact substring of "
                        f"eligibility_text: {one[:80]!r}")

    missing = [f for f in fields
               if field_is_constraining(constraints, f) and f not in prov]
    if missing:
        return f"missing provenance for non-null fields: {missing}"
    # A span for a field that ended up null claims a source for a value that
    # does not exist -- a sign the model contradicted itself.
    orphaned = sorted(f for f in prov if not field_is_set(constraints, f))
    if orphaned:
        return f"provenance given for null fields: {orphaned}"
    return None


def parse_and_validate(raw_text, eligibility_text):
    text = lc._strip_fences(raw_text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as e:
        return None, f"invalid_json: {e}"
    if not isinstance(payload, dict):
        return None, "response is not a JSON object"

    unknown = sorted(set(payload) - {"constraints", "provenance", "parse_flags"})
    if unknown:
        return None, f"unknown top-level keys: {unknown}"
    if "constraints" not in payload:
        return None, "missing constraints"

    constraints = payload["constraints"]
    err = validate_constraints(constraints)
    if err:
        return None, err

    # fill in any omitted optional keys so downstream code sees a uniform shape
    merged = json.loads(json.dumps(EMPTY_CONSTRAINTS))
    for field, value in constraints.items():
        if isinstance(merged.get(field), dict):
            if isinstance(value, dict):
                merged[field].update(value)
            # value is None -> keep the all-null default for that object
        else:
            merged[field] = value

    prov = normalize_provenance(payload.get("provenance") or {})
    err = validate_provenance(prov, merged, eligibility_text)
    if err:
        return None, err

    flags = payload.get("parse_flags") or []
    if not isinstance(flags, list) or any(not isinstance(f, str) for f in flags):
        return None, "parse_flags must be a list of strings"

    return {"constraints": merged, "provenance": prov, "parse_flags": flags}, None


def parse_and_validate_booleans(raw_text, eligibility_text, residuals):
    """Migration-pass response: the four fields, their provenance (validated
    exactly as a full extraction's), and residuals the model says they replace.
    A residual named for removal must be one already in the record, word for
    word, and every field it cites must have come back true."""
    text = lc._strip_fences(raw_text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as e:
        return None, f"invalid_json: {e}"
    if not isinstance(payload, dict):
        return None, "response is not a JSON object"

    unknown = sorted(set(payload) - {"constraints", "provenance", "remove_from_residuals"})
    if unknown:
        return None, f"unknown top-level keys: {unknown}"
    c = payload.get("constraints")
    if not isinstance(c, dict):
        return None, "missing constraints"
    unknown = sorted(set(c) - set(BOOLEAN_FIELDS))
    if unknown:
        return None, f"unknown keys in constraints: {unknown}"
    err = validate_boolean_fields(c)
    if err:
        return None, err
    fields = {f: c.get(f) for f in BOOLEAN_FIELDS}

    prov = normalize_provenance(payload.get("provenance") or {})
    err = validate_provenance(prov, fields, eligibility_text, BOOLEAN_FIELDS)
    if err:
        return None, err

    removals = payload.get("remove_from_residuals") or []
    if not isinstance(removals, list):
        return None, "remove_from_residuals must be a list"
    for item in removals:
        if not (isinstance(item, dict) and set(item) == {"text", "fields", "also_states"}):
            return None, "remove_from_residuals entries must be {text, fields, also_states}"
        if not isinstance(item["also_states"], list):
            return None, "remove_from_residuals.also_states must be a list"
        if item["text"] not in residuals:
            return None, f"remove_from_residuals names a residual not in the record: {str(item['text'])[:80]!r}"
        cited = item["fields"]
        if not isinstance(cited, list):
            return None, "remove_from_residuals.fields must be a list"
        # An entry citing no field claims nothing the new fields capture (the
        # model tidying away non-conditions like "500 cities are covered").
        # Not worth failing a valid response over: with no span to cover it,
        # apply_boolean_fields keeps that residual.
        not_true = [f for f in cited if fields.get(f) is not True]
        if not_true:
            return None, f"remove_from_residuals cites fields that are not true: {not_true}"

    return {"constraints": fields, "provenance": prov, "remove": removals}, None


def _words(text):
    return re.findall(r"\w+", text.lower())


def residual_covered(residual, spans):
    """Share of the residual's words that appear in the cited fields' spans.
    Both are copies of the same eligibility clause when the removal is right."""
    words = _words(residual)
    span_words = {w for s in spans for w in _words(s)}
    return bool(words) and sum(w in span_words for w in words) / len(words) >= RESIDUAL_COVERAGE_MIN


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_one(pool, record, delay, reasoning_effort=None):
    """-> (result|None, error|None, status). Status mirrors pool_generate plus
    'invalid' for a response that failed validation."""
    prompt = build_prompt(record["eligibility_text"])
    res = lc.pool_generate(pool, prompt, delay,
                           reasoning_effort if pool.name == "groq" else None)
    if res["status"] != "ok":
        return None, res["error"], res["status"]
    parsed, err = parse_and_validate(res["text"], record["eligibility_text"])
    if parsed is None:
        return None, err, "invalid"
    parsed["raw_response"] = res["text"]
    return parsed, None, "ok"


def extract_booleans(pool, record, existing, delay, reasoning_effort=None):
    """Migration pass for one scheme: same contract as extract_one."""
    residuals = existing["constraints"]["residual_conditions"]
    prompt = build_boolean_prompt(record["eligibility_text"], residuals)
    res = lc.pool_generate(pool, prompt, delay,
                           reasoning_effort if pool.name == "groq" else None)
    if res["status"] != "ok":
        return None, res["error"], res["status"]
    parsed, err = parse_and_validate_booleans(res["text"], record["eligibility_text"], residuals)
    if parsed is None:
        return None, err, "invalid"
    parsed["raw_response"] = res["text"]
    return parsed, None, "ok"


def _with_boolean_fields(constraints, values, residuals):
    """Constraints with the four fields set, keeping residual_conditions last."""
    out = {k: v for k, v in constraints.items() if k != "residual_conditions"}
    out.update(values)
    out["residual_conditions"] = residuals
    return out


def apply_boolean_fields(existing, parsed, pool, latency):
    """Fold a validated migration result into an existing record. A residual
    is dropped only if its words are covered by the spans of the fields it is
    said to duplicate; otherwise it is kept and listed, never lost."""
    removed, kept, states_more = [], [], []
    for item in parsed["remove"]:
        # Checked first: the model saying the residual carries another
        # condition outranks its own request to remove it.
        if item["also_states"]:
            states_more.append(item["text"])
            continue
        spans = []
        for f in item["fields"]:
            span = parsed["provenance"][f]
            spans += span if isinstance(span, list) else [span]
        (removed if residual_covered(item["text"], spans) else kept).append(item["text"])

    residuals = [r for r in existing["constraints"]["residual_conditions"] if r not in removed]
    existing["constraints"] = _with_boolean_fields(existing["constraints"],
                                                   parsed["constraints"], residuals)
    existing["provenance"].update(parsed["provenance"])
    existing["boolean_fields_pass"] = {
        "provider": pool.name,
        "model": pool.model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "latency_seconds": round(latency, 3),
        "residuals_removed": removed,
        "residuals_kept_not_covered_by_span": kept,
        "residuals_kept_states_more": states_more,
        "raw_response": parsed["raw_response"],
    }
    return existing


def mark_boolean_fields_failed(existing, error):
    """The pass never validated: the four fields are null (nothing claimed)
    and the residuals stay untouched, so the scheme reads exactly as before.
    Having the keys means later runs skip it, like a failed extraction."""
    c = existing["constraints"]
    existing["constraints"] = _with_boolean_fields(c, {f: None for f in BOOLEAN_FIELDS},
                                                   c["residual_conditions"])
    existing["boolean_fields_pass"] = {
        "failed": True,
        "failure_reason": error,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return existing


def build_record(slug, record, parsed, pool, latency):
    return {
        "slug": slug,
        "constraints": parsed["constraints"],
        "provenance": parsed["provenance"],
        "parse_flags": parsed["parse_flags"],
        "eligibility_text_chars": len(record["eligibility_text"]),
        "provider": pool.name,
        "model": pool.model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "latency_seconds": round(latency, 3),
        "raw_response": parsed["raw_response"],
        "source_url": record.get("source_url"),
    }


def build_failed_record(slug, record, error):
    """Stand-in for a scheme whose output never validated. Every typed field is
    null, so downstream it has zero binding constraints and goes to manual
    review like any other low-coverage scheme. The flag is what stops it being
    read as a scheme that genuinely states no conditions -- its residuals are
    unknown, not empty."""
    return {
        "slug": slug,
        "extraction_failed": True,
        "failure_reason": error,
        "constraints": json.loads(json.dumps(EMPTY_CONSTRAINTS)),
        "provenance": {},
        "parse_flags": [],
        "eligibility_text_chars": len(record["eligibility_text"]),
        "provider": None,
        "model": None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "latency_seconds": None,
        "raw_response": None,
        "source_url": record.get("source_url"),
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def build_report(records, total_candidates, failures, pools, skipped):
    n = len(records)
    fill = {}
    for field in TYPED_FIELDS:
        hits = sum(1 for r in records if field_is_constraining(r["constraints"], field))
        fill[field] = round(hits / n, 4) if n else 0.0

    residual_counts = [len(r["constraints"]["residual_conditions"]) for r in records]
    dist = {}
    if residual_counts:
        srt = sorted(residual_counts)
        dist = {
            "min": srt[0],
            "median": statistics.median(srt),
            "p90": srt[min(len(srt) - 1, int(round(0.9 * (len(srt) - 1))))],
            "max": srt[-1],
            "mean": round(sum(srt) / len(srt), 2),
            "histogram": dict(sorted(collections.Counter(srt).items())),
            "schemes_with_zero": sum(1 for c in srt if c == 0),
        }

    flag_counts = collections.Counter(f for r in records for f in r["parse_flags"])
    top_residual = sorted(records, key=lambda r: -len(r["constraints"]["residual_conditions"]))[:10]

    total_tokens = sum(p.usage["total"] for p in pools)
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_candidates": total_candidates,
        "extracted_this_run": n,
        "skipped_already_extracted": skipped,
        "failed": len(failures),
        "failed_slugs": sorted(failures),
        "failure_reasons": failures,
        "field_fill_rate": fill,
        "residual_conditions_distribution": dist,
        "parse_flag_counts": dict(flag_counts),
        "most_residual_conditions": [
            {"slug": r["slug"], "residuals": len(r["constraints"]["residual_conditions"])}
            for r in top_residual
        ],
        "token_usage": {
            p.name: {"model": p.model, "requests": p.usage["requests"],
                     "prompt_tokens": p.usage["prompt"], "completion_tokens": p.usage["completion"],
                     "total_tokens": p.usage["total"],
                     "keys_exhausted": p.exhausted_keys}
            for p in pools
        },
        "total_tokens": total_tokens,
        "tokens_per_scheme": round(total_tokens / n, 1) if n else None,
    }


def print_report(rep):
    print()
    print("=" * 66)
    print("CONSTRAINT EXTRACTION REPORT")
    print("=" * 66)
    print(f"queued this run        : {rep['total_candidates']}")
    print(f"extracted this run     : {rep['extracted_this_run']}")
    print(f"total extracted on disk: {rep.get('total_on_disk', rep['extracted_this_run'])}")
    print(f"skipped (already done) : {rep['skipped_already_extracted']}")
    print(f"marked failed on disk  : {len(rep.get('extraction_failed_on_disk', []))}"
          "  (manual review; excluded from the stats below)")
    print(f"failed                 : {rep['failed']}")
    for slug in rep["failed_slugs"]:
        print(f"    {slug}: {rep['failure_reasons'][slug][:110]}")
    print()
    print("field fill rate (share of schemes with a non-null value):")
    for field, rate in sorted(rep["field_fill_rate"].items(), key=lambda kv: -kv[1]):
        bar = "#" * int(round(rate * 40))
        print(f"  {field:16s} {rate*100:5.1f}%  {bar}")
    d = rep["residual_conditions_distribution"]
    if d:
        print()
        print("residual_conditions per scheme:")
        print(f"  min={d['min']}  median={d['median']}  p90={d['p90']}  max={d['max']}  mean={d['mean']}")
        print(f"  schemes with zero residuals: {d['schemes_with_zero']}")
        print("  histogram (count -> schemes): " +
              "  ".join(f"{k}:{v}" for k, v in d["histogram"].items()))
    print()
    print("parse flags:")
    if rep["parse_flag_counts"]:
        for flag, count in sorted(rep["parse_flag_counts"].items(), key=lambda kv: -kv[1]):
            print(f"  {flag:30s} {count}")
    else:
        print("  none")
    print()
    bp = rep.get("boolean_fields_pass")
    if bp:
        print("boolean fields pass (all records on disk):")
        print(f"  records migrated                    : {bp['records_migrated']}")
        print(f"  records failed (fields left null)   : {len(bp['records_failed'])}")
        print(f"  residuals removed (now typed)       : {bp['residuals_removed']}")
        print(f"  removal refused (span doesn't cover): {bp['residuals_kept_not_covered_by_span']}")
        print(f"  removal refused (states more)       : {bp['residuals_kept_states_more']}")
        print(f"  records still without the fields    : {bp['records_still_without_fields']}")
        print()
    print("10 schemes with the most residual_conditions:")
    for entry in rep["most_residual_conditions"]:
        print(f"  {entry['residuals']:3d}  {entry['slug']}")
    print()
    for name, u in rep["token_usage"].items():
        if u["requests"]:
            print(f"  {name:6s} {u['requests']:4d} req  {u['total_tokens']:7d} tokens")
    print(f"total tokens           : {rep['total_tokens']}")
    print(f"tokens per scheme      : {rep['tokens_per_scheme']}")
    print("=" * 66)


def write_json(path, obj):
    """Write through a temp file: an interrupted run must never leave a
    half-written record, least of all over a good one."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def load_on_disk(out_dir):
    on_disk = []
    for path in sorted(Path(out_dir).glob("*.json")):
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        # The directory also holds report files (extract_report.json,
        # coverage_report.json), so identify records by shape, not filename.
        if isinstance(rec, dict) and "constraints" in rec and "slug" in rec:
            on_disk.append(rec)
    return on_disk


class Terminated(KeyboardInterrupt):
    """SIGTERM/SIGHUP, raised so they take the same exit path as Ctrl-C."""


def _raise_terminated(signum, frame):
    raise Terminated(signal.Signals(signum).name)


def run_queue(pools, pending, attempt, on_success, on_failure, out_dir, delay,
              checkpoint=None):
    """Work through `pending` on the provider chain, writing each scheme's
    record as soon as it is settled.

    attempt(pool, slug) returns (parsed, err, status) like extract_one;
    on_success(slug, parsed, pool, elapsed) and on_failure(slug, err) return
    the record to write. checkpoint(written, failures, still_queued) is called
    every CHECKPOINT_EVERY settled schemes.
    -> (records written, failures, stopped_reason, still_queued)
    """
    queue = deque(pending)
    attempts = collections.Counter()
    written = []
    failures = {}
    consecutive_call_failures = 0
    stopped_reason = "completed"

    try:
        with tqdm(total=len(pending), desc="extracting", unit="scheme") as bar:
            while queue:
                pool = lc.select_pool(pools)
                if pool is None:
                    reasons = {k["reason"] for p in pools for k in p.exhausted_keys}
                    stopped_reason = ("quota_exhausted"
                                      if reasons and reasons <= {"daily_quota", "repeated_429"}
                                      else "providers_unavailable")
                    break
                for p in pools:
                    if p is not pool and p.cooldown:
                        p.cooldown -= 1

                slug = queue.popleft()
                t0 = time.monotonic()
                parsed, err, status = attempt(pool, slug)
                elapsed = time.monotonic() - t0

                if status == "exhausted":
                    queue.appendleft(slug)
                    tqdm.write(f"  [{pool.name}] out of quota; handing off")
                    continue

                if status in ("failed", "fatal"):
                    # Provider's fault, not the scheme's: keep the scheme's
                    # retry budget intact and bench the provider instead.
                    queue.appendleft(slug)
                    if status == "fatal":
                        pool.disable("fatal_error")
                        tqdm.write(f"  [{pool.name}] disabled for this run: {str(err)[:120]}")
                    else:
                        pool.cooldown = 2
                        tqdm.write(f"  [{pool.name}] call failed ({str(err)[:100]}); benching")
                    consecutive_call_failures += 1
                    if consecutive_call_failures >= MAX_CONSECUTIVE_CALL_FAILURES:
                        stopped_reason = "repeated_provider_failures"
                        break
                    continue
                consecutive_call_failures = 0

                if status == "invalid":
                    attempts[slug] += 1
                    if attempts[slug] >= MAX_ATTEMPTS_PER_SLUG:
                        # Recorded, not retried on later runs, so one scheme
                        # the model cannot get right never blocks the pipeline.
                        failures[slug] = err
                        write_json(out_dir / f"{slug}.json", on_failure(slug, err))
                        bar.update(1)
                        tqdm.write(f"  FAILED {slug} after {attempts[slug]} attempts: {str(err)[:110]}")
                    else:
                        # Retry on the other provider where possible: at
                        # temperature 0 the same model would likely repeat itself.
                        pool.cooldown = 1
                        queue.append(slug)
                        tqdm.write(f"  retrying {slug}: {str(err)[:100]}")
                    continue

                out = on_success(slug, parsed, pool, elapsed)
                write_json(out_dir / f"{slug}.json", out)
                written.append(out)
                failures.pop(slug, None)
                bar.update(1)
                bar.set_postfix_str(" ".join(
                    f"{p.name}={p.usage['total']/1000:.1f}k" for p in pools if p.usage["requests"]))
                if checkpoint and bar.n % CHECKPOINT_EVERY == 0:
                    checkpoint(written, failures, len(queue))
                if queue:
                    time.sleep(delay)
    except KeyboardInterrupt as e:
        stopped_reason = f"terminated_{e}" if isinstance(e, Terminated) else "interrupted"
        print(f"\n  {stopped_reason} -- keeping everything extracted so far")
    return written, failures, stopped_reason, len(queue)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=30,
                     help="max schemes to extract this run (default 30; 0 extracts nothing "
                          "and just rebuilds the report). Use --all for the whole corpus.")
    ap.add_argument("--slug", type=str, default=None,
                     help="extract a single scheme and print the full request/response")
    ap.add_argument("--force", action="store_true", help="re-extract schemes that already have output")
    ap.add_argument("--provider", choices=["gemini", "groq"], default=None,
                     help="restrict to one provider")
    ap.add_argument("--out-dir", type=Path, default=CONSTRAINTS_DIR,
                     help=f"output directory (default {CONSTRAINTS_DIR})")
    ap.add_argument("--gemini-model", default=lc.DEFAULT_GEMINI_MODEL)
    ap.add_argument("--delay", type=float, default=1.0, help="seconds between API calls")
    ap.add_argument("--groq-reasoning-effort", choices=["low", "medium", "high"], default=None)
    ap.add_argument("--all", action="store_true",
                     help="no cap: extract every remaining scheme (resumable across sittings)")
    ap.add_argument("--slugs", type=str, default=None,
                     help="comma-separated slugs to extract (e.g. to re-run specific failures)")
    ap.add_argument("--order", choices=["alpha", "varied"], default="alpha",
                     help="'varied' spreads the selection across the eligibility-length "
                          "distribution instead of taking the first N alphabetically")
    ap.add_argument("--add-boolean-fields", action="store_true",
                     help="migration pass: add the four true|null fields to existing records "
                          "that lack them, instead of extracting new schemes")
    args = ap.parse_args()
    if args.add_boolean_fields and (args.slug or args.force):
        # --force would re-run on records whose residuals were already trimmed;
        # a clean redo of a migrated record is a full --force extraction.
        ap.error("--add-boolean-fields takes neither --slug nor --force")
    # tqdm.write sends retry/quota/FAILED messages to stdout, which is block-
    # buffered when a long run is logged to a file: they would appear late, or
    # never if the run is killed, while the progress bar (stderr) looks live.
    sys.stdout.reconfigure(line_buffering=True)

    load_dotenv(ROOT / ".env")
    logging.getLogger("google_genai.models").setLevel(logging.ERROR)

    pools = [
        lc.ProviderPool("gemini", args.gemini_model, "GEMINI_API_KEY", lc.make_gemini_client,
                        brief=False, batch_size=1),
        lc.ProviderPool("groq", lc.GROQ_MODEL, "GROQ_API_KEY", lc.make_groq_client,
                        brief=False, batch_size=1),
    ]
    if args.provider:
        pools = [p for p in pools if p.name == args.provider]
    for pool in pools:
        if not pool.keys:
            print(f"No API keys found for {pool.name} (see .env.example).", file=sys.stderr)
            sys.exit(1)
    print("Providers (tried in order): " +
          ", ".join(f"{p.name}={p.model}" for p in pools))

    records = lc.load_records()
    out_dir = Path(args.out_dir)

    if args.slug:
        if args.slug not in records:
            print(f"no such slug: {args.slug}", file=sys.stderr)
            sys.exit(1)
        rec = records[args.slug]
        if not (rec["eligibility_text"] or "").strip():
            print(f"{args.slug} has empty eligibility_text", file=sys.stderr)
            sys.exit(1)
        print("=" * 66)
        print("REQUEST PROMPT")
        print("=" * 66)
        print(build_prompt(rec["eligibility_text"]))
        t0 = time.monotonic()
        parsed, err, status = extract_one(pools[0], rec, args.delay, args.groq_reasoning_effort)
        print("=" * 66)
        print(f"RESULT ({status})")
        print("=" * 66)
        if parsed:
            print(json.dumps(build_record(args.slug, rec, parsed, pools[0],
                                           time.monotonic() - t0), indent=2, ensure_ascii=False))
        else:
            print(json.dumps({"slug": args.slug, "status": status, "error": err},
                              indent=2, ensure_ascii=False))
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    if args.add_boolean_fields:
        existing = {r["slug"]: r for r in load_on_disk(out_dir)}
        # Stand-ins for failed extractions are left as they are: there is no
        # extraction to extend, and reopening one is a full --force re-run.
        candidates = sorted(s for s, r in existing.items()
                            if s in records and not r.get("extraction_failed"))
        done = {s for s in candidates
                if all(f in existing[s]["constraints"] for f in BOOLEAN_FIELDS)}
    else:
        candidates = [s for s in sorted(records)
                      if (records[s]["eligibility_text"] or "").strip()]
        done = {s for s in candidates if (out_dir / f"{s}.json").exists()}
    if args.slugs:
        wanted = [s.strip() for s in args.slugs.split(",") if s.strip()]
        missing = [s for s in wanted if s not in records]
        if missing:
            print(f"unknown slugs: {missing}", file=sys.stderr)
            sys.exit(1)
        candidates = [s for s in candidates if s in set(wanted)]
    pending = [s for s in candidates if args.force or s not in done]
    skipped = len(candidates) - len(pending)

    if args.all:
        args.limit = None
    if args.order == "varied":
        # Walk the eligibility-length distribution so a small run spans trivial
        # one-liners through to the 9k-character monsters.
        by_len = sorted(pending, key=lambda s: len(records[s]["eligibility_text"]))
        if args.limit is not None and args.limit < len(by_len):
            step = len(by_len) / args.limit
            pending = [by_len[min(len(by_len) - 1, int(i * step))] for i in range(args.limit)]
        else:
            pending = by_len
    elif args.limit is not None:
        pending = pending[: args.limit]   # --limit 0 = extract nothing, just rebuild the report

    print(f"{len(candidates)} schemes with eligibility_text, {skipped} already extracted, "
          f"{len(pending)} queued this run (order={args.order}).")

    if args.add_boolean_fields:
        def attempt(pool, slug):
            return extract_booleans(pool, records[slug], existing[slug],
                                    args.delay, args.groq_reasoning_effort)

        def on_success(slug, parsed, pool, elapsed):
            return apply_boolean_fields(existing[slug], parsed, pool, elapsed)

        def on_failure(slug, err):
            return mark_boolean_fields_failed(existing[slug], err)
    else:
        def attempt(pool, slug):
            return extract_one(pool, records[slug], args.delay, args.groq_reasoning_effort)

        def on_success(slug, parsed, pool, elapsed):
            return build_record(slug, records[slug], parsed, pool, elapsed)

        def on_failure(slug, err):
            return build_failed_record(slug, records[slug], err)

    # Its own report file in this mode, so the last full extraction's survives.
    report_path = out_dir / (BOOLEAN_REPORT_FILENAME if args.add_boolean_fields
                             else REPORT_FILENAME)

    def write_report(written, failures, stopped_reason, still_queued):
        on_disk = load_on_disk(out_dir)
        # Failed stand-ins are all-null by construction; counting them would read
        # as schemes with no fields and no residuals, so stats cover real extractions.
        report = build_report([r for r in on_disk if not r.get("extraction_failed")],
                              len(pending), failures, pools, skipped)
        report["extracted_this_run"] = len(written)
        report["total_on_disk"] = len(on_disk)
        report["extraction_failed_on_disk"] = sorted(
            r["slug"] for r in on_disk if r.get("extraction_failed"))
        report["stopped_reason"] = stopped_reason
        report["still_queued"] = still_queued
        if args.add_boolean_fields:
            # Read back from every record, not just this run's, because the
            # pass spans several sittings and each one rewrites this file.
            passes = [r["boolean_fields_pass"] for r in on_disk if "boolean_fields_pass" in r]
            ok = [bp for bp in passes if not bp.get("failed")]
            report["boolean_fields_pass"] = {
                "records_migrated": len(ok),
                "records_failed": sorted(r["slug"] for r in on_disk
                                         if r.get("boolean_fields_pass", {}).get("failed")),
                "records_still_without_fields": sum(
                    1 for r in on_disk if not r.get("extraction_failed")
                    and not all(f in r["constraints"] for f in BOOLEAN_FIELDS)),
                "residuals_removed": sum(len(bp["residuals_removed"]) for bp in ok),
                "residuals_kept_not_covered_by_span": sum(
                    len(bp["residuals_kept_not_covered_by_span"]) for bp in ok),
                "residuals_kept_states_more": sum(
                    len(bp.get("residuals_kept_states_more", [])) for bp in ok),
            }
        write_json(report_path, report)
        return report

    # A kill from whatever launched the run (a closed terminal, a supervising
    # session ending) must still reach the report below, as Ctrl-C does.
    signal.signal(signal.SIGTERM, _raise_terminated)
    signal.signal(signal.SIGHUP, _raise_terminated)
    write_report([], {}, "running", len(pending))
    written, failures, stopped_reason, still_queued = run_queue(
        pools, pending, attempt, on_success, on_failure, out_dir, args.delay,
        checkpoint=lambda w, f, q: write_report(w, f, "running", q))

    report = write_report(written, failures, stopped_reason, still_queued)
    print_report(report)
    if stopped_reason != "completed":
        print(f"stopped early: {stopped_reason} ({still_queued} still queued)")
    print(f"Output: {out_dir}/<slug>.json")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()

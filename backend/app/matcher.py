"""Deterministic eligibility check of candidate schemes against a profile (matcher v2).

match(profile, other_facts, candidates) evaluates each candidate's typed
constraints (data/interim/constraints/<slug>.json) against the profile from
backend/app/understand.py. Pure code, no LLM; bge-m3 embeddings are used for
the two fuzzy comparisons (occupation, and other_facts against residual
conditions). Every condition comes back as pass / fail / unknown with the
constraint, the profile value, its confidence and the source_span that stated
the rule.

A condition can only fail when the profile value is present, has high or
medium confidence, and is unambiguous:

- numeric (age, income, land, education_class): compared with min/max.
  Income fails only when the bases line up (family vs individual income, read
  from the evidence); land only when the evidence says the land is owned.
- categorical (gender, category, residence, marital_status, state,
  bpl_household, registered_construction_worker): same rule. category is a
  disjunction; it fails only when every listed category is ruled out
  (caste_category, is_minority, and PwD from disability_percent >= 40, the
  benchmark disability). Free-text marital_status rules that don't map onto
  the profile's values never fail.
- state: central schemes pass; state schemes (the scheme record's canonical
  state, not the free-text constraint) fail when the profile's state differs.
- occupation and education stage are fuzzy: they pass or stay unknown, never
  fail. Occupation passes when bge-m3 similarity to a constraint occupation is
  at least OCCUPATION_THRESHOLD, or (rule bocw_occupation) when the scheme
  requires construction-board registration and the person says they are
  registered.
- The person's own age, gender, marital status and disability cannot decide
  a scheme when profile.applying_for says they are asking for someone else
  (rule applying_for), or the scheme requires a dependent.
- not_availing_other_scheme stays unknown and non-decisive unless
  prior_benefit_schemes or other_facts show a prior benefit; then it is a
  decisive unknown (needs_checking), never a fail.
- citizenship and residence in the scheme's state are listed under
  to_confirm and never change the status.

Rules added in v2, each can be switched off through `rules` (used only to
attribute changed verdicts in evaluation; all are on by default):

- applying_for: whether the person is asking for someone else comes from
  profile.applying_for (off: v1's guess from family/other_facts wording)
- priority: category/occupation conditions whose source span is priority,
  preference, reservation or "first be assigned" language are not conditions;
  they are returned as preferences
- multi_branch: records flagged multi_branch_eligibility cannot fail on
  category (occupation never fails anyway)
- bocw_occupation: see above
- person_level_pass: eligible needs at least one passed condition about the
  person (anything but state); otherwise needs_checking, "nothing about you
  could be checked automatically"
- residual_touch: an other_fact whose similarity to one of the scheme's
  residual conditions is at least RESIDUAL_TOUCH_THRESHOLD is attached to the
  result and turns eligible into needs_checking. It never produces
  not_eligible.

Status: not_eligible if any typed condition fails; needs_checking if none
fails and a decisive condition is unknown (or a v2 rule above downgrades an
eligible); eligible otherwise. Schemes with no usable constraint record are
always needs_checking. Residual conditions and family notes are always
returned as unverified_conditions.

select_clarifying_field(results) picks the profile field that blocks the
most top candidates not marked not_eligible, weighted 1/rank: fields that are
missing, and fields that are filled but could not be used (an occupation no
scheme wording reaches, an income whose basis is unclear). Below
MIN_CLARIFY_SCORE it asks nothing.
"""
import json
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CONSTRAINTS_DIR = ROOT / "data" / "interim" / "constraints"
SCHEMES_DIR = ROOT / "data" / "interim" / "schemes"

RULES = ("applying_for", "priority", "multi_branch", "bocw_occupation", "person_level_pass", "residual_touch")

# Calibrated on the corpus's 781 distinct constraint occupation strings
# (2026-09-22): exact synonyms score 0.85+ ("farmer engaged in agriculture"
# 0.855, "building worker" 0.965), while the closest wrong pairs seen were
# below it ("unemployed" vs "employed" 0.764, "school teacher" vs "School
# students" 0.769, "construction worker" vs a bare "Worker" 0.849).
OCCUPATION_THRESHOLD = 0.85
# Chosen from the 977 (other_fact, residual) pairs of the dev split's top-10
# and target schemes (understand-v4, 2026-09-22). Genuine touches scored
# 0.442-0.776 and unrelated pairs reach 0.499, so similarity cannot separate
# them below about 0.50. 0.48 is the lowest value that still discriminates
# (it touches 72% of candidates that have other_facts, against 96% at 0.40)
# and sits just under the lowest reachable genuine touch: three months of
# work vs a one-year membership rule (0.493). A lower value costs little,
# because the only effect is eligible -> needs_checking.
RESIDUAL_TOUCH_THRESHOLD = 0.48
# A field must block at least the weight of the second-ranked candidate
# (1/2) before it is worth a question.
MIN_CLARIFY_SCORE = 0.5
BENCHMARK_DISABILITY = 40            # RPwD Act benchmark disability, for the PwD category
CONFIDENT = ("high", "medium")
CASTE_CATEGORIES = ("SC", "ST", "OBC", "EWS", "General")
ON_BEHALF = ("child", "spouse", "parent", "other")
NOTHING_ABOUT_YOU = "nothing about you could be checked automatically"

_PRIORITY = re.compile(r"\bprior(?:ity|ities|itis\w*|itiz\w*)\b|\bprefer(?:ence|ences|ential|red)\b|\breservation\b|"
                       r"first be assigned", re.I)
# v1's guess at "asking for someone else", kept only for rule attribution.
_LEGACY_DEPENDENT = re.compile(r"\b(?:child|children|kids?|sons?|daughters?|baby|infant|newborn|girl|boy|mother|father|"
                               r"parents?|husband|wife|spouse|dependents?|grandchild(?:ren)?)\b", re.I)
_FAMILY_BASIS = re.compile(r"family|household|parents?|father|mother|husband|wife|\bwe\b|\bour\b|together|"
                           r"परिवार|घर की|కుటుంబ|పరివార|পরিবার|குடும்ப", re.I)
_INDIVIDUAL_BASIS = re.compile(r"\bI (?:earn|make|get)\b|\bmy (?:salary|income|earnings?|wages?|pension)\b|"
                               r"मेरी (?:आय|आमदनी|कमाई|तनख्वाह)|कमाता|कमाती|నా (?:ఆదాయం|జీతం)", re.I)
_OWNED = re.compile(r"\bown(?:s|ed)?\b|\bmy (?:land|farm)|landholding|patta|मेरे पास|मेरी ज़मीन|నాకు|నా భూమి", re.I)
_LEASED = re.compile(r"leas|\brent|tenant|share ?crop|batai|ठेके|बटाई|కౌలు", re.I)
_PRIOR_BENEFIT = re.compile(r"\b(?:already|currently)\s+(?:getting|receiving|availing|get|receive|avail)|"
                            r"\b(?:received|availed|beneficiary|benefited)\b|\b(?:took|taken|got)\s+an?\s+[\w-]+\s+loan\b",
                            re.I)
_NEGATED = re.compile(r"\b(?:never|not|no|without)\b|n't", re.I)

# Education stage: constraint text -> the profile stages it admits. Tried in
# order; a matched phrase is removed so "post-graduation" is not also read as
# "graduation".
_STAGE_PATTERNS = [
    (r"ph\.?\s*d|doctora", {"doctoral"}),
    (r"post[\s-]?grad\w*|\bp\.?g\.?\b|master\w*|\bm\.?(?:a|sc|com|tech|phil)\b\.?|\bmba\b", {"postgraduate"}),
    (r"post[\s-]?matric\w*", {"higher_secondary", "diploma", "undergraduate", "postgraduate", "doctoral"}),
    (r"pre[\s-]?matric\w*", {"school"}),
    (r"higher education", {"diploma", "undergraduate", "postgraduate", "doctoral"}),
    (r"under[\s-]?grad\w*|graduat\w*|bachelor\w*|degree|college|\bb\.?(?:a|sc|com|tech|e)\b\.?", {"undergraduate"}),
    (r"higher secondary|senior secondary|intermediate|plus two|\+2|\b1[12](?:th)?\b", {"higher_secondary"}),
    (r"diploma|\biti\b|polytechnic|vocational", {"diploma"}),
    (r"school|primary|secondary", {"school", "higher_secondary"}),
]
_MARITAL_WORDS = {"widow": "widowed", "widows": "widowed", "widowed": "widowed", "widower": "widowed",
                  "unmarried": "unmarried", "married": "married", "divorced": "divorced", "divorcee": "divorced",
                  "divorcees": "divorced", "separated": "separated", "abandoned": "abandoned", "deserted": "abandoned"}
_MARITAL_FILLER = {"women", "woman", "including", "persons", "person", "girls", "girl", "men"}
_MARITAL_ANY = {"any", "for all", "all"}
_CATEGORY_FIELD = {**{c: "caste_category" for c in CASTE_CATEGORIES}, "Minority": "is_minority", "PwD": "disability_percent"}


@lru_cache(maxsize=None)
def load_record(slug):
    path = CONSTRAINTS_DIR / f"{slug}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


@lru_cache(maxsize=None)
def scheme_meta(slug):
    d = json.loads((SCHEMES_DIR / f"{slug}.json").read_text(encoding="utf-8"))
    return {"scheme_name": d.get("scheme_name"), "level": d.get("level"), "state": d.get("state")}


_EMBEDDINGS = {}


def embed_many(texts):
    """Normalized bge-m3 vectors for texts, computed once per distinct text."""
    missing = sorted({t for t in texts if t not in _EMBEDDINGS})
    if missing:
        from app.retrieval import default_retriever
        for t, v in zip(missing, default_retriever().embed(missing)):
            _EMBEDDINGS[t] = v
    return np.stack([_EMBEDDINGS[t] for t in texts]) if texts else np.zeros((0, 1024), dtype="float32")


def _similarity(a, b):
    va, vb = embed_many([a.strip().lower(), b.strip().lower()])
    return float(va @ vb)


def _span(evidence, key):
    v = (evidence or {}).get(key)
    return " ".join(v) if isinstance(v, list) else (v or "")


def fact_text(fact):
    """The English text of an other_facts entry (v4 objects or v1-v3 strings)."""
    return fact["en"] if isinstance(fact, dict) else fact


def income_basis(evidence_text):
    """'family', 'individual' or None, read from the income evidence."""
    if _FAMILY_BASIS.search(evidence_text):
        return "family"
    if _INDIVIDUAL_BASIS.search(evidence_text):
        return "individual"
    return None


def land_basis(evidence_text):
    if _LEASED.search(evidence_text):
        return "leased"
    if _OWNED.search(evidence_text):
        return "owned"
    return None


def stages_admitted(stage_text):
    """Profile education stages a constraint's free-text stage admits, or None if it doesn't map."""
    rest, found = (stage_text or "").lower(), set()
    for pattern, stages in _STAGE_PATTERNS:
        if re.search(pattern, rest):
            found |= stages
            rest = re.sub(pattern, " ", rest)
    return found or None


def marital_admitted(text):
    """Profile marital statuses a constraint admits; None if it doesn't map
    cleanly; 'any' if it doesn't restrict."""
    t = (text or "").strip().lower()
    if t in _MARITAL_ANY:
        return "any"
    admitted = set()
    for piece in re.split(r"[,/;&]|\bor\b|\band\b", t):
        words = [w for w in re.findall(r"[a-z-]+", piece) if w not in _MARITAL_FILLER]
        if not words:
            continue
        if len(words) != 1 or words[0] not in _MARITAL_WORDS:
            return None
        admitted.add(_MARITAL_WORDS[words[0]])
    return admitted or None


def is_priority_span(span):
    return bool(span and _PRIORITY.search(span))


class _Context:
    def __init__(self, profile, other_facts, confidence, evidence, rules):
        self.profile = profile
        self.facts = [fact_text(f) for f in (other_facts or []) if fact_text(f)]
        self.confidence = confidence
        self.evidence = evidence or {}
        self.rules = rules
        if "applying_for" in rules:
            self.on_behalf = profile.get("applying_for") in ON_BEHALF
        else:
            cues = " ".join([(profile.get("family") or "").replace("_", " ")] + self.facts)
            self.on_behalf = bool(_LEGACY_DEPENDENT.search(cues))
        self.prior_benefit = [f"named scheme: {s}" for s in (profile.get("prior_benefit_schemes") or [])]
        self.prior_benefit += [f for f in self.facts if _PRIOR_BENEFIT.search(f) and not _NEGATED.search(f)]

    def value(self, key):
        return self.profile.get(key)

    def conf(self, key):
        if self.profile.get(key) is None:
            return None
        return "medium" if self.confidence is None else self.confidence.get(key)

    def usable(self, key):
        return self.value(key) is not None and self.conf(key) in CONFIDENT


def _condition(field, result, constraint, ctx=None, key=None, span=None, note=None, decisive=True,
               ask=None, ask_kind=None):
    """ask/ask_kind: the profile field a question could settle, and whether it
    is "missing" (not stated or low confidence) or "refine" (stated but too
    vague or ambiguous to use)."""
    return {"field": field, "result": result, "decisive": decisive, "constraint": constraint,
            "profile_value": ctx.value(key) if ctx and key else None,
            "confidence": ctx.conf(key) if ctx and key else None,
            "source_span": span, "note": note, "ask_field": ask, "ask_kind": ask_kind if ask else None}


def _unusable(field, key, ctx, span, constraint):
    note = "not stated" if ctx.value(key) is None else f"{ctx.conf(key)} confidence"
    return _condition(field, "unknown", constraint, ctx, key, span, note, ask=key, ask_kind="missing")


def _check_range(field, key, lo, hi, ctx, span, constraint, ambiguous=None):
    if not ctx.usable(key):
        return _unusable(field, key, ctx, span, constraint)
    if ambiguous:
        return _condition(field, "unknown", constraint, ctx, key, span, ambiguous)
    v = ctx.value(key)
    inside = (lo is None or v >= lo) and (hi is None or v <= hi)
    return _condition(field, "pass" if inside else "fail", constraint, ctx, key, span)


def _check_income(c, ctx, span):
    cap, scheme_basis = c["income"]["max_annual_inr"], c["income"]["basis"]
    constraint = {"max_annual_inr": cap, "basis": scheme_basis}
    key = "annual_income_inr"
    if not ctx.usable(key):
        return _unusable("income", key, ctx, span, constraint)
    v, basis = ctx.value(key), income_basis(_span(ctx.evidence, key))
    ask = key if basis is None else None
    if v <= cap:
        # Family income bounds individual income, so a family figure under the cap passes any basis.
        if basis == "family" or basis == scheme_basis:
            return _condition("income", "pass", constraint, ctx, key, span, f"profile income is {basis} income")
        return _condition("income", "unknown", constraint, ctx, key, span,
                          f"under the cap, but the profile's income basis ({basis}) may not match the scheme's "
                          f"({scheme_basis})", ask=ask, ask_kind="refine")
    if scheme_basis and basis and (basis == scheme_basis or (basis == "individual" and scheme_basis == "family")):
        return _condition("income", "fail", constraint, ctx, key, span, f"profile income is {basis} income")
    return _condition("income", "unknown", constraint, ctx, key, span,
                      f"over the cap, but the bases are ambiguous (profile {basis}, scheme {scheme_basis})",
                      ask=ask, ask_kind="refine")


def _check_land(c, ctx, span):
    lo, hi = c["land"]["min_acres"], c["land"]["max_acres"]
    constraint = {"min_acres": lo, "max_acres": hi}
    key = "land_acres"
    if not ctx.usable(key):
        return _unusable("land", key, ctx, span, constraint)
    v, basis = ctx.value(key), land_basis(_span(ctx.evidence, key))
    inside = (lo is None or v >= lo) and (hi is None or v <= hi)
    if inside and basis != "leased":
        return _condition("land", "pass", constraint, ctx, key, span)
    if not inside and basis == "owned":
        return _condition("land", "fail", constraint, ctx, key, span, "land stated as owned")
    return _condition("land", "unknown", constraint, ctx, key, span,
                      f"land basis is {basis or 'not stated'}; scheme limits usually mean owned land",
                      ask=key, ask_kind="refine")


def _check_category(listed, ctx, span, can_fail):
    known, undetermined = [], []
    for cat in listed:
        field = _CATEGORY_FIELD[cat]
        if cat in CASTE_CATEGORIES:
            if ctx.usable("caste_category"):
                known.append(ctx.value("caste_category") == cat)
            else:
                undetermined.append(field)
        elif cat == "Minority":
            if ctx.usable("is_minority"):
                known.append(ctx.value("is_minority") is True)
            else:
                undetermined.append(field)
        else:  # PwD
            if ctx.usable("disability_percent") and not ctx.on_behalf:
                known.append(ctx.value("disability_percent") >= BENCHMARK_DISABILITY)
            else:
                undetermined.append(field)
    shown = {k: ctx.value(k) for k in ("caste_category", "is_minority", "disability_percent") if ctx.value(k) is not None}
    base = {"field": "category", "decisive": True, "constraint": listed, "profile_value": shown or None,
            "confidence": None, "source_span": span, "ask_field": None, "ask_kind": None}
    if any(known):
        return {**base, "result": "pass", "note": None}
    if undetermined:
        return {**base, "result": "unknown", "note": f"not determined: {sorted(set(undetermined))}",
                "ask_field": undetermined[0], "ask_kind": "missing"}
    if not can_fail:
        return {**base, "result": "unknown",
                "note": "every listed category is ruled out, but the record has several eligible groups "
                        "(multi_branch_eligibility), so category cannot fail"}
    return {**base, "result": "fail", "note": "every listed category is ruled out"}


def _check_value(field, key, allowed, ctx, span, constraint, ambiguous=None):
    """Categorical check: pass when the profile value is in allowed."""
    if not ctx.usable(key):
        return _unusable(field, key, ctx, span, constraint)
    if ambiguous:
        return _condition(field, "unknown", constraint, ctx, key, span, ambiguous)
    return _condition(field, "pass" if ctx.value(key) in allowed else "fail", constraint, ctx, key, span)


def _check_occupation(listed, ctx, span, bocw_registered):
    key = "occupation"
    if bocw_registered:
        return _condition("occupation", "pass", listed, ctx, key, span,
                          "the scheme requires construction-board registration and the person says they are registered")
    if not ctx.usable(key):
        return _unusable("occupation", key, ctx, span, listed)
    sims = [(o, _similarity(ctx.value(key), o)) for o in listed]
    best, best_sim = max(sims, key=lambda x: x[1])
    if best_sim >= OCCUPATION_THRESHOLD:
        return _condition("occupation", "pass", listed, ctx, key, span, f"matches {best!r} ({best_sim:.3f})")
    return _condition("occupation", "unknown", listed, ctx, key, span,
                      f"closest is {best!r} ({best_sim:.3f}), below {OCCUPATION_THRESHOLD}",
                      ask=key, ask_kind="refine")


def _check_stage(stage_text, ctx, span):
    admitted = stages_admitted(stage_text)
    stage = ctx.value("education_stage")
    if stage is None and ctx.value("education_class") is not None:
        stage = "school" if ctx.value("education_class") <= 10 else "higher_secondary"
    base = {"field": "education_stage", "decisive": True, "constraint": stage_text, "profile_value": stage,
            "confidence": ctx.conf("education_stage") or ctx.conf("education_class"), "source_span": span}
    if admitted is None:
        return {**base, "result": "unknown", "note": "the stage text does not map to profile stages",
                "ask_field": None, "ask_kind": None}
    if stage is None:
        return {**base, "result": "unknown", "note": "not stated", "ask_field": "education_stage", "ask_kind": "missing"}
    if stage in admitted:
        return {**base, "result": "pass", "note": None, "ask_field": None, "ask_kind": None}
    return {**base, "result": "unknown", "note": f"{stage} is not among {sorted(admitted)}; stage is fuzzy, so no fail",
            "ask_field": "education_stage", "ask_kind": "refine"}


def _residual_touches(residuals, ctx):
    """(other_fact, residual, similarity) pairs at or above RESIDUAL_TOUCH_THRESHOLD."""
    if not ctx.facts or not residuals:
        return []
    if RESIDUAL_TOUCH_THRESHOLD is None:
        raise RuntimeError("RESIDUAL_TOUCH_THRESHOLD is not set")
    sims = embed_many(ctx.facts) @ embed_many(list(residuals)).T
    return [{"other_fact": f, "residual": r, "similarity": round(float(sims[i, j]), 4)}
            for i, f in enumerate(ctx.facts) for j, r in enumerate(residuals)
            if sims[i, j] >= RESIDUAL_TOUCH_THRESHOLD]


def evaluate_scheme(slug, ctx):
    rules = ctx.rules
    meta = scheme_meta(slug)
    out = {"slug": slug, **meta, "status": None, "reason": None, "conditions": [], "preferences": [],
           "unverified_conditions": [], "to_confirm": [], "residual_touches": [], "blocking_fields": {}, "note": None}
    rec = load_record(slug)
    if rec is None or rec.get("extraction_failed"):
        out["status"] = "needs_checking"
        out["reason"] = out["note"] = ("no constraint record (no eligibility text)" if rec is None
                                       else f"constraint extraction failed: {rec.get('failure_reason', '')[:120]}")
        return out
    c, prov = rec["constraints"], rec.get("provenance") or {}
    span = lambda f: (" | ".join(prov[f]) if isinstance(prov.get(f), list) else prov.get(f))
    multi_branch = "multi_branch_eligibility" in (rec.get("parse_flags") or [])
    requires_dependent = (c.get("family") or {}).get("requires_dependent") is True
    on_behalf = ("the person is asking on behalf of someone else; their own value does not decide this"
                 if ctx.on_behalf else
                 "the scheme requires a dependent, so this may be the dependent's attribute" if requires_dependent else None)
    conds = out["conditions"]

    def list_field_is_preference(field, value):
        if "priority" in rules and is_priority_span(span(field)):
            out["preferences"].append({"field": field, "value": value, "source_span": span(field)})
            return True
        return False

    if c["age"]["min"] is not None or c["age"]["max"] is not None:
        conds.append(_check_range("age", "age", c["age"]["min"], c["age"]["max"], ctx, span("age"), c["age"], on_behalf))
    if c["income"]["max_annual_inr"] is not None:
        conds.append(_check_income(c, ctx, span("income")))
    if c["land"]["min_acres"] is not None or c["land"]["max_acres"] is not None:
        conds.append(_check_land(c, ctx, span("land")))
    edu = c["education"]
    if edu["min_class"] is not None or edu["max_class"] is not None:
        conds.append(_check_range("education_class", "education_class", edu["min_class"], edu["max_class"], ctx,
                                  span("education"), {"min_class": edu["min_class"], "max_class": edu["max_class"]}))
    if edu["stage"]:
        conds.append(_check_stage(edu["stage"], ctx, span("education")))
    if c["gender"] not in (None, "any"):
        conds.append(_check_value("gender", "gender", {c["gender"]}, ctx, span("gender"), c["gender"], on_behalf))
    if c["category"] and not list_field_is_preference("category", c["category"]):
        conds.append(_check_category(c["category"], ctx, span("category"),
                                     can_fail=not (multi_branch and "multi_branch" in rules)))
    if c["residence"] not in (None, "any"):
        conds.append(_check_value("residence", "residence", {c["residence"]}, ctx, span("residence"), c["residence"]))
    if c["marital_status"]:
        admitted = marital_admitted(c["marital_status"])
        if admitted is None:
            conds.append(_condition("marital_status", "unknown", c["marital_status"], ctx, "marital_status",
                                    span("marital_status"), "the rule's wording does not map onto marital statuses"))
        elif admitted != "any":
            conds.append(_check_value("marital_status", "marital_status", admitted, ctx, span("marital_status"),
                                      c["marital_status"], on_behalf))
    # State: the scheme record's canonical level and state.
    if meta["level"] == "central":
        conds.append(_condition("state", "pass", "central scheme", ctx, "state", span("state"), "central scheme"))
    elif meta["state"]:
        conds.append(_check_value("state", "state", {meta["state"]}, ctx, span("state"), meta["state"]))
        out["to_confirm"].append(f"resident of {meta['state']} (any domicile rule in the scheme text)")
    if c["bpl_household"] is True:
        conds.append(_check_value("bpl_household", "bpl_household", {True}, ctx, span("bpl_household"), True))
    bocw = c["requires_bocw_registration"] is True
    if bocw:
        conds.append(_check_value("registered_construction_worker", "registered_construction_worker", {True}, ctx,
                                  span("requires_bocw_registration"), True))
    if c["occupation"] and not list_field_is_preference("occupation", c["occupation"]):
        bocw_registered = ("bocw_occupation" in rules and bocw and ctx.usable("registered_construction_worker")
                           and ctx.value("registered_construction_worker") is True)
        conds.append(_check_occupation(c["occupation"], ctx, span("occupation"), bocw_registered))
    if c["not_availing_other_scheme"] is True:
        if ctx.prior_benefit:
            conds.append(_condition("not_availing_other_scheme", "unknown", True, span=span("not_availing_other_scheme"),
                                    note=f"prior benefit stated ({'; '.join(ctx.prior_benefit)}); check it is not "
                                         f"for the same purpose", decisive=True))
        else:
            conds.append(_condition("not_availing_other_scheme", "unknown", True, span=span("not_availing_other_scheme"),
                                    note="no prior benefit stated", decisive=False))
    if c["citizen_of_india"] is True:
        out["to_confirm"].append("citizen of India")

    fam = c.get("family") or {}
    if fam.get("notes"):
        out["unverified_conditions"].append(fam["notes"])
    if requires_dependent:
        out["unverified_conditions"].append("the scheme requires a dependent (see the scheme text)")
    residuals = list(c.get("residual_conditions") or [])
    out["unverified_conditions"] += residuals

    decisive = [x for x in conds if x["decisive"]]
    if any(x["result"] == "fail" for x in decisive):
        out["status"], out["reason"] = "not_eligible", "a condition fails: " + ", ".join(
            x["field"] for x in decisive if x["result"] == "fail")
    elif any(x["result"] == "unknown" for x in decisive):
        out["status"], out["reason"] = "needs_checking", "could not check: " + ", ".join(
            x["field"] for x in decisive if x["result"] == "unknown")
    else:
        out["status"], out["reason"] = "eligible", "every checked condition passes"
        if "person_level_pass" in rules and not any(x["result"] == "pass" and x["field"] != "state" for x in decisive):
            out["status"], out["reason"] = "needs_checking", NOTHING_ABOUT_YOU
    if "residual_touch" in rules:
        out["residual_touches"] = _residual_touches(residuals, ctx)
        if out["residual_touches"] and out["status"] == "eligible":
            out["status"] = "needs_checking"
            out["reason"] = "something you said may bear on a condition that is not checked automatically"
    blocking = {}
    for x in decisive:
        if x["result"] == "unknown" and x["ask_field"]:
            blocking.setdefault(x["ask_field"], x["ask_kind"])
    out["blocking_fields"] = blocking
    out["typed_condition_count"] = len(decisive)
    return out


def match(profile, other_facts, candidates, confidence=None, evidence=None, rules=RULES):
    """Evaluate candidate schemes (slugs, or retrieval results with a "slug")
    against a profile, in candidate order.

    other_facts: understand-v4 {"original", "en"} objects (the English text
    is used) or plain strings. confidence and evidence are understand()'s
    dicts; without confidence, every stated field counts as medium
    confidence. Evidence is needed to read the income and land basis.
    rules: the v2 rules to apply (all by default)."""
    unknown_rules = set(rules) - set(RULES)
    if unknown_rules:
        raise ValueError(f"unknown rules: {sorted(unknown_rules)}")
    ctx = _Context(profile, other_facts, confidence, evidence, frozenset(rules))
    results = []
    for rank, cand in enumerate(candidates, 1):
        slug = cand["slug"] if isinstance(cand, dict) else cand
        results.append({"rank": rank, **evaluate_scheme(slug, ctx)})
    return results


def select_clarifying_field(results, top=10, min_score=MIN_CLARIFY_SCORE):
    """The profile field that blocks the most of the top candidates not
    marked not_eligible, each weighted 1/rank. Blocking fields are missing
    ones and filled-but-unusable ones ("refine"). None below min_score."""
    scores, blocks, kinds = defaultdict(float), defaultdict(list), {}
    for r in results[:top]:
        if r["status"] == "not_eligible":
            continue
        for field, kind in r["blocking_fields"].items():
            scores[field] += 1.0 / r["rank"]
            blocks[field].append(r["slug"])
            kinds[field] = "refine" if "refine" in (kinds.get(field), kind) else kind
    if not scores:
        return None
    field = max(sorted(scores), key=lambda f: scores[f])
    ranked = {f: round(s, 4) for f, s in sorted(scores.items(), key=lambda kv: -kv[1])}
    if scores[field] < min_score:
        return {"field": None, "below_min_score": {"field": field, "score": round(scores[field], 4)},
                "min_score": min_score, "scores": ranked}
    return {"field": field, "kind": kinds[field], "score": round(scores[field], 4), "blocks": blocks[field],
            "min_score": min_score, "scores": ranked}

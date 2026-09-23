"""Post-processing of the frozen pipeline's output, shared by the API and
evaluation.

finalize(results, profile, other_facts) takes what the matcher and generate.py
produced for one query and returns the same entries with

- status: the matcher's, or needs_checking when an unchecked condition says
  something stricter than what was checked. Downgrade only: a status can move
  eligible -> needs_checking and never the other way, and nothing here ever
  produces not_eligible.
- reason: the generated text, plus the first surviving caveat and a count of
  the rest
- caveats: every unchecked condition of the scheme that states a requirement
  the matcher has not already checked

and ordered eligible, then needs_checking, then not_eligible, each group by
how much of the person's own situation the verdict rests on (see sort_key).

Residual conditions restate typed ones often enough ("The farmers must be
from Telangana state." beside a state check that passed) that they have to be
filtered. generate.py applies the same rule to its relevance flags from the
other side: it knows which of the person's facts a passed condition covers
(_CHECKED_KEYS) and lets the model pair the text. Reading a residual's
subject is only needed here, so those patterns live here, keyed by the same
condition fields.
"""
import re

from app.generate import _CHECKED_KEYS as CONDITION_FIELDS    # what a passed condition covers
from app.generate import _digit_numbers as digit_numbers
from app.generate import fallback_reason

STATUS_ORDER = {"eligible": 0, "needs_checking": 1, "not_eligible": 2}
# An eligible scheme whose conditions are not all checked says so, quoting the
# first unverified one.
CAVEAT = {"en": "This may not fit you: {condition}",
          "hi": "यह आपके लिए शायद ठीक न बैठे: {condition}",
          "te": "ఇది మీకు సరిపోకపోవచ్చు: {condition}"}
CAVEAT_CHARS = 160
# "... and 2 more conditions to check", after the quoted one.
MORE_CONDITIONS = {"en": "And {n} more condition{s} to check.",
                   "hi": "और {n} और शर्तें जाँचनी हैं।",
                   "te": "మరో {n} షరతులు తనిఖీ చేయాలి."}
# Used when the quote fails the grounding check and is dropped.
ONLY_COUNT = {"en": "{n} condition{s} of this scheme still have to be checked.",
              "hi": "इस योजना की {n} शर्तें अभी जाँचनी बाकी हैं।",
              "te": "ఈ పథకం {n} షరతులు ఇంకా తనిఖీ చేయాలి."}
_CONDITION_LEAD = re.compile(r"^(?:note\s*\d*\s*[:.\-]\s*|\d{1,2}[.)]\s*|[a-z][.)]\s*)", re.I)
# Residual conditions include descriptions and objectives; only a sentence
# that states a requirement is worth showing as a caveat.
_REQUIREMENT = re.compile(r"\b(?:must|should|shall|required|only|cannot|excluded|at least|minimum|maximum|"
                          r"not more than|not less than|up to|ineligible|not eligible|will not|shall not)\b", re.I)
# "Dwarfs are also eligible for this scheme." grants eligibility instead of
# restricting it, so it is not a caveat.
_INCLUSION = re.compile(r"\b(?:is|are|shall be|will be)\s+also\s+eligible\b", re.I)
# Kinds of stricter number that can be compared against the person's profile,
# and the profile field each one is about.
NUMERIC_KINDS = {"age": "age", "income": "annual_income_inr", "land": "land_acres",
                 "category": "disability_percent"}
CONFIDENT = ("high", "medium")
PLAUSIBLE = {"age": (1, 120), "disability_percent": (1, 100), "land_acres": (0, 100000),
             "annual_income_inr": (1000, 10 ** 9)}
_RANGE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:-|–|—|to|and)\s*(\d+(?:\.\d+)?)", re.I)
_LOWER = re.compile(r"(?:(at least|not less than|minimum(?: of)?|more than|above|over|greater than|exceeding)\s*"
                    r"(?:rs\.?|₹)?\s*(\d+(?:[.,]\d+)?)|(\d+(?:[.,]\d+)?)\s*%?\s*(?:and|or)\s+(more|above))", re.I)
_UPPER = re.compile(r"(not more than|no more than|not exceeding|maximum(?: of)?|up to|less than|below|under|within)\s*"
                    r"(?:rs\.?|₹)?\s*(\d+(?:[.,]\d+)?)", re.I)
_EXCLUSIVE_LOWER = ("more than", "above", "over", "greater than", "exceeding")
_EXCLUSIVE_UPPER = ("less than", "below", "under")
_SUBJECTS = {
    "state": re.compile(r"\b(?:resident|residents|residing|residence|domicile[ds]?|domiciled|native|state)\b", re.I),
    "age": re.compile(r"\b(?:age|aged|ages|age group|years? of age|years old)\b", re.I),
    "category": re.compile(r"\b(?:sc|st|obc|ews|scheduled caste|scheduled tribe|backward class(?:es)?|minority|"
                           r"minorities|caste|disabilit(?:y|ies)|disabled|divyang|handicapped)\b", re.I),
    "land": re.compile(r"\b(?:land|lands|landholding|acres?|hectares?|cultivable)\b", re.I),
    "income": re.compile(r"\b(?:income|salary|salaries|earnings?|wages?)\b", re.I),
    "bpl_household": re.compile(r"\b(?:bpl|below poverty line|poverty line)\b", re.I),
    "marital_status": re.compile(r"\b(?:married|unmarried|widow(?:ed|er)?s?|divorced|separated|abandoned)\b", re.I),
    "residence": re.compile(r"\b(?:rural|urban|village|town|city)\b", re.I),
}
assert set(_SUBJECTS) <= set(CONDITION_FIELDS), "unknown condition field in _SUBJECTS"


def _norm_text(text):
    return re.sub(r"[^a-z0-9 ]+", " ", re.sub(r"\s+", " ", (text or "").lower())).strip()


def _constraint_numbers(constraint):
    """Every number a constraint states ({"min": 18, "max": 60} -> 18, 60)."""
    found = []

    def walk(value):
        if isinstance(value, bool) or value is None:
            return
        if isinstance(value, (int, float)):
            found.append(float(value))
        elif isinstance(value, str):
            found.extend(digit_numbers(value))
        elif isinstance(value, dict):
            for v in value.values():
                walk(v)
        elif isinstance(value, (list, tuple)):
            for v in value:
                walk(v)

    walk(constraint)
    return found


def _numbers_covered(residual, condition):
    """False when the residual states a number the condition's own constraint
    does not, which means it says something stricter or narrower than what was
    checked ("disability of 80% or more" against a PwD check made at 40%)."""
    allowed = _constraint_numbers(condition["constraint"])
    return all(any(abs(n - a) < 0.01 for a in allowed) for n in digit_numbers(residual))


def is_requirement(condition):
    """True for a condition that restricts who qualifies, rather than
    describing the scheme or granting eligibility to another group. The
    inclusion clause is removed before the test, so "Dwarfs are also eligible"
    is not a requirement while "... must have passed ... however, students
    promoted ... are also eligible" still is."""
    return bool(_REQUIREMENT.search(_INCLUSION.sub(" ", condition or "")))


def classify_residual(residual, result):
    """(kind, field) for one unverified condition, against what was checked:

    "restatement"  the ground is already covered by a condition that passed
    "stricter"     same subject, but it states a number the check did not
    "other"        something no condition touched
    """
    normalized = _norm_text(residual)
    stricter_field = None
    for c in result["conditions"]:
        if not (c["decisive"] and c["result"] == "pass"):
            continue
        span = _norm_text(c["source_span"])
        if span and (span in normalized or normalized in span):
            return "restatement", c["field"]
        same_subject = (c["field"] == "occupation"
                        and any(_norm_text(o) in normalized for o in (c["constraint"] or [])))
        pattern = _SUBJECTS.get(c["field"])
        same_subject = same_subject or bool(pattern and pattern.search(residual or ""))
        if not same_subject:
            continue
        if _numbers_covered(residual, c):
            return "restatement", c["field"]
        stricter_field = stricter_field or c["field"]
    return ("stricter", stricter_field) if stricter_field else ("other", None)


def _plausible(number, key):
    low, high = PLAUSIBLE[key]
    return low <= number <= high


def stricter_bound(residual, key):
    """The bound a residual states for a profile field, as
    (low, low_inclusive, high, high_inclusive), or None when it cannot be read
    with confidence."""
    text = re.sub(r"\s+", " ", residual or "")
    numbers = [n for n in digit_numbers(text) if _plausible(n, key)]
    if not numbers:
        return None
    for m in _RANGE.finditer(text):
        low, high = float(m.group(1)), float(m.group(2))
        if low < high and _plausible(low, key) and _plausible(high, key):
            return low, True, high, True
    low = low_inclusive = high = high_inclusive = None
    if m := _LOWER.search(text):
        phrase = (m.group(1) or "").lower()
        value = float((m.group(2) or m.group(3)).replace(",", ""))
        if _plausible(value, key):
            low, low_inclusive = value, phrase not in _EXCLUSIVE_LOWER
    if m := _UPPER.search(text):
        phrase = m.group(1).lower()
        value = float(m.group(2).replace(",", ""))
        if _plausible(value, key):
            high, high_inclusive = value, phrase not in _EXCLUSIVE_UPPER
    if low is None and high is None:
        return None                      # a number, but no readable bound
    return low, bool(low_inclusive), high, bool(high_inclusive)


def satisfies(value, bound):
    low, low_inclusive, high, high_inclusive = bound
    if low is not None and not (value >= low if low_inclusive else value > low):
        return False
    if high is not None and not (value <= high if high_inclusive else value < high):
        return False
    return True


def _is_settled(residual, field, profile, confidence):
    """True when the person's own answer already satisfies a stricter bound,
    so the residual says nothing new (rule b)."""
    key = NUMERIC_KINDS.get(field)
    if key is None:
        return False
    value, conf = profile.get(key), (confidence or {}).get(key)
    if value is None or conf not in CONFIDENT:
        return False                     # missing or unsure: it has to be checked (rule c)
    bound = stricter_bound(residual, key)
    return bool(bound) and satisfies(value, bound)


def caveats_for(result, profile, confidence):
    """([conditions to show], downgrade) for one scheme: the unverified
    conditions that state a requirement nothing has checked, and whether one
    of them is a stricter bound that the person's answers do not settle."""
    shown, downgrade = [], False
    for residual in result["unverified_conditions"]:
        if not is_requirement(residual):
            continue
        kind, field = classify_residual(residual, result)
        if kind == "restatement":
            continue
        if kind == "stricter":
            if _is_settled(residual, field, profile, confidence):
                continue                 # (b) the person's value meets it
            if field in NUMERIC_KINDS:   # (c) unknown or unreadable, (d) violated
                downgrade = True
        shown.append(residual)
    return shown, downgrade


def quotable(condition):
    """The condition as it would be quoted: no leading numbering, truncated."""
    text = _CONDITION_LEAD.sub("", re.sub(r"\s+", " ", condition or "").strip())
    return text[:CAVEAT_CHARS].rsplit(" ", 1)[0].rstrip(" .,;:") + "…" if len(text) > CAVEAT_CHARS else text


def is_grounded(quote, sources):
    """Every quoted caveat has to be an exact substring of the scheme's own
    text; a quote that is not is dropped rather than shown."""
    needle = re.sub(r"\s+", " ", quote or "").rstrip("…").strip()
    return bool(needle) and any(needle in re.sub(r"\s+", " ", source or "") for source in sources)


def caveat(condition, language):
    """"This may not fit you: <the scheme's first unchecked condition>"."""
    return CAVEAT[language if language in CAVEAT else "en"].format(condition=quotable(condition))


def _caveat_sentences(shown, language, sources):
    """The caveat sentence and the count of the rest, as text to append."""
    lang = language if language in CAVEAT else "en"
    quote = quotable(shown[0])
    if is_grounded(quote, sources):
        parts = [CAVEAT[lang].format(condition=quote)]
        if len(shown) > 1:
            parts.append(MORE_CONDITIONS[lang].format(n=len(shown) - 1, s="" if len(shown) == 2 else "s"))
    else:                                # the quote does not appear in the scheme text
        parts = [ONLY_COUNT[lang].format(n=len(shown), s="" if len(shown) == 1 else "s")]
    return " ".join(parts)


def sort_key(result, status, rank):
    """Within a status group, the more of the person's own situation a verdict
    rests on, the higher it ranks: first schemes with a condition about the
    person that was actually checked (state and board registration are not:
    registration is a gate into a board's schemes, not a check of this
    scheme), then the fewest conditions left unverified, then the most checks
    passed, then retrieval order."""
    passes = [c for c in result["conditions"] if c["decisive"] and c["result"] == "pass"]
    personal = [c for c in passes if c["field"] != "state" and not c.get("gate")]
    return (STATUS_ORDER.get(status, 3), 0 if personal else 1,
            len(result["unverified_conditions"]), -len(personal), rank)


def finalize(results, profile, other_facts=(), *, confidence=None, language="en"):
    """results: one dict per scheme, each with

        "match":     the matcher result (matcher.match() entry)
        "status":    the status to show (the matcher's, or generate.py's after
                     a relevance downgrade)
        "reason":    the generated or code-written explanation
        "citations": the explanation's citations

    Returns copies of those dicts, with "status", "reason" and "caveats" set,
    in display order. profile, other_facts and confidence come from
    understand(); language selects the wording of anything written here.
    """
    finalized = []
    for item in results:
        m = item["match"]
        status, reason = item["status"], item["reason"]
        # Caveats are listed for anything still in play; only an eligible
        # scheme can be downgraded by one (nothing here moves a status up, or
        # to not_eligible).
        caveats, downgrade = ([], False) if status == "not_eligible" else caveats_for(m, profile, confidence)
        downgrade = downgrade and status == "eligible"
        if downgrade:
            # eligible -> needs_checking only, and the text is rebuilt the way
            # other code-written reasons are, so no "you meet the ..." is left.
            status = "needs_checking"
            reason, _ = fallback_reason({**m, "status": status}, language)
        if caveats and (status == "eligible" or downgrade):
            # The generated needs_checking text already says what to confirm,
            # so only an eligible or freshly downgraded reason gets the line.
            sources = list(m["unverified_conditions"]) + [item.get("eligibility_text") or ""]
            reason = f"{reason} {_caveat_sentences(caveats, language, sources)}"
        finalized.append((sort_key(m, status, m["rank"]),
                          {**item, "status": status, "reason": reason, "caveats": caveats}))
    return [entry for _, entry in sorted(finalized, key=lambda pair: pair[0])]

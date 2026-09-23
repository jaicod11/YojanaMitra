"""Post-processing of the frozen pipeline's output, shared by the API and
evaluation.

finalize(results, profile, other_facts) takes what the matcher and generate.py
produced for one query and returns the same entries with

- status: the matcher's, or needs_checking when an unchecked condition says
  something stricter than what was checked. Downgrade only: a status can move
  eligible -> needs_checking and never the other way, and nothing here ever
  produces not_eligible.
- reason: the generated text plus a count of the conditions still to
  confirm; for a scheme downgraded by a stricter bound, a code-written reason
  that names that condition instead (see downgrade_reason)
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
from app.generate import _inr, _join, _label, _lang, _num, _passed, fallback_reason

STATUS_ORDER = {"eligible": 0, "needs_checking": 1, "not_eligible": 2}
CAVEAT_CHARS = 160
# A scheme with unchecked conditions says how many; the conditions themselves
# are in "caveats". {noun} is singular or plural for n.
NOUN = {"en": ("condition", "conditions"), "hi": ("शर्त", "शर्तों"), "te": ("షరతు", "షరతులు")}
COUNT = {"en": "{n} {noun} of this scheme still to confirm.",
         "hi": "इस योजना की {n} {noun} की पुष्टि अभी बाकी है।",
         "te": "ఈ పథకంలో {n} {noun} ఇంకా నిర్ధారించాలి."}
COUNT_OTHER = {"en": "{n} other {noun} of this scheme still to confirm.",
               "hi": "इस योजना की {n} और {noun} की पुष्टि अभी बाकी है।",
               "te": "ఈ పథకంలో మరో {n} {noun} ఇంకా నిర్ధారించాలి."}
# A scheme downgraded by a stricter bound names that condition: with the
# person's own value when they gave one (rule d, or an unreadable bound), and
# as a plain requirement when they did not (rule c). Neither says they fail.
SAID_VALUE = {
    "en": {"age": "You said your age is {v}", "annual_income_inr": "You said your income is ₹{v} a year",
           "land_acres": "You said you have {v} acres of land", "disability_percent": "You said your disability is {v}%"},
    "hi": {"age": "आपने अपनी आयु {v} वर्ष बताई है", "annual_income_inr": "आपने अपनी वार्षिक आय ₹{v} बताई है",
           "land_acres": "आपने {v} एकड़ ज़मीन बताई है", "disability_percent": "आपने अपनी विकलांगता {v}% बताई है"},
    "te": {"age": "మీరు మీ వయస్సు {v} సంవత్సరాలు అని చెప్పారు",
           "annual_income_inr": "మీరు మీ వార్షిక ఆదాయం ₹{v} అని చెప్పారు",
           "land_acres": "మీకు {v} ఎకరాల భూమి ఉందని చెప్పారు", "disability_percent": "మీరు మీ వైకల్యం {v}% అని చెప్పారు"},
}
OWN_TEXT = {"en": "{said}; this scheme's own text says: {quote}",
            "hi": "{said}; इस योजना के अपने पाठ में लिखा है: {quote}",
            "te": "{said}; ఈ పథకం సొంత పాఠం ఇలా చెబుతోంది: {quote}"}
REQUIRES = {"en": "This scheme requires: {quote} That couldn't be checked from what you told us.",
            "hi": "यह योजना माँगती है: {quote} आपकी बताई जानकारी से इसकी जाँच नहीं हो सकी।",
            "te": "ఈ పథకానికి ఇది అవసరం: {quote} మీరు చెప్పిన వివరాలతో దీన్ని తనిఖీ చేయలేకపోయాము."}
# When the condition's text fails the grounding check it is named, not quoted.
OWN_TEXT_UNQUOTED = {"en": "{said}; this scheme's own text sets its own {label} condition, which has to be confirmed.",
                     "hi": "{said}; इस योजना के पाठ में {label} की अपनी शर्त है, जिसकी पुष्टि ज़रूरी है।",
                     "te": "{said}; ఈ పథకం పాఠంలో {label}కు సొంత షరతు ఉంది, దాన్ని నిర్ధారించాలి."}
REQUIRES_UNQUOTED = {"en": "This scheme has a {label} condition that couldn't be checked from what you told us.",
                     "hi": "इस योजना में {label} की एक शर्त है, जिसकी जाँच आपकी बताई जानकारी से नहीं हो सकी।",
                     "te": "ఈ పథకంలో {label}కు ఒక షరతు ఉంది, మీరు చెప్పిన వివరాలతో దాన్ని తనిఖీ చేయలేకపోయాము."}
CHECKED = {"en": "Checked: {passed}.", "hi": "जाँची गई शर्तें: {passed}।", "te": "తనిఖీ చేసినవి: {passed}."}
TRIGGER_LABEL = {"en": {"age": "age", "income": "income", "land": "land", "category": "disability"},
                 "hi": {"age": "आयु", "income": "आय", "land": "ज़मीन", "category": "विकलांगता"},
                 "te": {"age": "వయస్సు", "income": "ఆదాయం", "land": "భూమి", "category": "వైకల్యం"}}
_LEADING_JUNK = re.compile(r"^[\s:;,.\-–—\"'“”‘’]+")
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
    """([conditions to show], trigger) for one scheme: the unverified
    conditions that state a requirement nothing has checked, and the first of
    them that is a stricter bound the person's answers do not settle (None if
    there is none). trigger is {"residual", "field", "key", "value"}; value is
    the person's own answer when they gave a confident one."""
    shown, trigger = [], None
    for residual in result["unverified_conditions"]:
        if not is_requirement(residual):
            continue
        kind, field = classify_residual(residual, result)
        if kind == "restatement":
            continue
        if kind == "stricter":
            if _is_settled(residual, field, profile, confidence):
                continue                 # (b) the person's value meets it
            if field in NUMERIC_KINDS and trigger is None:   # (c) unknown or unreadable, (d) violated
                key = NUMERIC_KINDS[field]
                known = profile.get(key) is not None and (confidence or {}).get(key) in CONFIDENT
                trigger = {"residual": residual, "field": field, "key": key,
                           "value": profile.get(key) if known else None}
        shown.append(residual)
    return shown, trigger


def quotable(condition):
    """The condition as it would be quoted: no leading numbering, punctuation,
    colons or quote marks, and truncated."""
    text = _LEADING_JUNK.sub("", _CONDITION_LEAD.sub("", re.sub(r"\s+", " ", condition or "").strip()))
    return text[:CAVEAT_CHARS].rsplit(" ", 1)[0].rstrip(" .,;:") + "…" if len(text) > CAVEAT_CHARS else text


def is_grounded(quote, sources):
    """Every quoted caveat has to be an exact substring of the scheme's own
    text; a quote that is not is dropped rather than shown."""
    needle = re.sub(r"\s+", " ", quote or "").rstrip("…").strip()
    return bool(needle) and any(needle in re.sub(r"\s+", " ", source or "") for source in sources)


def count_sentence(n, language, other=False):
    """"N condition(s) of this scheme still to confirm." (or "N other ...")."""
    lang = _lang(language)
    noun = NOUN[lang][0 if n == 1 else 1]
    return (COUNT_OTHER if other else COUNT)[lang].format(n=n, noun=noun)


def _value_text(key, value):
    return _inr(value) if key == "annual_income_inr" else _num(value)


def _as_sentence(quote):
    return quote if quote.endswith((".", "…", "।", "?", "!")) else quote + "."


def downgrade_reason(result, trigger, others, language, sources):
    """The reason for a scheme a stricter bound moved to needs_checking,
    written in code: the condition that did it (quoted when it is grounded in
    the scheme's text), the checks that did pass except that field, and a
    count of the other conditions still to confirm."""
    lang = _lang(language)
    quote = quotable(trigger["residual"])
    label = TRIGGER_LABEL[lang][trigger["field"]]
    grounded = is_grounded(quote, sources)
    if trigger["value"] is not None:
        said = SAID_VALUE[lang][trigger["key"]].format(v=_value_text(trigger["key"], trigger["value"]))
        first = (OWN_TEXT[lang].format(said=said, quote=_as_sentence(quote)) if grounded
                 else OWN_TEXT_UNQUOTED[lang].format(said=said, label=label))
    else:
        first = (REQUIRES[lang].format(quote=_as_sentence(quote)) if grounded
                 else REQUIRES_UNQUOTED[lang].format(label=label))
    parts = [first]
    checked = [_label(c, lang) for c in _passed(result) if c["field"] != trigger["field"]]
    if checked:
        parts.append(CHECKED[lang].format(passed=_join(checked, lang)))
    if others:
        parts.append(count_sentence(others, language, other=True))
    return " ".join(parts)


def sort_key(result, status, rank):
    """Within a status group, schemes where a condition about the person was
    actually checked come first (state and board registration are not:
    registration is a gate into a board's schemes, not a check of this
    scheme), then retrieval order."""
    passes = [c for c in result["conditions"] if c["decisive"] and c["result"] == "pass"]
    personal = [c for c in passes if c["field"] != "state" and not c.get("gate")]
    return (STATUS_ORDER.get(status, 3), 0 if personal else 1, rank)


def collect(matched, explained, language, eligibility_text=None):
    """The entries finalize() takes, one per matched scheme in matcher order:
    explain()'s text where it covered the scheme, generate.py's code template
    for the rest. eligibility_text(slug) supplies the scheme text quotes are
    checked against. The API and evaluation both build their input here."""
    explanations = {e["slug"]: e for e in explained["schemes"]}
    items = []
    for m in matched:
        e = explanations.get(m["slug"])
        if e is None:                                # outside the five explain() covers
            reason, citations = fallback_reason(m, language)
            status = m["status"]
        else:
            reason, citations, status = e["reason"], e["citations"], e["status"]
        items.append({"match": m, "explanation": e, "status": status, "reason": reason, "citations": citations,
                      "eligibility_text": eligibility_text(m["slug"]) if eligibility_text else None})
    return items


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
        caveats, trigger = ([], None) if status == "not_eligible" else caveats_for(m, profile, confidence)
        if trigger and status == "eligible":
            # eligible -> needs_checking only. The generated text claimed the
            # checks passed, so the reason is rewritten in code around the
            # condition that caused the downgrade.
            status = "needs_checking"
            sources = list(m["unverified_conditions"]) + [item.get("eligibility_text") or ""]
            reason = downgrade_reason(m, trigger, len(caveats) - 1, language, sources)
        elif caveats:
            reason = f"{reason} {count_sentence(len(caveats), language)}"
        finalized.append((sort_key(m, status, m["rank"]),
                          {**item, "status": status, "reason": reason, "caveats": caveats}))
    return [entry for _, entry in sorted(finalized, key=lambda pair: pair[0])]

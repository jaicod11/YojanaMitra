"""Query understanding: one LLM call turns what a person says into

- search_query_en: the text used for retrieval. For English input it is the
            person's text exactly, set in code with no LLM involvement. For
            other languages it is a faithful sentence-by-sentence English
            translation that keeps every fact, number, duration, negation and
            scheme name, with nothing summarised or added.
- profile:  the gold set's eight expected_profile keys (occupation, state, age,
            gender, annual_income_inr, land_acres, caste_category, family),
            null when not stated, plus other_facts: short verbatim statements
            that could decide eligibility but fit no typed field (durations,
            loan history, benefits already received, registration status,
            employment type)
- confidence / evidence: for every non-null typed field, high|medium|low and
            the span(s) of the text it comes from
- clarifying_question / clarifying_field: at most one question, about the
            single most decisive missing field, and which field that is

The call goes through the provider pool from scripts/label_categories.py
(Gemini flash-lite first, Groq fallback, key rotation on daily caps,
temperature 0). The answer is validated before use:

- unknown keys and bad enums are rejected
- every evidence span and other_facts entry must occur in the input (for
  non-English input, it may instead be a piece of the translation)
- age, income and land need evidence containing a number, and the value must
  follow from it using only conversions the evidence states (lakh, thousand,
  crore, monthly x12, hectares); otherwise the field must be null
- a translation must keep every number and any negation in the source, must
  not merge the text into fewer sentences or rephrase it as a third-person
  summary, and must not add numbers, states or scheme names
- the clarifying question must be a single question about one field that is
  missing or low-confidence

A rejected answer is retried once with the problems listed. If the retry also
fails, the result has status "invalid" and an empty profile; for English input
search_query_en is still the person's text.

Every result is cached in data/cache/understand.jsonl, keyed by (text,
language, prompt version), so evaluation is deterministic and never spends
quota twice. Older prompt versions are retired: their cached results can still
be read, but no new calls are made with them. Provider failures (quota,
network) raise UnderstandError and are not cached.
"""
import hashlib
import json
import logging
import re
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import label_categories as lc  # noqa: E402  provider pool shared with the labelling pipeline

PROMPT_VERSION = "understand-v2"
CACHE_PATH = ROOT / "data" / "cache" / "understand.jsonl"
SCHEMES_DIR = ROOT / "data" / "interim" / "schemes"

PROFILE_KEYS = ("occupation", "state", "age", "gender", "annual_income_inr",
                "land_acres", "caste_category", "family")
STATES = (
    "Andaman and Nicobar Islands", "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chandigarh",
    "Chhattisgarh", "Dadra and Nagar Haveli and Daman and Diu", "Delhi", "Goa", "Gujarat", "Haryana",
    "Himachal Pradesh", "Jammu and Kashmir", "Jharkhand", "Karnataka", "Kerala", "Ladakh", "Lakshadweep",
    "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Puducherry",
    "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand",
    "West Bengal",
)
GENDERS = ("female", "male", "transgender")
CASTES = ("SC", "ST", "OBC", "EWS", "General")
CONFIDENCE = ("high", "medium", "low")
LANGUAGE_NAMES = {"en": "English", "hi": "Hindi", "te": "Telugu", "ta": "Tamil", "bn": "Bengali"}
MAX_OTHER_FACTS = 8


class UnderstandError(RuntimeError):
    """No answer available: no provider could answer, or a retired prompt
    version was asked for a text it has no cached result for."""


PROMPT = """You help Indian citizens find government welfare schemes. Read what one person wrote and return a JSON object describing them.

The person wrote in {language_name}:
<<<
{text}
>>>

Return ONLY a JSON object with exactly these keys, no markdown: {keys}.
{translation_rules}
"profile": an object with exactly these nine keys:
  "occupation": short lowercase English noun phrase for their work or role, e.g. "farmer", "tenant farmer", "student", "construction worker", "street vendor"; null if not stated.
  "state": the Indian state or union territory, written exactly as one of: {states}. Fill it when they name the state, or a city or district that clearly belongs to one state; otherwise null.
  "age": the person's own age in whole years (integer), only if they state a number; otherwise null.
  "gender": "female", "male" or "transgender", only when they describe themself with a gendered word (woman, widow, mother, pregnant, girl, man...). Do not infer it from a spouse or from grammatical gender. Otherwise null.
  "annual_income_inr": their personal or family income per year in rupees (integer), only if they state an amount. Convert only what they state: lakh (x 100000), thousand (x 1000), a monthly amount (x 12). Never estimate; never write 0 unless they say their income is zero. Otherwise null.
  "land_acres": land they own or cultivate in acres (number), only if they state an amount; convert hectares (x 2.471). If the unit varies by region (bigha, kanal, guntha), null.
  "caste_category": "SC", "ST", "OBC", "EWS" or "General", only when stated; otherwise null.
  "family": a short snake_case summary of household facts that matter for eligibility (marital status, children and their ages or classes, BPL card, disability percentage, pregnancy, a death in the family), e.g. "widow_bpl" or "two_children_in_school"; null if none.
  "other_facts": a list of short statements copied word for word from {fact_source} that could decide eligibility but fit none of the eight fields above: how long something has lasted, loans taken or never taken, benefits or schemes already received, registration or membership status, type of employment (permanent, contract, government). Include negations as written. An empty list if there are none. Do not repeat what the typed fields already hold.

"confidence": an object with one entry for every non-null typed field (not other_facts): "high" if stated plainly, "medium" if clearly implied, "low" if uncertain.

"evidence": an object with one entry for every non-null typed field (not other_facts): the words the value comes from, copied character for character from {fact_source}. Use a list of strings when the value comes from separate parts of the text. For age, income and land the evidence must contain the number.

"clarifying_question": if a fact that decides which schemes fit what they are asking is missing or low-confidence, one short question about the single most decisive such fact, in the language they wrote in. Ask about one thing only: never two things joined by "and". null if nothing decisive is missing.

"clarifying_field": the field that question is about: one of the eight typed fields, or "other" for a fact outside them. null when clarifying_question is null.

{example}
"""

TRANSLATION_RULES = """
"search_query_en": a faithful English translation of the person's text, sentence by sentence, in the same order and the same person ("I", "my"). Keep every fact, number, amount, duration, negation ("never", "not", "no") and scheme name exactly; keep amounts as they wrote them (e.g. "1.8 lakh"). Do not summarise, do not add anything (no "looking for government schemes"), do not drop anything.
"""

EXAMPLE_EN = """Example. Person wrote: "I run a small tea stall in Pune and want a loan to expand it. I've run it for 6 years and have never taken a bank loan. I'm 34, my wife and I earn about 1.2 lakh a year, and we have two school-going kids."
{"profile": {"occupation": "tea stall owner", "state": "Maharashtra", "age": 34, "gender": null, "annual_income_inr": 120000, "land_acres": null, "caste_category": null, "family": "married_two_school_children", "other_facts": ["I've run it for 6 years", "have never taken a bank loan"]},
 "confidence": {"occupation": "high", "state": "high", "age": "high", "annual_income_inr": "medium", "family": "high"},
 "evidence": {"occupation": "I run a small tea stall", "state": "Pune", "age": "I'm 34", "annual_income_inr": "earn about 1.2 lakh a year", "family": "we have two school-going kids"},
 "clarifying_question": null, "clarifying_field": null}"""

EXAMPLE_TRANSLATED = """Example. Person wrote (Hindi): "मैं इंदौर में साइकिल रिपेयर की दुकान चलाता हूँ और महीने के 8 हज़ार कमाता हूँ। मैंने पहले कभी लोन नहीं लिया।"
{"search_query_en": "I run a bicycle repair shop in Indore and earn 8 thousand a month. I have never taken a loan before.",
 "profile": {"occupation": "bicycle repair shop owner", "state": "Madhya Pradesh", "age": null, "gender": null, "annual_income_inr": 96000, "land_acres": null, "caste_category": null, "family": null, "other_facts": ["I have never taken a loan before"]},
 "confidence": {"occupation": "high", "state": "high", "annual_income_inr": "high"},
 "evidence": {"occupation": "I run a bicycle repair shop", "state": "इंदौर", "annual_income_inr": "earn 8 thousand a month"},
 "clarifying_question": "आपकी उम्र कितनी है?", "clarifying_field": "age"}"""

RETRY_SUFFIX = """
Your previous answer was rejected for these reasons:
{errors}
Return a corrected JSON object only."""


def build_prompt(text, language):
    translated = language != "en"
    keys = (["search_query_en"] if translated else []) + ["profile", "confidence", "evidence",
                                                         "clarifying_question", "clarifying_field"]
    return PROMPT.format(
        language_name=LANGUAGE_NAMES.get(language, language), text=text.strip(),
        keys=", ".join(f'"{k}"' for k in keys),
        translation_rules=TRANSLATION_RULES if translated else "",
        fact_source="the person's text or from your English translation" if translated else "the person's text",
        states=", ".join(STATES),
        example=EXAMPLE_TRANSLATED if translated else EXAMPLE_EN,
    )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

_NUM_RE = re.compile(r"\d+(?:[.,]\d+)*")       # \d also matches Devanagari/Telugu/Tamil/Bengali digits
_INDIC_RE = re.compile("[" + chr(0x0900) + "-" + chr(0x0DFF) + "]")   # Devanagari through Sinhala blocks
_WORD_NUMBERS = {
    "half": 0.5, "quarter": 0.25, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100,
}
# Units a person might state. A conversion is allowed only when its unit word
# appears in the evidence.
_UNITS = {
    "lakh": (re.compile(r"lakh|\blacs?\b|लाख|లక్ష|লাখ|லட்ச", re.I), 1e5),
    "crore": (re.compile(r"crore|करोड़|కోటి|কোটি|கோடி", re.I), 1e7),
    "thousand": (re.compile(r"thousand|हज़ार|हजार|వేల|వెయ్య|হাজার|ஆயிர", re.I), 1e3),
}
_MONTH = re.compile(r"month|महीन|माह|मासिक|నెల|মাস|மாத", re.I)
_HECTARE = re.compile(r"hectare|हेक्टेयर|హెక్టార|হেক্টর|ஹெக்டே", re.I)
# Conversions a translation may apply to a source number (units kept or expanded).
_RATIOS = (1, 1e3, 1e5, 1e7, 12, 2.471, 1e-3, 1e-5, 1e-7, 1 / 12, 1 / 2.471, 1e5 * 12, 1e3 * 12)
_SCHEME_WORDS = re.compile(r"\b(?:yojana|yojna|yojane|abhiyan|abhiyaan|pradhan\s+mantri|mukhya\s*mantri|pm[\s-]+[a-z]{3,})\b",
                           re.IGNORECASE)
# The person asked about a named scheme themself, so the text may name it.
_USER_NAMES_SCHEME = re.compile(r"yojana|yojna|scheme|nidhi|card|योजना|निधि|कार्ड|యోజన|పథకం|కార్డ్|నిధి|திட்டம்|যোজনা|প্রকল্প",
                                re.IGNORECASE)
# Everyday acronyms that are not scheme names.
_PLAIN_ACRONYMS = {"SC", "ST", "OBC", "EWS", "BPL", "APL", "PWD", "PwD", "ITI", "NRI", "LPG", "SHG", "MSME",
                   "PSU", "UT", "ID", "NGO", "HIV", "AIDS", "OBCs", "DNT", "EBC", "SEBC", "MBC", "BC", "VJNT",
                   "NT", "SBC", "PG", "UG", "ASHA", "RTE"}
_EN_NEGATION = re.compile(r"\b(?:no|not|never|none|nothing|without|neither|nor|cannot)\b|n't", re.I)
_SOURCE_NEGATION = {
    "hi": re.compile(r"नहीं|नही|कभी न|बिना"),
    "te": re.compile(r"లేదు|లేక|కాదు|లేని|లేకుండా|ఎప్పుడూ"),
    "ta": re.compile(r"இல்லை|இல்லாம|அல்ல"),
    "bn": re.compile(r"না|নেই|ছাড়া"),
}
_SUMMARY_STYLE = re.compile(r"^\s*(?:a|an|the)\s+(?:\d+[- ]year[- ]old\s+)?(?:person|individual|user|applicant|people)\b"
                            r"|\b(?:is|are)\s+(?:looking|searching|seeking)\s+for\b", re.I)
_CONJUNCTION = re.compile(r"\band\b|और|तथा|एवं|మరియు|மற்றும்|এবং", re.I)


def _norm(s):
    return re.sub(r"\s+", " ", (s or "").casefold()).strip(" \"'“”‘’.,;:!?।")


def _numbers(text):
    """Digits (in any script) and English number words."""
    out = []
    for m in _NUM_RE.finditer(text or ""):
        s = m.group()
        # 1,80,000 / 90,000 are digit grouping; 2.5 is a decimal.
        s = s.replace(",", "") if re.fullmatch(r"\d{1,3}(?:,\d{2,3})+", s) else s.replace(",", ".")
        try:
            out.append(float(s))
        except ValueError:
            pass
    words = re.findall(r"[a-z]+", (text or "").lower())
    i = 0
    while i < len(words):
        if words[i] in _WORD_NUMBERS:
            value = _WORD_NUMBERS[words[i]]
            nxt = _WORD_NUMBERS.get(words[i + 1]) if i + 1 < len(words) else None
            if value in (20, 30, 40, 50, 60, 70, 80, 90) and nxt is not None and 1 <= nxt <= 9:
                value += nxt                                   # "seventy five"
                i += 1
            out.append(float(value))
        i += 1
    return out


def _close(a, b):
    return abs(a - b) <= max(0.02 * abs(b), 0.011)


def _derivable(value, numbers, ratios=_RATIOS):
    return any(_close(value, n * r) for n in numbers for r in ratios)


def _sentence_count(text):
    text = re.sub(r"\b(?:Rs|e\.g|i\.e|Dr|Mr|Mrs|No)\.", "", text or "")
    return len([p for p in re.split(r"[.!?।॥]+(?:\s+|$)", text) if p.strip()])


@lru_cache(maxsize=1)
def _scheme_acronyms():
    """Acronyms written in parentheses in corpus scheme names (PMEGP, IGNOAPS,
    HBOCWWB...) plus slugs that double as acronyms (pmegp, pmjdy). Generic
    names such as "Old Age Pension" are not checked: they also read as plain
    descriptions of a need."""
    found = set()
    for p in SCHEMES_DIR.glob("*.json"):
        name = json.loads(p.read_text(encoding="utf-8")).get("scheme_name") or ""
        for inner in re.findall(r"\(([^)]*)\)", name):
            for tok in re.split(r"[\s/:,]+", inner):
                tok = tok.strip("-.")
                if len(tok) >= 3 and sum(c.isupper() for c in tok) >= 2:
                    found.add(tok)
    found |= {p.stem.replace("-", "").upper() for p in SCHEMES_DIR.glob("*.json")
              if len(p.stem.replace("-", "")) >= 4 and p.stem.replace("-", "").isalpha()}
    return found - _PLAIN_ACRONYMS


def _added_scheme_names(candidate, text):
    """Scheme names in candidate (English) that the person's text does not contain."""
    if _USER_NAMES_SCHEME.search(text):
        return []
    text_n = _norm(text)
    named = [m.group() for m in _SCHEME_WORDS.finditer(candidate) if _norm(m.group()) not in text_n]
    named += [a for a in _scheme_acronyms() if re.search(rf"(?<![A-Za-z]){re.escape(a)}(?![A-Za-z])", candidate)
              and a.casefold() not in text_n]
    return named


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(obj, text, language):
    """Return (cleaned_result, errors). Normalizes harmless format slips
    (case, snake_case, int-valued floats) and rejects everything else."""
    translated = language != "en"
    top_keys = {"profile", "confidence", "evidence", "clarifying_question", "clarifying_field"}
    if translated:
        top_keys.add("search_query_en")
    if not isinstance(obj, dict):
        return None, ["the answer is not a JSON object"]
    errors = []
    if set(obj) - top_keys:
        errors.append(f"unknown top-level keys: {sorted(set(obj) - top_keys)}")
    if top_keys - set(obj):
        errors.append(f"missing top-level keys: {sorted(top_keys - set(obj))}")
    profile = obj.get("profile")
    if not isinstance(profile, dict):
        return None, errors + ["profile must be an object"]
    if set(profile) != set(PROFILE_KEYS) | {"other_facts"}:
        errors.append(f"profile keys must be exactly {list(PROFILE_KEYS) + ['other_facts']}; got {sorted(profile)}")
    conf, evid = obj.get("confidence"), obj.get("evidence")
    if not isinstance(conf, dict) or not isinstance(evid, dict):
        return None, errors + ["confidence and evidence must be objects"]
    text_n = _norm(text)

    # -- translation (non-English only) ----------------------------------
    query = None
    if translated:
        query = obj.get("search_query_en")
        if not isinstance(query, str) or not query.strip():
            errors.append("search_query_en must be the English translation of the text")
            query = None
        else:
            query = re.sub(r"\s+", " ", query).strip()
            if _INDIC_RE.search(query):
                errors.append("search_query_en must be entirely in English")
            source_numbers, query_numbers = _numbers(text), _numbers(query)
            dropped = [n for n in source_numbers if not _derivable(n, query_numbers)]
            if dropped:
                errors.append(f"search_query_en drops numbers from the text: {dropped}")
            added = [n for n in query_numbers if not _derivable(n, source_numbers)]
            if added:
                errors.append(f"search_query_en adds numbers that are not in the text: {added}")
            neg = _SOURCE_NEGATION.get(language)
            if neg and neg.search(text) and not _EN_NEGATION.search(query):
                errors.append("the text contains a negation, and search_query_en must keep it")
            if _sentence_count(text) >= 2 and _sentence_count(query) < _sentence_count(text) - 1:
                errors.append("search_query_en must translate sentence by sentence, without merging or summarising")
            if _SUMMARY_STYLE.search(query):
                errors.append("search_query_en must be a translation in the person's own voice, not a summary about them")
            named = _added_scheme_names(query, text)
            if named:
                errors.append(f"search_query_en names a scheme the person did not mention: {named[:3]}")
    query_n = _norm(query) if query else ""

    def grounded(span):
        return isinstance(span, str) and bool(_norm(span)) and (
            _norm(span) in text_n or bool(query_n and _norm(span) in query_n))

    source_hint = "the person's text" + (" or your translation" if translated else "")

    # -- typed profile fields -------------------------------------------
    clean = dict.fromkeys(PROFILE_KEYS)
    for key in PROFILE_KEYS:
        v = profile.get(key)
        if v is None:
            if key in conf or key in evid:
                errors.append(f"{key} is null but has a confidence or evidence entry")
            continue
        if conf.get(key) not in CONFIDENCE:
            errors.append(f"confidence for {key} must be one of {list(CONFIDENCE)}")
        spans = evid.get(key)
        spans = [spans] if isinstance(spans, str) else spans
        if not isinstance(spans, list) or not spans or not all(grounded(s) for s in spans):
            errors.append(f"evidence for {key} must be a string or list of strings copied exactly from "
                          f"{source_hint}; got {evid.get(key)!r}")
            spans = []
        span_text = " ".join(s for s in spans if isinstance(s, str))

        if key in ("occupation", "family"):
            if not isinstance(v, str) or not v.strip():
                errors.append(f"{key} must be a non-empty string")
                continue
            v = v.strip().lower()
            if key == "family":
                v = re.sub(r"[^a-z0-9]+", "_", v).strip("_")
        elif key == "state":
            match = next((s for s in STATES if isinstance(v, str) and s.casefold() == v.strip().casefold()), None)
            if not match:
                errors.append(f"state {v!r} is not one of the listed states and union territories")
                continue
            v = match
        elif key == "gender":
            if not isinstance(v, str) or v.strip().lower() not in GENDERS:
                errors.append(f"gender must be one of {list(GENDERS)}; got {v!r}")
                continue
            v = v.strip().lower()
        elif key == "caste_category":
            match = next((c for c in CASTES if isinstance(v, str) and c.casefold() == v.strip().casefold()), None)
            if not match:
                errors.append(f"caste_category must be one of {list(CASTES)}; got {v!r}")
                continue
            v = match
        else:  # age, annual_income_inr, land_acres
            if isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0:
                errors.append(f"{key} must be a non-negative number; got {v!r}")
                continue
            if key in ("age", "annual_income_inr"):
                if float(v) != int(v):
                    errors.append(f"{key} must be a whole number; got {v!r}")
                    continue
                v = int(v)
            if key == "age" and not 0 < v < 120:
                errors.append(f"age {v} is not plausible")
                continue
            numbers = _numbers(span_text)
            if not numbers:
                errors.append(f"{key} needs evidence that states a number; if none is stated, {key} must be null")
                continue
            if key == "age":
                ratios = (1,)
            elif key == "land_acres":
                ratios = (1, 2.471) if _HECTARE.search(span_text) else (1,)
            else:
                multipliers = [1] + [m for rx, m in _UNITS.values() if rx.search(span_text)]
                periods = [1, 12] if _MONTH.search(span_text) else [1]
                ratios = tuple(m * p for m in multipliers for p in periods)
            if not _derivable(float(v), numbers, ratios):
                errors.append(f"{key}={v} does not follow from the number in its evidence {span_text!r} "
                              f"(only conversions the evidence states are allowed)")
                continue
        clean[key] = v
    for key in set(conf) | set(evid):
        if key not in PROFILE_KEYS:
            errors.append(f"confidence/evidence has an unknown key {key!r}")

    # -- other_facts ------------------------------------------------------
    facts = profile.get("other_facts")
    if facts is None:
        facts = []
    if not isinstance(facts, list) or not all(isinstance(f, str) and f.strip() for f in facts):
        errors.append("other_facts must be a list of strings (empty if there are none)")
        facts = []
    else:
        facts = [re.sub(r"\s+", " ", f).strip() for f in facts]
        if len(facts) > MAX_OTHER_FACTS:
            errors.append(f"other_facts has more than {MAX_OTHER_FACTS} entries")
        ungrounded = [f for f in facts if not grounded(f)]
        if ungrounded:
            errors.append(f"other_facts entries must be copied word for word from {source_hint}: {ungrounded[:3]}")

    # -- states named in the translation ----------------------------------
    if query:
        for s in STATES:
            if re.search(rf"\b{re.escape(s.casefold())}\b", query_n) and s != clean["state"] \
                    and s.casefold() not in text_n:
                errors.append(f"search_query_en names the state {s!r}, but profile.state is {clean['state']!r}")

    # -- clarifying question ------------------------------------------------
    question, field = obj.get("clarifying_question"), obj.get("clarifying_field")
    if question is None:
        if field is not None:
            errors.append("clarifying_field must be null when clarifying_question is null")
    elif not isinstance(question, str) or question.count("?") != 1 or len(question) > 200:
        errors.append("clarifying_question must be null or exactly one short question with one '?'")
        question, field = None, None
    else:
        question = question.strip()
        if _CONJUNCTION.search(question):
            errors.append("clarifying_question must ask about one thing only, not two joined by 'and'")
        if field not in PROFILE_KEYS and field != "other":
            errors.append(f"clarifying_field must be one of {list(PROFILE_KEYS)} or 'other'; got {field!r}")
        elif field in PROFILE_KEYS and clean[field] is not None and conf.get(field) != "low":
            errors.append(f"clarifying_question asks about {field}, which is already known with "
                          f"{conf.get(field)} confidence")

    result = {
        "search_query_en": query if translated else text,
        "profile": {**clean, "other_facts": facts},
        "confidence": {k: conf[k] for k in PROFILE_KEYS if clean[k] is not None and k in conf},
        "evidence": {k: evid[k] for k in PROFILE_KEYS if clean[k] is not None and k in evid},
        "clarifying_question": question,
        "clarifying_field": field if question else None,
    }
    return result, errors


def _parse(raw):
    try:
        return json.loads(lc._strip_fences(raw)), None
    except (json.JSONDecodeError, TypeError) as e:
        m = re.search(r"\{.*\}", raw or "", re.DOTALL)
        if m:
            try:
                return json.loads(m.group()), None
            except json.JSONDecodeError:
                pass
        return None, f"the answer is not valid JSON ({e})"


# ---------------------------------------------------------------------------
# Providers and cache
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _pools():
    lc.load_dotenv(ROOT / ".env")
    logging.getLogger("google_genai.models").setLevel(logging.ERROR)
    pools = [lc.ProviderPool("gemini", lc.DEFAULT_GEMINI_MODEL, "GEMINI_API_KEY", lc.make_gemini_client),
             lc.ProviderPool("groq", lc.GROQ_MODEL, "GROQ_API_KEY", lc.make_groq_client)]
    if not any(p.available for p in pools):
        raise UnderstandError("no GEMINI_API_KEY or GROQ_API_KEY configured in .env")
    return pools


def _generate(prompt):
    """One answer from the first provider that can give one."""
    failures = []
    for pool in _pools():
        if not pool.available:
            continue
        res = lc.pool_generate(pool, prompt, delay=1.0)
        if res["status"] == "ok":
            return res["text"], pool.name, pool.model
        if res["status"] == "fatal":
            pool.disable("fatal")
        failures.append(f"{pool.name}: {res['status']} ({res['error']})")
    raise UnderstandError("no provider answered: " + "; ".join(failures or ["all providers exhausted"]))


def cache_key(text, language, version=PROMPT_VERSION):
    return hashlib.sha256(json.dumps([text, language, version], ensure_ascii=False).encode()).hexdigest()


@lru_cache(maxsize=1)
def _cache():
    entries = {}
    if CACHE_PATH.exists():
        with open(CACHE_PATH, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    e = json.loads(line)
                    entries[e["key"]] = e
    return entries


def _store(entry):
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    _cache()[entry["key"]] = entry


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def understand(text, language, version=None):
    """Search query, profile and clarifying question for one input.

    Returns a dict with search_query_en, profile (including other_facts),
    confidence, evidence, clarifying_question, clarifying_field, status
    ("ok" | "invalid"), errors and _meta. version selects a prompt version;
    anything but the current one is served from the cache only."""
    version = version or PROMPT_VERSION
    key = cache_key(text, language, version)
    hit = _cache().get(key)
    if hit:
        return {**hit["result"], "_meta": {**hit["meta"], "cached": True}}
    if version != PROMPT_VERSION:
        raise UnderstandError(f"{version} is retired and has no cached result for this text")

    base = build_prompt(text, language)
    prompt, attempts, errors, result = base, [], [], None
    for _ in (1, 2):
        raw, provider, model = _generate(prompt)
        obj, parse_error = _parse(raw)
        result, errors = validate(obj, text, language) if obj is not None else (None, [parse_error])
        attempts.append({"provider": provider, "model": model, "errors": errors})
        if not errors:
            break
        prompt = base + RETRY_SUFFIX.format(errors="\n".join(f"- {e}" for e in errors))

    if errors:
        result = {"search_query_en": text if language == "en" else None,
                  "profile": {**dict.fromkeys(PROFILE_KEYS), "other_facts": []},
                  "confidence": {}, "evidence": {}, "clarifying_question": None, "clarifying_field": None}
    result = {**result, "status": "invalid" if errors else "ok", "errors": errors}
    meta = {"prompt_version": version, "language": language, "attempts": attempts,
            "provider": attempts[-1]["provider"], "model": attempts[-1]["model"],
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    _store({"key": key, "text": text, "language": language, "prompt_version": version,
            "result": result, "meta": meta})
    return {**result, "_meta": {**meta, "cached": False}}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Run understand() on one input and print the result.")
    ap.add_argument("text")
    ap.add_argument("--language", default="en", choices=sorted(LANGUAGE_NAMES))
    a = ap.parse_args()
    print(json.dumps(understand(a.text, a.language), ensure_ascii=False, indent=2))

"""Query understanding: one LLM call turns what a person says into

- search_query_en: the text used for retrieval. For English input it is the
            person's text exactly, set in code with no LLM involvement. For
            other languages it is a faithful sentence-by-sentence English
            translation that keeps every fact, number, duration, negation and
            scheme name, with nothing summarised or added.
- profile:  typed fields, each null unless stated:
              the gold set's eight expected_profile keys (occupation, state,
              age, gender, annual_income_inr, land_acres, caste_category, family)
              and the fields the constraint schema needs (education_class,
              education_stage, residence, marital_status, disability_percent,
              is_minority, bpl_household, registered_construction_worker,
              prior_benefit_schemes)
            plus other_facts: short verbatim statements that could decide
            eligibility but fit no typed field (durations, loan history,
            registration details, employment type)
- confidence / evidence: for every non-null typed field, high|medium|low and
            the span(s) of the text it comes from
- clarifying_question: only a fallback for a nearly empty profile (fewer than
            two substantive fields, e.g. "I am a farmer, please help me").
            Choosing what to ask is otherwise the matcher's job
            (backend/app/matcher.py); phrase_question() words its choice.

The call goes through the provider pool from scripts/label_categories.py
(Gemini flash-lite first, Groq fallback, key rotation on daily caps,
temperature 0). The answer is validated before use:

- unknown keys and bad enums are rejected
- every evidence span, other_facts entry and prior_benefit_schemes entry must
  occur in the input (for non-English input, it may instead be a piece of the
  translation)
- numeric fields need evidence containing a number, and the value must follow
  from it using only conversions the evidence states (lakh, thousand, crore,
  monthly x12, hectares); otherwise the field must be null
- a translation must keep every number and any negation in the source, must
  not merge the text into fewer sentences or rephrase it as a third-person
  summary, and must not add numbers, states or scheme names

A rejected answer is retried once with the problems listed. If the retry also
fails, the result has status "invalid" and an empty profile; for English input
search_query_en is still the person's text. A problem with the clarifying
question alone never costs the profile: the question is dropped, noted in
"notes", and the rest is kept.

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

PROMPT_VERSION = "understand-v3"
CACHE_PATH = ROOT / "data" / "cache" / "understand.jsonl"
SCHEMES_DIR = ROOT / "data" / "interim" / "schemes"

GOLD_KEYS = ("occupation", "state", "age", "gender", "annual_income_inr",
             "land_acres", "caste_category", "family")
SCHEMA_KEYS = ("education_class", "education_stage", "residence", "marital_status", "disability_percent",
               "is_minority", "bpl_household", "registered_construction_worker", "prior_benefit_schemes")
PROFILE_KEYS = GOLD_KEYS + SCHEMA_KEYS
NUMERIC_KEYS = ("age", "annual_income_inr", "land_acres", "education_class", "disability_percent")
BOOLEAN_KEYS = ("is_minority", "bpl_household", "registered_construction_worker")
# Fields that say something about the person; family and prior_benefit_schemes
# are context, so a profile holding only those is still "nearly empty".
SUBSTANTIVE_KEYS = tuple(k for k in PROFILE_KEYS if k not in ("family", "prior_benefit_schemes"))
NEARLY_EMPTY_BELOW = 2

STATES = (
    "Andaman and Nicobar Islands", "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chandigarh",
    "Chhattisgarh", "Dadra and Nagar Haveli and Daman and Diu", "Delhi", "Goa", "Gujarat", "Haryana",
    "Himachal Pradesh", "Jammu and Kashmir", "Jharkhand", "Karnataka", "Kerala", "Ladakh", "Lakshadweep",
    "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Puducherry",
    "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand",
    "West Bengal",
)
ENUMS = {
    "gender": ("female", "male", "transgender"),
    "caste_category": ("SC", "ST", "OBC", "EWS", "General"),
    "education_stage": ("school", "higher_secondary", "diploma", "undergraduate", "postgraduate", "doctoral"),
    "residence": ("rural", "urban"),
    "marital_status": ("married", "unmarried", "widowed", "divorced", "separated", "abandoned"),
}
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
"profile": an object with exactly these eighteen keys. Every typed field is null unless the person stated it:
  "occupation": short lowercase English noun phrase for their work or role, e.g. "farmer", "tenant farmer", "student", "construction worker", "street vendor".
  "state": the Indian state or union territory, written exactly as one of: {states}. Fill it when they name the state, or a city or district that clearly belongs to one state.
  "age": the person's own age in whole years (integer), only if they state a number.
  "gender": "female", "male" or "transgender", only when they describe themself with a gendered word (woman, widow, mother, pregnant, girl, man...). Do not infer it from a spouse or from grammatical gender.
  "annual_income_inr": their personal or family income per year in rupees (integer), only if they state an amount. Convert only what they state: lakh (x 100000), thousand (x 1000), a monthly amount (x 12). Never estimate; never write 0 unless they say their income is zero.
  "land_acres": land they own or cultivate in acres (number), only if they state an amount; convert hectares (x 2.471). If the unit varies by region (bigha, kanal, guntha), null.
  "caste_category": "SC", "ST", "OBC", "EWS" or "General", only when stated.
  "family": a short snake_case summary of household facts that matter for eligibility (children and their ages or classes, pregnancy, a death in the family), e.g. "two_children_in_school".
  "education_class": the school class (integer 1 to 12) of the student they are asking about (themself, or their child if they ask for the child), only if stated.
  "education_stage": "school" (classes 1-10), "higher_secondary" (classes 11-12), "diploma" (ITI, polytechnic), "undergraduate", "postgraduate" or "doctoral", for that same student, only if stated or given by a stated class or course.
  "residence": "rural" if they say they live in a village or rural area, "urban" if they say they live in a city or town.
  "marital_status": "married", "unmarried", "widowed", "divorced", "separated" or "abandoned": the person's own, only if stated.
  "disability_percent": the person's disability percentage (integer 0-100), only if they state a percentage.
  "is_minority": true if they say they are Muslim, Christian, Sikh, Buddhist, Jain or Parsi, or say they belong to a minority; false if they state a religion that is not a minority.
  "bpl_household": true if they say they are BPL, below the poverty line, or hold a BPL card; false if they say they are not BPL or hold an APL card.
  "registered_construction_worker": true if they say they are registered with a construction workers' welfare board; false if they say they are not registered. Working in construction is not registration.
  "prior_benefit_schemes": a list of the schemes they say they already receive or have received, copied as they wrote them (e.g. ["PM-KISAN"], ["Mudra loan"]). Only schemes they name; null if they name none.
  "other_facts": a list of short statements copied word for word from {fact_source} that could decide eligibility but fit none of the fields above: how long something has lasted, loans taken or never taken, registration or membership details, type of employment (permanent, contract, government). Include negations as written. An empty list if there are none. Do not repeat what the typed fields already hold.

"confidence": an object with one entry for every non-null typed field (not other_facts): "high" if stated plainly, "medium" if clearly implied, "low" if uncertain.

"evidence": an object with one entry for every non-null typed field (not other_facts): the words the value comes from, copied character for character from {fact_source}. Use a list of strings when the value comes from separate parts of the text. For numeric fields the evidence must contain the number.

"clarifying_question": null, unless the text gives almost nothing to go on (for example "I am a farmer, please help me", or "I need a house"). Then one short question, in the language they wrote in, asking what would help most to find schemes for them.

{example}
"""

TRANSLATION_RULES = """
"search_query_en": a faithful English translation of the person's text, sentence by sentence, in the same order and the same person ("I", "my"). Keep every fact, number, amount, duration, negation ("never", "not", "no") and scheme name exactly; keep amounts as they wrote them (e.g. "1.8 lakh"). Do not summarise, do not add anything (no "looking for government schemes"), do not drop anything.
"""

_EMPTY_NEW = ('"education_class": null, "education_stage": null, "residence": null, "marital_status": null, '
              '"disability_percent": null, "is_minority": null, "bpl_household": null, '
              '"registered_construction_worker": null')

EXAMPLE_EN = """Example. Person wrote: "I run a small tea stall in Pune and want a loan to expand it. I've run it for 6 years and have never taken a bank loan. I'm 34, married, my wife and I earn about 1.2 lakh a year, and we have two school-going kids."
{"profile": {"occupation": "tea stall owner", "state": "Maharashtra", "age": 34, "gender": null, "annual_income_inr": 120000, "land_acres": null, "caste_category": null, "family": "two_school_going_children", "education_class": null, "education_stage": null, "residence": null, "marital_status": "married", "disability_percent": null, "is_minority": null, "bpl_household": null, "registered_construction_worker": null, "prior_benefit_schemes": null, "other_facts": ["I've run it for 6 years", "have never taken a bank loan"]},
 "confidence": {"occupation": "high", "state": "high", "age": "high", "annual_income_inr": "medium", "family": "high", "marital_status": "high"},
 "evidence": {"occupation": "I run a small tea stall", "state": "Pune", "age": "I'm 34", "annual_income_inr": "earn about 1.2 lakh a year", "family": "we have two school-going kids", "marital_status": "married"},
 "clarifying_question": null}"""

EXAMPLE_TRANSLATED = """Example. Person wrote (Hindi): "मैं इंदौर में साइकिल रिपेयर की दुकान चलाता हूँ और महीने के 8 हज़ार कमाता हूँ। मेरे पास बीपीएल कार्ड है, पर मैंने पहले कभी लोन नहीं लिया।"
{"search_query_en": "I run a bicycle repair shop in Indore and earn 8 thousand a month. I have a BPL card, but I have never taken a loan before.",
 "profile": {"occupation": "bicycle repair shop owner", "state": "Madhya Pradesh", "age": null, "gender": null, "annual_income_inr": 96000, "land_acres": null, "caste_category": null, "family": null, "education_class": null, "education_stage": null, "residence": "urban", "marital_status": null, "disability_percent": null, "is_minority": null, "bpl_household": true, "registered_construction_worker": null, "prior_benefit_schemes": null, "other_facts": ["I have never taken a loan before"]},
 "confidence": {"occupation": "high", "state": "high", "annual_income_inr": "high", "residence": "medium", "bpl_household": "high"},
 "evidence": {"occupation": "I run a bicycle repair shop", "state": "इंदौर", "annual_income_inr": "earn 8 thousand a month", "residence": "Indore", "bpl_household": "I have a BPL card"},
 "clarifying_question": null}"""

RETRY_SUFFIX = """
Your previous answer was rejected for these reasons:
{errors}
Return a corrected JSON object only."""


def build_prompt(text, language):
    translated = language != "en"
    keys = (["search_query_en"] if translated else []) + ["profile", "confidence", "evidence", "clarifying_question"]
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
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8,
    "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12,
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


def _norm(s):
    return re.sub(r"\s+", " ", (s or "").casefold()).strip(" \"'“”‘’.,;:!?।")


def _numbers(text):
    """Digits (in any script) and English number and ordinal words."""
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
    """Return (cleaned_result, errors, question_errors).

    errors reject the answer; question_errors concern only the clarifying
    question, which is dropped instead. Harmless format slips (case,
    snake_case, int-valued floats) are normalized."""
    translated = language != "en"
    top_keys = {"profile", "confidence", "evidence", "clarifying_question"}
    if translated:
        top_keys.add("search_query_en")
    if not isinstance(obj, dict):
        return None, ["the answer is not a JSON object"], []
    errors, question_errors = [], []
    if set(obj) - top_keys:
        errors.append(f"unknown top-level keys: {sorted(set(obj) - top_keys)}")
    if top_keys - set(obj):
        errors.append(f"missing top-level keys: {sorted(top_keys - set(obj))}")
    profile = obj.get("profile")
    if not isinstance(profile, dict):
        return None, errors + ["profile must be an object"], []
    if set(profile) != set(PROFILE_KEYS) | {"other_facts"}:
        errors.append(f"profile keys must be exactly {list(PROFILE_KEYS) + ['other_facts']}; got {sorted(profile)}")
    conf, evid = obj.get("confidence"), obj.get("evidence")
    if not isinstance(conf, dict) or not isinstance(evid, dict):
        return None, errors + ["confidence and evidence must be objects"], []
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
        if v is None or (key == "prior_benefit_schemes" and v == []):
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
        elif key in ENUMS:
            if key == "marital_status" and isinstance(v, str) and v.strip().casefold() in ("widow", "widower"):
                v = "widowed"
            match = next((e for e in ENUMS[key] if isinstance(v, str) and e.casefold() == v.strip().casefold()), None)
            if not match:
                errors.append(f"{key} must be one of {list(ENUMS[key])}; got {v!r}")
                continue
            v = match
        elif key in BOOLEAN_KEYS:
            if not isinstance(v, bool):
                errors.append(f"{key} must be true, false or null; got {v!r}")
                continue
        elif key == "prior_benefit_schemes":
            if not isinstance(v, list) or not all(isinstance(x, str) and x.strip() for x in v):
                errors.append("prior_benefit_schemes must be a list of scheme names or null")
                continue
            v = [re.sub(r"\s+", " ", x).strip() for x in v]
            ungrounded = [x for x in v if not grounded(x)]
            if ungrounded:
                errors.append(f"prior_benefit_schemes may only list schemes named in {source_hint}: {ungrounded}")
                continue
        else:  # numeric
            if isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0:
                errors.append(f"{key} must be a non-negative number; got {v!r}")
                continue
            if key != "land_acres":
                if float(v) != int(v):
                    errors.append(f"{key} must be a whole number; got {v!r}")
                    continue
                v = int(v)
            bounds = {"age": (1, 119), "education_class": (1, 12), "disability_percent": (0, 100)}.get(key)
            if bounds and not bounds[0] <= v <= bounds[1]:
                errors.append(f"{key} {v} is outside {bounds[0]}-{bounds[1]}")
                continue
            numbers = _numbers(span_text)
            if not numbers:
                errors.append(f"{key} needs evidence that states a number; if none is stated, {key} must be null")
                continue
            if key == "annual_income_inr":
                multipliers = [1] + [m for rx, m in _UNITS.values() if rx.search(span_text)]
                periods = [1, 12] if _MONTH.search(span_text) else [1]
                ratios = tuple(m * p for m in multipliers for p in periods)
            elif key == "land_acres":
                ratios = (1, 2.471) if _HECTARE.search(span_text) else (1,)
            else:
                ratios = (1,)
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

    # -- clarifying question: a fallback for a nearly empty profile only ----
    question = obj.get("clarifying_question")
    if question is not None:
        filled = sum(clean[k] is not None for k in SUBSTANTIVE_KEYS)
        if not isinstance(question, str) or question.count("?") != 1 or len(question) > 200:
            question_errors.append("clarifying_question must be null or one short question with one '?'")
        elif filled >= NEARLY_EMPTY_BELOW:
            question_errors.append(f"clarifying_question is only for a nearly empty profile; this one has "
                                   f"{filled} substantive fields")
        if question_errors:
            question = None
        else:
            question = question.strip()

    result = {
        "search_query_en": query if translated else text,
        "profile": {**clean, "other_facts": facts},
        "confidence": {k: conf[k] for k in PROFILE_KEYS if clean[k] is not None and k in conf},
        "evidence": {k: evid[k] for k in PROFILE_KEYS if clean[k] is not None and k in evid},
        "clarifying_question": question,
    }
    return result, errors, question_errors


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
# Entry points
# ---------------------------------------------------------------------------

def understand(text, language, version=None):
    """Search query, profile and fallback clarifying question for one input.

    Returns a dict with search_query_en, profile (including other_facts),
    confidence, evidence, clarifying_question, status ("ok" | "invalid"),
    errors, notes and _meta. version selects a prompt version; anything but
    the current one is served from the cache only."""
    version = version or PROMPT_VERSION
    key = cache_key(text, language, version)
    hit = _cache().get(key)
    if hit:
        return {**hit["result"], "_meta": {**hit["meta"], "cached": True}}
    if version != PROMPT_VERSION:
        raise UnderstandError(f"{version} is retired and has no cached result for this text")

    base = build_prompt(text, language)
    prompt, attempts, errors, question_errors, result = base, [], [], [], None
    for _ in (1, 2):
        raw, provider, model = _generate(prompt)
        obj, parse_error = _parse(raw)
        if obj is None:
            result, errors, question_errors = None, [parse_error], []
        else:
            result, errors, question_errors = validate(obj, text, language)
        attempts.append({"provider": provider, "model": model, "errors": errors, "question_errors": question_errors})
        if not errors:                      # question problems alone never trigger a retry
            break
        prompt = base + RETRY_SUFFIX.format(errors="\n".join(f"- {e}" for e in errors + question_errors))

    notes = [f"clarifying_question dropped: {e}" for e in question_errors] if not errors else []
    if errors:
        result = {"search_query_en": text if language == "en" else None,
                  "profile": {**dict.fromkeys(PROFILE_KEYS), "other_facts": []},
                  "confidence": {}, "evidence": {}, "clarifying_question": None}
    result = {**result, "status": "invalid" if errors else "ok", "errors": errors, "notes": notes}
    meta = {"prompt_version": version, "language": language, "attempts": attempts,
            "provider": attempts[-1]["provider"], "model": attempts[-1]["model"],
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    _store({"key": key, "text": text, "language": language, "prompt_version": version,
            "result": result, "meta": meta})
    return {**result, "_meta": {**meta, "cached": False}}


# Plain descriptions of what each askable field means, for phrase_question().
FIELD_DESCRIPTIONS = {
    "occupation": "what work they do", "state": "which state they live in", "age": "their age",
    "gender": "their gender", "annual_income_inr": "their family's total income per year",
    "land_acres": "how much land they own, in acres", "caste_category": "their caste category (SC, ST, OBC, EWS or General)",
    "education_class": "which class the student is studying in", "education_stage": "the student's level of study",
    "residence": "whether they live in a village or a town/city", "marital_status": "their marital status",
    "disability_percent": "their disability percentage on the disability certificate",
    "is_minority": "whether they belong to a religious minority", "bpl_household": "whether they have a BPL card",
    "registered_construction_worker": "whether they are registered with a construction workers' welfare board",
}
PHRASE_PROMPT = """Write one short, polite question in {language_name} asking a person applying for government welfare schemes {what}.
Return ONLY a JSON object: {{"question": "<the question, ending with ?>"}}"""


def phrase_question(field, language):
    """Word the matcher's chosen field as one question in the person's
    language. One small LLM call, cached like understand()."""
    if field not in FIELD_DESCRIPTIONS:
        raise ValueError(f"no description for field {field!r}")
    version = f"phrase-{PROMPT_VERSION}"
    key = cache_key(f"field:{field}", language, version)
    hit = _cache().get(key)
    if hit:
        return hit["result"]["question"]
    prompt = PHRASE_PROMPT.format(language_name=LANGUAGE_NAMES.get(language, language), what=FIELD_DESCRIPTIONS[field])
    question, provider = None, None
    for _ in (1, 2):
        raw, provider, _model = _generate(prompt)
        obj, _ = _parse(raw)
        q = obj.get("question") if isinstance(obj, dict) else None
        if isinstance(q, str) and q.count("?") == 1 and len(q) <= 200 \
                and (language == "en" or _INDIC_RE.search(q)):
            question = q.strip()
            break
    if question is None:
        raise UnderstandError(f"could not phrase a question for {field!r} in {language}")
    _store({"key": key, "text": f"field:{field}", "language": language, "prompt_version": version,
            "result": {"question": question}, "meta": {"provider": provider}})
    return question


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Run understand() on one input and print the result.")
    ap.add_argument("text")
    ap.add_argument("--language", default="en", choices=sorted(LANGUAGE_NAMES))
    a = ap.parse_args()
    print(json.dumps(understand(a.text, a.language), ensure_ascii=False, indent=2))

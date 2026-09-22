"""Query understanding: one LLM call turns what a person says into

- profile:  the gold set's eight expected_profile keys (occupation, state, age,
            gender, annual_income_inr, land_acres, caste_category, family),
            null when not stated
- confidence / evidence: for every non-null profile field, high|medium|low
            and the exact span of the person's text it was taken from
- search_query_en: one or two plain English sentences restating the person's
            situation and needs, for retrieval
- clarifying_question: one question when a decisive field is missing or
            low-confidence, else null

The call goes through the provider pool from scripts/label_categories.py
(Gemini flash-lite first, Groq fallback, key rotation on daily caps,
temperature 0). The answer is validated: unknown keys, bad enums, profile
values whose evidence span is not in the text or whose digits don't support
the number, and rewrites that introduce numbers, states or scheme names the
person never gave are all rejected. A rejected answer is retried once with
the problems listed; if the retry also fails, the result has status
"invalid" and no search query.

Every result is cached in data/cache/understand.jsonl, keyed by (text,
language, PROMPT_VERSION), so evaluation is deterministic and never spends
quota twice. Provider failures (quota, network) raise UnderstandError and are
not cached.
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

PROMPT_VERSION = "understand-v1"
CACHE_PATH = ROOT / "data" / "cache" / "understand.jsonl"
SCHEMES_DIR = ROOT / "data" / "interim" / "schemes"

PROFILE_KEYS = ("occupation", "state", "age", "gender", "annual_income_inr",
                "land_acres", "caste_category", "family")
TOP_KEYS = ("profile", "confidence", "evidence", "search_query_en", "clarifying_question")
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


class UnderstandError(RuntimeError):
    """No provider could answer (quota exhausted, network, misconfiguration)."""


PROMPT = """You help Indian citizens find government welfare schemes. Read what one person wrote and return a JSON object describing them.

The person wrote in {language_name}:
<<<
{text}
>>>

Return ONLY a JSON object with exactly these keys, no markdown:

"profile": an object with exactly these eight keys, each null unless the person stated it:
  "occupation": short lowercase English noun phrase for their work or role, e.g. "farmer", "tenant farmer", "student", "construction worker", "street vendor", "unemployed".
  "state": the Indian state or union territory, written exactly as one of: {states}. Fill it when they name the state, or a city or district that clearly belongs to one state.
  "age": the person's own age in whole years (integer).
  "gender": "female", "male" or "transgender", only when they describe themself with a gendered word (woman, widow, mother, pregnant, girl, man...). Do not infer it from a spouse or from grammatical gender.
  "annual_income_inr": their personal or family income per year, as an integer number of rupees. Convert lakh (x 100000), thousand (x 1000) and monthly figures (x 12).
  "land_acres": land they own or cultivate, in acres (number). Convert hectares (x 2.471). If the unit varies by region (bigha, kanal, guntha), leave null.
  "caste_category": "SC", "ST", "OBC", "EWS" or "General", only when stated.
  "family": a short snake_case summary of household facts that matter for eligibility, such as marital status, children and their ages or classes, BPL card, disability percentage, pregnancy, a death in the family, e.g. "widow_bpl" or "two_children_in_school". null if none.

"confidence": an object with one entry for every non-null profile field: "high" if stated plainly, "medium" if clearly implied, "low" if uncertain.

"evidence": an object with one entry for every non-null profile field: the exact words from the person's text (copied character for character, in the original language) that the value comes from.

"search_query_en": one or two plain English sentences, in the third person, restating the person's situation and what they need, using everyday words that appear in scheme eligibility rules (for example "below poverty line", "annual family income of Rs 1,20,000", "Scheduled Tribe", "owns 2 acres of farmland"). Rules:
  - Use only facts the person stated. Do not add ages, amounts, places, caste, needs or other details they did not give.
  - Never name a specific scheme, yojana or programme unless the person named it.
  - Write amounts in rupees and land in acres; translate everything into English.

"clarifying_question": if a fact that decides which schemes fit (for example state, age, income, caste, class of study, land, BPL status) is missing or low-confidence for what they are asking, one short question to ask them, in the same language they wrote in. Otherwise null.

Example. Person wrote: "I run a small tea stall in Pune and want a loan to expand it. I'm 34, my wife and I earn about 1.2 lakh a year, and we have two school-going kids."
{{"profile": {{"occupation": "tea stall owner", "state": "Maharashtra", "age": 34, "gender": null, "annual_income_inr": 120000, "land_acres": null, "caste_category": null, "family": "married_two_school_children"}},
 "confidence": {{"occupation": "high", "state": "high", "age": "high", "annual_income_inr": "medium", "family": "high"}},
 "evidence": {{"occupation": "I run a small tea stall", "state": "Pune", "age": "I'm 34", "annual_income_inr": "earn about 1.2 lakh a year", "family": "we have two school-going kids"}},
 "search_query_en": "A 34-year-old self-employed tea stall owner in Maharashtra with an annual family income of Rs 1,20,000 and two school-going children needs a loan to expand a small business.",
 "clarifying_question": null}}
"""

RETRY_SUFFIX = """
Your previous answer was rejected for these reasons:
{errors}
Return a corrected JSON object only."""


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_NUM_RE = re.compile(r"\d+(?:[.,]\d+)*")       # \d also matches Devanagari/Telugu/Tamil/Bengali digits
_INDIC_RE = re.compile(r"[ऀ-෿]")
_SCHEME_WORDS = re.compile(r"\b(?:yojana|yojna|yojane|abhiyan|abhiyaan|pradhan\s+mantri|mukhya\s*mantri|pm[\s-]+[a-z]{3,})\b",
                           re.IGNORECASE)
# The person asked about a named scheme themself, so the rewrite may name it.
_USER_NAMES_SCHEME = re.compile(r"yojana|yojna|scheme|nidhi|card|योजना|निधि|कार्ड|యోజన|పథకం|కార్డ్|నిధి|திட்டம்|যোজনা|প্রকল্প",
                                re.IGNORECASE)
_RATIOS = (1, 1e3, 1e5, 1e7, 12, 2.471, 1e-3, 1e-5, 1e-7, 1 / 12, 1 / 2.471, 1e5 * 12, 1e3 * 12)


def _norm(s):
    return re.sub(r"\s+", " ", (s or "").casefold()).strip(" \"'“”‘’.,;:!?")


def _numbers(text):
    out = []
    for m in _NUM_RE.finditer(text or ""):
        s = m.group()
        # 1,80,000 / 90,000 are digit grouping; 2.5 is a decimal.
        s = s.replace(",", "") if re.fullmatch(r"\d{1,3}(?:,\d{2,3})+", s) else s.replace(",", ".")
        try:
            out.append(float(s))
        except ValueError:
            pass
    return out


def _close(a, b):
    return abs(a - b) <= max(0.02 * abs(b), 0.011)


def _derivable(value, numbers, ratios=_RATIOS):
    return any(_close(value, n * r) for n in numbers for r in ratios)


# Everyday acronyms that are not scheme names.
_PLAIN_ACRONYMS = {"SC", "ST", "OBC", "EWS", "BPL", "APL", "PWD", "PwD", "ITI", "NRI", "LPG", "SHG", "MSME",
                   "PSU", "UT", "ID", "NGO", "HIV", "AIDS", "OBCs", "DNT", "EBC", "SEBC", "MBC", "BC", "VJNT",
                   "NT", "SBC", "PG", "UG", "BPL-", "ASHA", "RTE"}


@lru_cache(maxsize=1)
def _scheme_acronyms():
    """Acronyms written in parentheses in corpus scheme names (PMEGP, IGNOAPS,
    HBOCWWB...). Full generic names such as "Old Age Pension" are not checked:
    they also read as plain descriptions of a need."""
    found = set()
    for p in SCHEMES_DIR.glob("*.json"):
        name = json.loads(p.read_text(encoding="utf-8")).get("scheme_name") or ""
        for inner in re.findall(r"\(([^)]*)\)", name):
            for tok in re.split(r"[\s/:,]+", inner):
                tok = tok.strip("-.")
                if len(tok) >= 3 and sum(c.isupper() for c in tok) >= 2 and tok not in _PLAIN_ACRONYMS:
                    found.add(tok)
    # Slugs are often the scheme's acronym too (pmegp, pmjdy, pmfby).
    found |= {p.stem.replace("-", "").upper() for p in SCHEMES_DIR.glob("*.json")
              if len(p.stem.replace("-", "")) >= 4 and p.stem.replace("-", "").isalpha()}
    return found - _PLAIN_ACRONYMS


def validate(obj, text):
    """Return (cleaned_result, errors). Normalizes harmless format slips
    (case, snake_case, int-valued floats) and rejects everything else."""
    errors = []
    if not isinstance(obj, dict):
        return None, ["the answer is not a JSON object"]
    extra = set(obj) - set(TOP_KEYS)
    missing = set(TOP_KEYS) - set(obj)
    if extra:
        errors.append(f"unknown top-level keys: {sorted(extra)}")
    if missing:
        errors.append(f"missing top-level keys: {sorted(missing)}")
    profile = obj.get("profile")
    if not isinstance(profile, dict):
        return None, errors + ["profile must be an object"]
    if set(profile) != set(PROFILE_KEYS):
        errors.append(f"profile keys must be exactly {list(PROFILE_KEYS)}; got {sorted(profile)}")
    conf = obj.get("confidence") or {}
    evid = obj.get("evidence") or {}
    if not isinstance(conf, dict) or not isinstance(evid, dict):
        return None, errors + ["confidence and evidence must be objects"]

    clean = dict.fromkeys(PROFILE_KEYS)
    text_n = _norm(text)
    for key in PROFILE_KEYS:
        v = profile.get(key)
        if v is None:
            if key in conf or key in evid:
                errors.append(f"{key} is null but has a confidence or evidence entry")
            continue
        if conf.get(key) not in CONFIDENCE:
            errors.append(f"confidence for {key} must be one of {list(CONFIDENCE)}")
        span = evid.get(key)
        if not isinstance(span, str) or not _norm(span) or _norm(span) not in text_n:
            errors.append(f"evidence for {key} must be words copied exactly from the person's text; got {span!r}")
            span = ""
        span_numbers = _numbers(span)

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
            # Digits in the evidence must support the value (lakh, thousand,
            # monthly and hectare conversions allowed). A span with no digits
            # ("half an acre", number words) cannot be checked this way.
            if span_numbers:
                ratios = (1,) if key == "age" else _RATIOS
                if not _derivable(float(v), span_numbers, ratios):
                    errors.append(f"{key}={v} does not follow from the numbers in its evidence {span!r}")
                    continue
        clean[key] = v
    for key in set(conf) | set(evid):
        if key not in PROFILE_KEYS:
            errors.append(f"confidence/evidence has an unknown key {key!r}")

    query = obj.get("search_query_en")
    if not isinstance(query, str) or not query.strip():
        errors.append("search_query_en must be a non-empty string")
        query = None
    else:
        query = re.sub(r"\s+", " ", query).strip()
        if _INDIC_RE.search(query):
            errors.append("search_query_en must be in English only")
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", re.sub(r"\b(?:Rs|e\.g|i\.e)\.", "", query)) if s.strip()]
        if len(sentences) > 2:
            errors.append(f"search_query_en must be one or two sentences; got {len(sentences)}")
        grounded = _numbers(text) + [float(x) for x in (clean["age"], clean["annual_income_inr"], clean["land_acres"])
                                     if x is not None]
        invented = [n for n in _numbers(query) if not _derivable(n, grounded)]
        if invented:
            errors.append(f"search_query_en contains numbers the person did not give: {invented}")
        q_n = _norm(query)
        for s in STATES:
            if re.search(rf"\b{re.escape(s.casefold())}\b", q_n) and s != clean["state"] and s.casefold() not in text_n:
                errors.append(f"search_query_en names the state {s!r}, which the person did not mention")
        if not _USER_NAMES_SCHEME.search(text):
            named = [m.group() for m in _SCHEME_WORDS.finditer(query) if _norm(m.group()) not in text_n]
            named += [a for a in _scheme_acronyms() if re.search(rf"(?<![A-Za-z]){re.escape(a)}(?![A-Za-z])", query)
                      and a.casefold() not in text_n]
            named += [t for t in re.findall(r"\b[A-Z][A-Z-]{2,}\b", query)
                      if t.replace("-", "") in _scheme_acronyms() and t.casefold() not in text_n]
            if named:
                errors.append(f"search_query_en names a scheme the person did not mention: {named[:3]}")

    question = obj.get("clarifying_question")
    if question is not None:
        if not isinstance(question, str) or "?" not in question or len(question) > 300:
            errors.append("clarifying_question must be null or one short question ending in '?'")
            question = None
        else:
            question = question.strip()

    result = {
        "profile": clean,
        "confidence": {k: conf[k] for k in PROFILE_KEYS if clean[k] is not None and k in conf},
        "evidence": {k: evid[k] for k in PROFILE_KEYS if clean[k] is not None and k in evid},
        "search_query_en": query,
        "clarifying_question": question,
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


def cache_key(text, language):
    return hashlib.sha256(json.dumps([text, language, PROMPT_VERSION], ensure_ascii=False).encode()).hexdigest()


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

def understand(text, language):
    """Profile, English search query and clarifying question for one input.

    Returns a dict with profile, confidence, evidence, search_query_en,
    clarifying_question, status ("ok" | "invalid"), errors and _meta."""
    key = cache_key(text, language)
    hit = _cache().get(key)
    if hit:
        return {**hit["result"], "_meta": {**hit["meta"], "cached": True}}

    base = PROMPT.format(language_name=LANGUAGE_NAMES.get(language, language), text=text.strip(),
                         states=", ".join(STATES))
    prompt, attempts, errors, result = base, [], [], None
    for attempt in (1, 2):
        raw, provider, model = _generate(prompt)
        obj, parse_error = _parse(raw)
        result, errors = validate(obj, text) if obj is not None else (None, [parse_error])
        attempts.append({"provider": provider, "model": model, "errors": errors})
        if not errors:
            break
        prompt = base + RETRY_SUFFIX.format(errors="\n".join(f"- {e}" for e in errors))

    if errors:
        result = {"profile": dict.fromkeys(PROFILE_KEYS), "confidence": {}, "evidence": {},
                  "search_query_en": None, "clarifying_question": None}
    result = {**result, "status": "invalid" if errors else "ok", "errors": errors}
    meta = {"prompt_version": PROMPT_VERSION, "language": language, "attempts": attempts,
            "provider": attempts[-1]["provider"], "model": attempts[-1]["model"],
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    _store({"key": key, "text": text, "language": language, "prompt_version": PROMPT_VERSION,
            "result": result, "meta": meta})
    return {**result, "_meta": {**meta, "cached": False}}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Run understand() on one input and print the result.")
    ap.add_argument("text")
    ap.add_argument("--language", default="en", choices=sorted(LANGUAGE_NAMES))
    a = ap.parse_args()
    print(json.dumps(understand(a.text, a.language), ensure_ascii=False, indent=2))

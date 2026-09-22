"""Explanations for matched schemes: one LLM call per query (generate-v1).

explain(profile, other_facts, match_results, language) explains the top
TOP_N match results that are not not_eligible, in the person's language, in
one call through the provider pool that backend/app/understand.py uses
(Gemini flash-lite first, Groq fallback, temperature 0). Results are cached in
data/cache/generate.jsonl, keyed by the prompt and GENERATE_VERSION. For each
scheme it returns

- reason: 1-2 sentences. eligible: the conditions that passed.
  needs_checking: what matches and what must still be confirmed; it never
  says the person qualifies.
- citations: [{"chunk_id", "quote"}], chunks of that scheme, each quote copied
  verbatim from the chunk's raw_text
- relevant_unverified: [{"condition", "fact", "fact_field"}], unverified
  conditions of the scheme (residual conditions and family notes) that bear
  on something the person said, each paired with that fact. A typed fact
  that one of the scheme's conditions already checked and passed cannot be
  paired: residual conditions often restate the typed ones ("farmers aged
  18-59"), and that fact has been checked. Downgrade-only: any pair on an
  eligible scheme makes it needs_checking. Nothing here can make a scheme
  not_eligible or raise its status.

not_eligible results get a reason written in code from the failed
condition(s), with no LLM ("This scheme's age limit is 40-79; you said 38").

Each scheme's answer is validated, and the call is retried once with the
problems listed:
- every citation is a chunk of that scheme and its quote an exact substring
  of the chunk's raw_text (at least MIN_QUOTE_CHARS long, or the whole chunk)
- the status matches the matcher's (needs_checking for an eligible scheme
  with a relevant_unverified pair), and the reason's wording agrees with it:
  no claim of eligibility for needs_checking, no denial for eligible
- user facts mentioned exist in the profile or other_facts: relevant_unverified
  pairs name listed facts; every number and state in the reason comes from
  the person's facts or the scheme's own text; and a relative the reason
  calls "your daughter", "your husband"... must appear in the person's facts
- the reason is 1-2 sentences in the person's language
Per scheme, the first valid answer across the two attempts is used. A scheme
with none gets a template built in code: the status, the passed conditions
and the scheme's first eligibility clause, which is also its citation. When
no provider answers, every scheme gets the template, and nothing is cached.
The code-written reasons and templates exist in English, Hindi and Telugu;
other languages get English.
"""
import hashlib
import json
import re
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from app import understand as u
from app.matcher import CASTE_CATEGORIES, marital_admitted
from app.retrieval import CHUNKS_PATH

ROOT = Path(__file__).resolve().parents[2]
GENERATE_VERSION = "generate-v1"
CACHE_PATH = ROOT / "data" / "cache" / "generate.jsonl"
TOP_N = 5
MAX_CHUNKS = 20                  # eligibility and exclusion chunks shown per scheme
MAX_CHUNK_CHARS = 700
MAX_CONDITION_CHARS = 300
MIN_QUOTE_CHARS = 12
MAX_SENTENCES = 2
MAX_CITATIONS = 3
TEMPLATE_LANGUAGES = ("en", "hi", "te")
_SCRIPT = {"hi": re.compile("[ऀ-ॿ]"), "te": re.compile("[ఀ-౿]"),
           "ta": re.compile("[஀-௿]"), "bn": re.compile("[ঀ-৿]")}


# ---------------------------------------------------------------------------
# Scheme text
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _chunks_by_slug():
    by_slug = {}
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            by_slug.setdefault(c["slug"], []).append(c)
    return by_slug


def scheme_chunks(slug):
    return _chunks_by_slug().get(slug, [])


def _condition_chunks(slug):
    """The chunks shown for citing: eligibility and exclusions, or the
    overview for a scheme that has neither."""
    chunks = scheme_chunks(slug)
    return [c for c in chunks if c["section"] in ("eligibility", "exclusions")] or chunks


def first_clause(slug):
    """The scheme's first eligibility chunk (or first chunk of any kind)."""
    chunks = scheme_chunks(slug)
    return next((c for c in chunks if c["section"] == "eligibility"), chunks[0] if chunks else None)


def _span_citation(slug, span):
    """A citation for a matcher source span, when one chunk contains it verbatim."""
    if not span:
        return None
    c = next((c for c in scheme_chunks(slug) if span in c["raw_text"]), None)
    return {"chunk_id": c["chunk_id"], "quote": span} if c else None


# ---------------------------------------------------------------------------
# Wording for code-written reasons (not_eligible, fallback)
# ---------------------------------------------------------------------------

FIELD_LABELS = {
    "en": {"age": "age", "income": "income", "land": "land", "education_class": "class",
           "education_stage": "course of study", "gender": "gender", "category": "category",
           "residence": "rural or urban residence", "marital_status": "marital status", "state": "state",
           "bpl_household": "BPL status", "registered_construction_worker": "welfare board registration",
           "occupation": "occupation", "not_availing_other_scheme": "other benefits already received",
           "other": "the scheme's other conditions"},
    "hi": {"age": "आयु", "income": "आय", "land": "ज़मीन", "education_class": "कक्षा",
           "education_stage": "पढ़ाई का स्तर", "gender": "लिंग", "category": "वर्ग",
           "residence": "ग्रामीण या शहरी निवास", "marital_status": "वैवाहिक स्थिति", "state": "राज्य",
           "bpl_household": "बीपीएल स्थिति", "registered_construction_worker": "कल्याण बोर्ड में पंजीकरण",
           "occupation": "व्यवसाय", "not_availing_other_scheme": "पहले से मिल रहे लाभ",
           "other": "योजना की बाकी शर्तों"},
    "te": {"age": "వయస్సు", "income": "ఆదాయం", "land": "భూమి", "education_class": "తరగతి",
           "education_stage": "చదువు స్థాయి", "gender": "లింగం", "category": "వర్గం",
           "residence": "గ్రామీణ లేదా పట్టణ నివాసం", "marital_status": "వైవాహిక స్థితి", "state": "రాష్ట్రం",
           "bpl_household": "BPL స్థితి", "registered_construction_worker": "సంక్షేమ బోర్డులో నమోదు",
           "occupation": "వృత్తి", "not_availing_other_scheme": "ఇప్పటికే పొందుతున్న ప్రయోజనాలు",
           "other": "పథకం ఇతర షరతులు"},
}
_AND = {"en": " and ", "hi": " और ", "te": " మరియు "}
_OR = {"en": " or ", "hi": " या ", "te": " లేదా "}
_STOP = {"en": ".", "hi": "।", "te": "."}

FALLBACK = {
    "en": {"eligible": "Every condition we could check is met: {passed}.",
           "partly": "Needs checking: {passed} checked out, but {pending} must still be confirmed.",
           "nothing": "Needs checking: nothing about you could be checked automatically, so {pending} must be "
                      "confirmed.",
           "clause": " The scheme's first condition: “{clause}”"},
    "hi": {"eligible": "जिन शर्तों की जाँच हो सकी, वे सब पूरी होती हैं: {passed}।",
           "partly": "जाँच ज़रूरी: मेल खाती शर्तें — {passed}; लेकिन {pending} की पुष्टि अभी बाकी है।",
           "nothing": "जाँच ज़रूरी: आपकी कोई जानकारी अपने-आप नहीं जाँची जा सकी, इसलिए {pending} की पुष्टि ज़रूरी है।",
           "clause": " योजना की पहली शर्त: “{clause}”"},
    "te": {"eligible": "తనిఖీ చేయగలిగిన అన్ని షరతులు నెరవేరాయి: {passed}.",
           "partly": "తనిఖీ అవసరం: సరిపోయిన షరతులు — {passed}; కానీ {pending} ఇంకా నిర్ధారించాలి.",
           "nothing": "తనిఖీ అవసరం: మీ గురించి ఏదీ స్వయంచాలకంగా తనిఖీ చేయలేకపోయాము, కాబట్టి {pending} నిర్ధారించాలి.",
           "clause": " పథకం మొదటి షరతు: “{clause}”"},
}

# not_eligible: "<rule>; <what you said>." per failed condition.
_RULE = {
    "en": {"age_range": "This scheme's age limit is {lo}-{hi}", "age_exact": "This scheme is for people aged {lo}",
           "age_min": "This scheme's minimum age is {lo}",
           "age_max": "This scheme's maximum age is {hi}",
           "income": "This scheme's limit on {basis}income is ₹{cap} a year",
           "land_range": "This scheme is for {lo}-{hi} acres of land", "land_min": "This scheme needs at least {lo} acres of land",
           "land_max": "This scheme is for up to {hi} acres of land",
           "class_range": "This scheme is for students in class {lo}-{hi}", "class_min": "This scheme is for students in class {lo} or above",
           "class_max": "This scheme is for students up to class {hi}",
           "gender": "This scheme is only for {g}", "category": "This scheme is only for: {cats}",
           "residence": "This scheme is only for people living in {r} areas",
           "marital_status": "This scheme is only for these marital statuses: {ms}",
           "state": "This scheme is only for residents of {state}",
           "bpl_household": "This scheme is only for BPL (below poverty line) households",
           "registered_construction_worker": "This scheme is only for workers registered with the construction "
                                             "workers' welfare board"},
    "hi": {"age_range": "इस योजना की आयु सीमा {lo}-{hi} वर्ष है", "age_exact": "यह योजना {lo} वर्ष की आयु के लोगों के लिए है",
           "age_min": "इस योजना के लिए न्यूनतम आयु {lo} वर्ष है",
           "age_max": "इस योजना के लिए अधिकतम आयु {hi} वर्ष है",
           "income": "इस योजना में {basis}वार्षिक आय की सीमा ₹{cap} है",
           "land_range": "यह योजना {lo}-{hi} एकड़ ज़मीन वालों के लिए है", "land_min": "इस योजना के लिए कम से कम {lo} एकड़ ज़मीन चाहिए",
           "land_max": "यह योजना {hi} एकड़ तक ज़मीन वालों के लिए है",
           "class_range": "यह योजना कक्षा {lo}-{hi} के विद्यार्थियों के लिए है",
           "class_min": "यह योजना कक्षा {lo} या उससे ऊपर के विद्यार्थियों के लिए है",
           "class_max": "यह योजना कक्षा {hi} तक के विद्यार्थियों के लिए है",
           "gender": "यह योजना केवल {g} के लिए है", "category": "यह योजना केवल इन वर्गों के लिए है: {cats}",
           "residence": "यह योजना केवल {r} क्षेत्रों में रहने वालों के लिए है",
           "marital_status": "यह योजना केवल इन वैवाहिक स्थितियों के लिए है: {ms}",
           "state": "यह योजना केवल {state} के निवासियों के लिए है",
           "bpl_household": "यह योजना केवल बीपीएल (गरीबी रेखा से नीचे) परिवारों के लिए है",
           "registered_construction_worker": "यह योजना केवल निर्माण श्रमिक कल्याण बोर्ड में पंजीकृत श्रमिकों के लिए है"},
    "te": {"age_range": "ఈ పథకం వయో పరిమితి {lo}-{hi} సంవత్సరాలు", "age_exact": "ఈ పథకం {lo} సంవత్సరాల వయస్సు ఉన్నవారికి",
           "age_min": "ఈ పథకానికి కనీస వయస్సు {lo} సంవత్సరాలు",
           "age_max": "ఈ పథకానికి గరిష్ఠ వయస్సు {hi} సంవత్సరాలు",
           "income": "ఈ పథకంలో వార్షిక {basis}ఆదాయ పరిమితి ₹{cap}",
           "land_range": "ఈ పథకం {lo}-{hi} ఎకరాల భూమి ఉన్నవారికి", "land_min": "ఈ పథకానికి కనీసం {lo} ఎకరాల భూమి ఉండాలి",
           "land_max": "ఈ పథకం {hi} ఎకరాల వరకు భూమి ఉన్నవారికి",
           "class_range": "ఈ పథకం {lo}-{hi} తరగతుల విద్యార్థులకు", "class_min": "ఈ పథకం {lo}వ తరగతి లేదా అంతకంటే పై విద్యార్థులకు",
           "class_max": "ఈ పథకం {hi}వ తరగతి వరకు విద్యార్థులకు",
           "gender": "ఈ పథకం {g} మాత్రమే", "category": "ఈ పథకం ఈ వర్గాలకు మాత్రమే: {cats}",
           "residence": "ఈ పథకం {r} ప్రాంతాల్లో నివసించే వారికి మాత్రమే",
           "marital_status": "ఈ పథకం ఈ వైవాహిక స్థితులకు మాత్రమే: {ms}",
           "state": "ఈ పథకం {state} నివాసులకు మాత్రమే",
           "bpl_household": "ఈ పథకం BPL (దారిద్ర్య రేఖకు దిగువన ఉన్న) కుటుంబాలకు మాత్రమే",
           "registered_construction_worker": "ఈ పథకం భవన నిర్మాణ కార్మికుల సంక్షేమ బోర్డులో నమోదైన కార్మికులకు మాత్రమే"},
}
_SAID = {
    "en": {"age": "you said {v}", "income": "you said ₹{v}", "land": "you said {v} acres", "education_class": "you said class {v}",
           "gender": "you said you are {g}", "residence": "you said you live in a {r} area",
           "marital_status": "you said you are {ms}", "state": "you said you live in {v}",
           "bpl_household": "you said you are not BPL", "registered_construction_worker": "you said you are not registered",
           "caste": "you said your category is {v}", "minority": "you said you are not from a religious minority",
           "disability": "you said your disability is {v}%"},
    "hi": {"age": "आपने अपनी आयु {v} वर्ष बताई है", "income": "आपने ₹{v} बताया है", "land": "आपने {v} एकड़ बताया है",
           "education_class": "आपने कक्षा {v} बताई है", "gender": "आपने बताया कि आप {g} हैं",
           "residence": "आपने बताया कि आप {r} क्षेत्र में रहते हैं", "marital_status": "आपने अपनी वैवाहिक स्थिति '{ms}' बताई है",
           "state": "आपने बताया कि आप {v} में रहते हैं", "bpl_household": "आपने बताया कि आप बीपीएल नहीं हैं",
           "registered_construction_worker": "आपने बताया कि आप पंजीकृत नहीं हैं", "caste": "आपने अपना वर्ग {v} बताया है",
           "minority": "आपने बताया कि आप अल्पसंख्यक समुदाय से नहीं हैं", "disability": "आपने अपनी विकलांगता {v}% बताई है"},
    "te": {"age": "మీరు మీ వయస్సు {v} సంవత్సరాలు అని చెప్పారు", "income": "మీరు ₹{v} అని చెప్పారు",
           "land": "మీరు {v} ఎకరాలు అని చెప్పారు", "education_class": "మీరు {v}వ తరగతి అని చెప్పారు",
           "gender": "మీరు {g} అని చెప్పారు", "residence": "మీరు {r} ప్రాంతంలో నివసిస్తున్నారని చెప్పారు",
           "marital_status": "మీరు మీ వైవాహిక స్థితి '{ms}' అని చెప్పారు", "state": "మీరు {v}లో నివసిస్తున్నారని చెప్పారు",
           "bpl_household": "మీరు BPL కాదని చెప్పారు", "registered_construction_worker": "మీరు నమోదు కాలేదని చెప్పారు",
           "caste": "మీరు మీ వర్గం {v} అని చెప్పారు", "minority": "మీరు మైనారిటీ వర్గానికి చెందరని చెప్పారు",
           "disability": "మీరు మీ వైకల్యం {v}% అని చెప్పారు"},
}
_GENDER_GROUP = {"en": {"female": "women", "male": "men", "transgender": "transgender persons"},
                 "hi": {"female": "महिलाओं", "male": "पुरुषों", "transgender": "ट्रांसजेंडर व्यक्तियों"},
                 "te": {"female": "మహిళలకు", "male": "పురుషులకు", "transgender": "ట్రాన్స్‌జెండర్ వ్యక్తులకు"}}
_GENDER_ONE = {"en": {"female": "a woman", "male": "a man", "transgender": "transgender"},
               "hi": {"female": "महिला", "male": "पुरुष", "transgender": "ट्रांसजेंडर"},
               "te": {"female": "మహిళ", "male": "పురుషుడు", "transgender": "ట్రాన్స్‌జెండర్"}}
_RESIDENCE = {"en": {"rural": "rural", "urban": "urban"}, "hi": {"rural": "ग्रामीण", "urban": "शहरी"},
              "te": {"rural": "గ్రామీణ", "urban": "పట్టణ"}}
_MARITAL = {"en": {"married": "married", "unmarried": "unmarried", "widowed": "widowed", "divorced": "divorced",
                   "separated": "separated", "abandoned": "abandoned"},
            "hi": {"married": "विवाहित", "unmarried": "अविवाहित", "widowed": "विधवा/विधुर", "divorced": "तलाकशुदा",
                   "separated": "अलग रह रहे", "abandoned": "परित्यक्त"},
            "te": {"married": "వివాహిత", "unmarried": "అవివాహిత", "widowed": "వితంతు", "divorced": "విడాకులు పొందిన",
                   "separated": "విడిగా ఉంటున్న", "abandoned": "వదిలివేయబడిన"}}
_CATEGORY = {"en": {"General": "General", "Minority": "religious minorities",
                    "PwD": "persons with a disability of 40% or more"},
             "hi": {"General": "सामान्य वर्ग", "Minority": "अल्पसंख्यक",
                    "PwD": "40% या अधिक विकलांगता वाले व्यक्ति"},
             "te": {"General": "జనరల్ వర్గం", "Minority": "మైనారిటీలు",
                    "PwD": "40% లేదా అంతకంటే ఎక్కువ వైకల్యం ఉన్నవారు"}}
_FAMILY_BASIS = {"en": "family ", "hi": "पारिवारिक ", "te": "కుటుంబ "}
# Profile fields a passed condition has checked.
_CHECKED_KEYS = {"age": ("age",), "income": ("annual_income_inr",), "land": ("land_acres",),
                 "education_class": ("education_class",), "education_stage": ("education_stage", "education_class"),
                 "gender": ("gender",), "category": ("caste_category", "is_minority", "disability_percent"),
                 "residence": ("residence",), "marital_status": ("marital_status",), "state": ("state",),
                 "bpl_household": ("bpl_household",), "registered_construction_worker": ("registered_construction_worker",),
                 "occupation": ("occupation",)}


def _lang(language):
    return language if language in TEMPLATE_LANGUAGES else "en"


def _num(v):
    return str(int(v)) if float(v) == int(v) else f"{v:g}"


def _inr(v):
    """Indian digit grouping: 120000 -> 1,20,000."""
    s = str(int(round(v)))
    head, tail = s[:-3], s[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    return ",".join(([head] if head else []) + groups + [tail])


def _join(items, lang, sep=None):
    items = [i for i in items if i]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + (sep or _AND)[lang] + items[-1]


def _bounds(prefix, lo, hi):
    if lo is not None and hi is not None:
        return f"{prefix}_range"
    return f"{prefix}_min" if lo is not None else f"{prefix}_max"


def _failed_sentence(c, lang):
    rule, said, field, con, v = _RULE[lang], _SAID[lang], c["field"], c["constraint"], c["profile_value"]
    if field == "age":
        which = "age_exact" if con["min"] is not None and con["min"] == con["max"] else _bounds("age", con["min"], con["max"])
        r = rule[which].format(lo=con["min"], hi=con["max"])
        s = said["age"].format(v=_num(v))
    elif field == "income":
        basis = _FAMILY_BASIS[lang] if con.get("basis") == "family" else ""
        r, s = rule["income"].format(basis=basis, cap=_inr(con["max_annual_inr"])), said["income"].format(v=_inr(v))
    elif field == "land":
        lo, hi = con["min_acres"], con["max_acres"]
        r = rule[_bounds("land", lo, hi)].format(lo=_num(lo) if lo is not None else "", hi=_num(hi) if hi is not None else "")
        s = said["land"].format(v=_num(v))
    elif field == "education_class":
        lo, hi = con["min_class"], con["max_class"]
        r, s = rule[_bounds("class", lo, hi)].format(lo=lo, hi=hi), said["education_class"].format(v=_num(v))
    elif field == "gender":
        r = rule["gender"].format(g=_GENDER_GROUP[lang].get(con, con))
        s = said["gender"].format(g=_GENDER_ONE[lang].get(v, v))
    elif field == "category":
        r = rule["category"].format(cats=_join([_CATEGORY[lang].get(x, x) for x in con], lang, _OR))
        shown, parts = v or {}, []
        if shown.get("caste_category") and any(x in CASTE_CATEGORIES for x in con):
            parts.append(said["caste"].format(v=shown["caste_category"]))
        if shown.get("is_minority") is False and "Minority" in con:
            parts.append(said["minority"])
        if shown.get("disability_percent") is not None and "PwD" in con:
            parts.append(said["disability"].format(v=_num(shown["disability_percent"])))
        s = _join(parts, lang)
    elif field == "residence":
        r = rule["residence"].format(r=_RESIDENCE[lang].get(con, con))
        s = said["residence"].format(r=_RESIDENCE[lang].get(v, v))
    elif field == "marital_status":
        admitted = marital_admitted(con)
        ms = _join([_MARITAL[lang][m] for m in sorted(admitted)], lang, _OR) if isinstance(admitted, set) else con
        r, s = rule["marital_status"].format(ms=ms), said["marital_status"].format(ms=_MARITAL[lang].get(v, v))
    elif field == "state":
        r, s = rule["state"].format(state=con), said["state"].format(v=v)
    elif field in ("bpl_household", "registered_construction_worker"):
        r, s = rule[field], said[field]
    else:
        raise ValueError(f"no not_eligible wording for field {field!r}")
    return f"{r}; {s}{_STOP[lang]}" if s else f"{r}{_STOP[lang]}"


def not_eligible_reason(result, language):
    """Reason for a not_eligible result, written from its failed conditions."""
    lang = _lang(language)
    failed = [c for c in result["conditions"] if c["decisive"] and c["result"] == "fail"]
    return " ".join(_failed_sentence(c, lang) for c in failed), failed


def _passed(result):
    return [c for c in result["conditions"] if c["decisive"] and c["result"] == "pass"
            and not (c["field"] == "state" and c["constraint"] == "central scheme")]


def _label(c, lang):
    label = FIELD_LABELS[lang].get(c["field"], c["field"])
    v = c["profile_value"]
    if c["field"] == "income" and isinstance(v, (int, float)):
        return f"{label} (₹{_inr(v)})"
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return f"{label} ({_num(v)})"
    if isinstance(v, str):
        return f"{label} ({v})"
    return label


def fallback_reason(result, language):
    """The code template: status, passed conditions, first eligibility clause."""
    lang = _lang(language)
    t = FALLBACK[lang]
    passed = _join([_label(c, lang) for c in _passed(result)], lang)
    pending = [FIELD_LABELS[lang].get(c["field"], c["field"])
               for c in result["conditions"] if c["decisive"] and c["result"] == "unknown"]
    if result.get("own_conditions") or result.get("unverified_conditions") or not pending:
        pending.append(FIELD_LABELS[lang]["other"])
    pending = _join(list(dict.fromkeys(pending)), lang)
    if result["status"] == "eligible":
        text = t["eligible"].format(passed=passed)
    else:
        text = (t["partly"] if passed else t["nothing"]).format(passed=passed, pending=pending)
    clause = first_clause(result["slug"])
    citations = []
    if clause:
        shown = clause["raw_text"] if len(clause["raw_text"]) <= 200 else clause["raw_text"][:200].rstrip() + "…"
        text += t["clause"].format(clause=shown)
        citations.append({"chunk_id": clause["chunk_id"], "quote": clause["raw_text"]})
    return text, citations


# ---------------------------------------------------------------------------
# Status wording checks
# ---------------------------------------------------------------------------

_CLAIM = {
    "en": re.compile(r"\byou(?:'re|\s+are)\s+(?:(\w+)\s+)?(?:eligible|qualified|entitled)\b|\byou\s+(?:(\w+)\s+)?"
                     r"qualify\b|\byou\s+(?:will|shall)\s+(?:(\w+)\s+)?(?:get|receive)\b", re.I),
    "hi": re.compile(r"(?:पात्र|योग्य|हकदार|हक़दार)\s*(?:हैं|है|हो|होंगे|होंगी|होगा|होगी)(?![\u0900-\u0963\u0971-\u097F])|"
                     r"लाभ\s+(?:मिलेगा|मिल\s+जाएगा|ले\s+सकते)"),
    "te": re.compile(r"అర్హులు(?![\u0C00-\u0C7F])(?!\s*కా)|అర్హత\s+(?:ఉంది|కలదు)|ప్రయోజనం\s+(?:పొందుతారు|లభిస్తుంది)"),
}
_HEDGE = {
    "en": re.compile(r"\b(?:may|might|could|would|if|whether|once|until|unless|before|after|confirm\w*|check\w*|"
                     r"verif\w*|not|possibly|potentially|provided|only)\b|n't", re.I),
    "hi": re.compile(r"सकते|सकती|सकता|नहीं|यदि|अगर|पुष्टि|जाँच|जांच|बाद|शायद|संभव|तभी"),
    "te": re.compile(r"కావచ్చు|అవుతారో|అర్హులా|కాదు|కారు|ఉంటే|నిర్ధారించ|తనిఖీ|తర్వాత|ఉండవచ్చు|లేదో|ఏమో"),
}
_DENIAL = {
    "en": re.compile(r"\bnot\s+(?:be\s+)?eligible\b|\bineligible\b|\b(?:do|does)\s+not\s+qualify\b|\b(?:don|doesn)'t\s+qualify\b",
                     re.I),
    "hi": re.compile(r"(?:पात्र|योग्य)\s+नहीं"),
    "te": re.compile(r"అర్హులు\s+కా(?:రు|దు)"),
}


# Relatives a reason may call "your ...": (English, Hindi, Telugu, words in the
# person's facts that support the mention).
_RELATIVES = {
    "daughter": (r"\byour\s+(?:\w+\s+)?daughters?\b", r"आपकी\s+(?:बेटी|बेटियों|पुत्री)", r"మీ\s+(?:కూతురు|కుమార్తె)",
                 ("daughter", "girl")),
    "son": (r"\byour\s+(?:\w+\s+)?sons?\b", r"आपके\s+(?:बेटे|पुत्र)|आपका\s+बेटा", r"మీ\s+(?:కొడుకు|కుమారుడు)", ("son", "boy")),
    "child": (r"\byour\s+(?:\w+\s+)?(?:child|children|kids?|baby)\b", r"आपके\s+बच्च|आपका\s+बच्चा|आपकी\s+संतान",
              r"మీ\s+(?:పిల్ల|బిడ్డ)", ("child", "kid", "baby", "son", "daughter", "girl", "boy", "year-old", "pregnan")),
    "spouse": (r"\byour\s+(?:wife|husband|spouse)\b", r"आपकी\s+पत्नी|आपके\s+पति", r"మీ\s+(?:భార్య|భర్త)",
               ("wife", "husband", "spouse", "married")),
    "parent": (r"\byour\s+(?:mother|father|parents?)\b", r"आपकी\s+(?:माँ|मां|माता)|आपके\s+(?:पिता|माता-पिता)",
               r"మీ\s+(?:తల్లి|తండ్రి|తల్లిదండ్రులు)", ("mother", "father", "parent")),
}
_LANG_INDEX = {"en": 0, "hi": 1, "te": 2}


def unsupported_relatives(text, language, fact_texts):
    """Relatives the text calls "your ..." that nothing in fact_texts mentions."""
    i = _LANG_INDEX.get(language)
    if i is None:
        return []
    facts = " ".join(fact_texts).lower()
    return [who for who, spec in _RELATIVES.items()
            if re.search(spec[i], text or "", re.I) and not any(w in facts for w in spec[3])]


def _sentence_of(text, start, end):
    """The sentence around text[start:end]."""
    left = max(text.rfind(p, 0, start) for p in ".!?।") + 1
    rights = [i for i in (text.find(p, end) for p in ".!?।") if i != -1]
    return text[left:start], text[end:min(rights) if rights else len(text)]


def claims_eligibility(text, language):
    """Unhedged phrases saying the person qualifies ("you are eligible", but
    not "you may be eligible" or "to confirm you qualify")."""
    lang = _lang(language)
    found = []
    for m in _CLAIM[lang].finditer(text or ""):
        before, after = _sentence_of(text, m.start(), m.end())
        middle = " ".join(g for g in m.groups() if g) if m.groups() else ""
        if _HEDGE[lang].search(before[-60:]) or _HEDGE[lang].search(after[:30]) or _HEDGE[lang].search(middle):
            continue
        found.append(m.group())
    return found


def denies_eligibility(text, language):
    return [m.group() for m in _DENIAL[_lang(language)].finditer(text or "")]


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

PROMPT = """You explain welfare scheme results to one person in India. Write every "reason" in {language_name}.

What the person told us. These are the only facts about them you may use:
{facts}

Rules have already checked each scheme below against these facts. The statuses are final, except as stated under "relevant_unverified".

{schemes}

Return ONLY a JSON object, no markdown: {{"schemes": [one object per scheme above, in the same order]}}. Each object has exactly these keys:
"id": the scheme's id (S1, S2, ...).
"relevant_unverified": the scheme's unchecked conditions (R ids) that bear on something the person told us (F ids): the fact may meet the condition, break it, or leave it open. A list of {{"condition": "R<n>", "fact": "F<n>"}}, or an empty list. Pair a condition and a fact only when both are about the same thing. Never use a fact listed under that scheme's "Facts already checked".
"status": the status given above, except that an "eligible" scheme with any relevant_unverified entry becomes "needs_checking".
"reason": one or two short sentences in {language_name}, speaking to the person as "you". For "eligible": name the conditions that were checked and passed. For "needs_checking": say what matches and what must still be confirmed (the conditions that could not be checked, and any relevant_unverified ones). For "needs_checking", never say or suggest that they qualify, are eligible or will get the benefit. Use only the facts listed above and the scheme's own text: add nothing else about the person, and no number that appears in neither. Do not name any other scheme. If one of their facts names a scheme they asked about, do not suggest that this scheme is that one.
"citations": one to three chunks of this scheme that support the reason, each {{"chunk_id": the id in square brackets, "quote": a sentence or phrase copied character for character from that chunk}}.

Example of one object, for an English speaker:
{{"id": "S2", "relevant_unverified": [{{"condition": "R2", "fact": "F4"}}], "status": "needs_checking", "reason": "Your age and your state meet this scheme's rules. It still has to be confirmed that the land is in your name, and the scheme's rule on earlier loans applies to the loan you mentioned.", "citations": [{{"chunk_id": "abc::eligibility::001", "quote": "The applicant must own agricultural land in his/her name."}}]}}
"""

RETRY_SUFFIX = """
Your previous answer had these problems:
{errors}
Return the corrected JSON object for all schemes."""


def _fact_value(key, v):
    if key == "annual_income_inr":
        return f"annual income ₹{_inr(v)}"
    if key == "family":
        return f"family: {v.replace('_', ' ')}"
    if key == "prior_benefit_schemes":
        return "already received: " + ", ".join(v)
    if key == "applying_for":
        return "asking for: " + ("themself" if v == "self" else f"their {v}")
    if isinstance(v, bool):
        return f"{key.replace('_', ' ')}: {'yes' if v else 'no'}"
    return f"{key.replace('_', ' ')}: {v}"


def person_facts(profile, other_facts):
    """[(F id, field, text)] for everything the person said."""
    facts = []
    for key in u.PROFILE_KEYS:
        v = profile.get(key)
        if v not in (None, []) and not (key == "applying_for" and v == "self"):   # self is the default
            facts.append((f"F{len(facts) + 1}", key, _fact_value(key, v)))
    for f in other_facts or []:
        text = f["en"] if isinstance(f, dict) else f
        facts.append((f"F{len(facts) + 1}", "other_facts", text))
    return facts


def _why_unknown(c):
    if c["ask_kind"] == "missing":
        return "not stated"
    if c["ask_kind"] == "refine":
        return "stated, but too vague to check"
    return "could not be decided automatically"


def _scheme_block(sid, r, rids, checked):
    where = "central scheme" if r.get("level") == "central" else f"{r.get('level') or 'scheme'}, {r.get('state')}"
    lines = [f'{sid} "{r["scheme_name"]}" ({where}). Status: {r["status"]}. Why: {r["reason"]}.']
    passed = _passed(r)
    lines.append("  Checked and passed: " + ("; ".join(
        f"{c['field']} (rule: {json.dumps(c['constraint'], ensure_ascii=False)}; person: "
        f"{json.dumps(c['profile_value'], ensure_ascii=False)})" for c in passed) or "none"))
    unknown = [c for c in r["conditions"] if c["decisive"] and c["result"] == "unknown"]
    lines.append("  Could not check: " + ("; ".join(
        f"{c['field']} (rule: {json.dumps(c['constraint'], ensure_ascii=False)}; {_why_unknown(c)})"
        for c in unknown) or "none"))
    if checked:
        lines.append("  Facts already checked (not for relevant_unverified): " + ", ".join(checked))
    if r.get("own_conditions"):
        lines.append("  The person is registered with the welfare board, but this scheme's own conditions "
                     "(the R conditions below) must be confirmed.")
    if r.get("to_confirm"):
        lines.append("  Also to confirm: " + "; ".join(r["to_confirm"]))
    if r.get("preferences"):
        lines.append("  Given priority (not a condition): " + "; ".join(
            f"{p['field']} {json.dumps(p['value'], ensure_ascii=False)}" for p in r["preferences"]))
    lines.append("  Unchecked conditions:" + ("" if rids else " none"))
    lines += [f"    {rid}: {text[:MAX_CONDITION_CHARS]}" for rid, text in rids]
    lines.append("  Scheme text (cite from these chunks):")
    lines += [f"    [{c['chunk_id']}] {c['raw_text'][:MAX_CHUNK_CHARS]}" for c in _condition_chunks(r["slug"])[:MAX_CHUNKS]]
    return "\n".join(lines)


def _contexts(covered, facts):
    out = []
    for i, r in enumerate(covered, 1):
        keys = {k for c in _passed(r) for k in _CHECKED_KEYS.get(c["field"], ())}
        out.append({"sid": f"S{i}", "result": r,
                    "rids": [(f"R{j}", t) for j, t in enumerate(dict.fromkeys(r.get("unverified_conditions") or []), 1)],
                    "checked": [fid for fid, field, _ in facts if field in keys]})
    return out


def build_prompt(facts, contexts, language):
    return PROMPT.format(
        language_name=u.LANGUAGE_NAMES.get(language, language),
        facts="\n".join(f"{fid} {text}" for fid, _, text in facts) or "(nothing)",
        schemes="\n\n".join(_scheme_block(x["sid"], x["result"], x["rids"], x["checked"]) for x in contexts))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _digit_numbers(text):
    """Numbers written with digits (any script); number words are left out."""
    return u._numbers(re.sub(r"[A-Za-z]+", " ", text or ""))


def _allowed_numbers(facts, profile, r):
    texts = [t for _, _, t in facts] + [r["scheme_name"] or ""] + list(r.get("unverified_conditions") or [])
    texts += [c["raw_text"] for c in scheme_chunks(r["slug"])]
    texts += [json.dumps(c["constraint"], ensure_ascii=False) + " " + (c["source_span"] or "") for c in r["conditions"]]
    nums = [n for t in texts for n in u._numbers(t)]
    nums += [float(v) for v in profile.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return nums


def _check_scheme(entry, x, facts, profile, language, stats):
    """(cleaned entry, errors) for one scheme's answer."""
    r, sid = x["result"], x["sid"]
    if entry is None:
        return None, [f"{sid}: missing from the answer"]
    errors = []
    keys = {"id", "relevant_unverified", "status", "reason", "citations"}
    if set(entry) != keys:
        errors.append(f"{sid}: keys must be exactly {sorted(keys)}; got {sorted(entry)}")
    rids, fids = dict(x["rids"]), {fid: (field, text) for fid, field, text in facts}

    flags = []
    raw_flags = entry.get("relevant_unverified")
    if not isinstance(raw_flags, list):
        errors.append(f"{sid}: relevant_unverified must be a list")
        raw_flags = []
    for f in raw_flags:
        if not (isinstance(f, dict) and set(f) == {"condition", "fact"}):
            errors.append(f'{sid}: relevant_unverified entries must be {{"condition": "R<n>", "fact": "F<n>"}}; got {f!r}')
        elif f["condition"] not in rids:
            errors.append(f"{sid}: {f['condition']!r} is not one of this scheme's unchecked conditions {sorted(rids)}")
        elif f["fact"] not in fids:
            errors.append(f"{sid}: {f['fact']!r} is not one of the person's facts {sorted(fids)}")
        elif f["fact"] in x["checked"]:
            errors.append(f"{sid}: {f['fact']} was already checked and passed for this scheme, so it cannot be "
                          f"paired in relevant_unverified")
        else:
            pair = {"condition": rids[f["condition"]], "fact": fids[f["fact"]][1], "fact_field": fids[f["fact"]][0]}
            if pair not in flags:
                flags.append(pair)

    expected = "needs_checking" if r["status"] == "eligible" and flags else r["status"]
    if entry.get("status") != expected:
        errors.append(f"{sid}: status must be {expected!r}"
                      + (" (it is eligible, but relevant_unverified is not empty)" if flags else "")
                      + f"; got {entry.get('status')!r}")

    reason = entry.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        errors.append(f"{sid}: reason must be a non-empty string")
        reason = ""
    reason = re.sub(r"\s+", " ", reason).strip()
    if reason:
        script = _SCRIPT.get(language)
        if language == "en" and u._INDIC_RE.search(reason):
            errors.append(f"{sid}: reason must be in English")
        elif script and not script.search(reason):
            errors.append(f"{sid}: reason must be in {u.LANGUAGE_NAMES.get(language, language)}")
        if u._sentence_count(reason) > MAX_SENTENCES:
            errors.append(f"{sid}: reason must be one or two sentences; it has {u._sentence_count(reason)}")
        if expected != "eligible" and claims_eligibility(reason, language):
            errors.append(f"{sid}: the status is {expected}, so the reason must not say they qualify; it says "
                          f"{claims_eligibility(reason, language)}")
        if expected == "eligible" and denies_eligibility(reason, language):
            errors.append(f"{sid}: the status is eligible, but the reason says {denies_eligibility(reason, language)}")
        allowed = _allowed_numbers(facts, profile, r)
        invented = [n for n in _digit_numbers(reason) if not u._derivable(n, allowed)]
        if invented:
            errors.append(f"{sid}: the reason has numbers that are in neither the person's facts nor the scheme's "
                          f"text: {[_num(n) for n in invented]}")
        relatives = unsupported_relatives(reason, language, [t for _, _, t in facts])
        if relatives:
            errors.append(f"{sid}: the reason speaks of your {', '.join(relatives)}, but the person said nothing "
                          f"about one; mention only the people in their facts")
        scheme_text = " ".join([r["scheme_name"] or ""] + [c["raw_text"] for c in scheme_chunks(r["slug"])]).casefold()
        for s in u.STATES:
            if re.search(rf"\b{re.escape(s)}\b", reason, re.I) and s not in (profile.get("state"), r.get("state")) \
                    and s.casefold() not in scheme_text:
                errors.append(f"{sid}: the reason names {s}, which is neither the person's state nor in the scheme's text")

    citations = []
    raw_cites = entry.get("citations")
    if not isinstance(raw_cites, list) or not 1 <= len(raw_cites) <= MAX_CITATIONS:
        errors.append(f"{sid}: citations must be a list of 1 to {MAX_CITATIONS} entries")
        raw_cites = raw_cites if isinstance(raw_cites, list) else []
    own = {c["chunk_id"]: c["raw_text"] for c in scheme_chunks(r["slug"])}
    for c in raw_cites:
        stats["total"] += 1
        if not (isinstance(c, dict) and set(c) == {"chunk_id", "quote"} and isinstance(c["quote"], str)):
            errors.append(f'{sid}: citations must be {{"chunk_id": ..., "quote": ...}}; got {c!r}')
            continue
        quote = c["quote"].strip()
        if c["chunk_id"] not in own:
            errors.append(f"{sid}: citation {c['chunk_id']!r} is not a chunk of this scheme")
        elif not quote or quote not in own[c["chunk_id"]]:
            errors.append(f"{sid}: the quote for {c['chunk_id']} is not copied exactly from that chunk: {quote[:80]!r}")
        elif len(quote) < min(MIN_QUOTE_CHARS, len(own[c["chunk_id"]].strip())):
            errors.append(f"{sid}: the quote for {c['chunk_id']} is too short to support anything: {quote!r}")
        else:
            stats["valid"] += 1
            citations.append({"chunk_id": c["chunk_id"], "quote": quote})

    if errors:
        return None, errors
    return {"status": expected, "reason": reason, "citations": citations, "relevant_unverified": flags}, []


def validate(obj, contexts, facts, profile, language):
    """({sid: cleaned entry or None}, {sid: errors}, structural errors, citation stats)."""
    stats = {"total": 0, "valid": 0}
    if not isinstance(obj, dict) or not isinstance(obj.get("schemes"), list):
        return {}, {}, ['the answer must be a JSON object {"schemes": [...]}'], stats
    structural = [f"unknown top-level keys: {sorted(set(obj) - {'schemes'})}"] if set(obj) - {"schemes"} else []
    given = {}
    for e in obj["schemes"]:
        if isinstance(e, dict) and isinstance(e.get("id"), str) and e["id"] not in given:
            given[e["id"]] = e
    unknown = sorted(set(given) - {x["sid"] for x in contexts})
    if unknown:
        structural.append(f"unknown scheme ids: {unknown}")
    cleaned, errors = {}, {}
    for x in contexts:
        cleaned[x["sid"]], errors[x["sid"]] = _check_scheme(given.get(x["sid"]), x, facts, profile, language, stats)
    return cleaned, errors, structural, stats


# ---------------------------------------------------------------------------
# Cache and entry point
# ---------------------------------------------------------------------------

def cache_key(prompt):
    return hashlib.sha256(json.dumps([GENERATE_VERSION, prompt], ensure_ascii=False).encode()).hexdigest()


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


def _entry(r, **kw):
    return {"rank": r.get("rank"), "slug": r["slug"], "scheme_name": r["scheme_name"], "matcher_status": r["status"],
            **kw}


def _explain_covered(profile, other_facts, covered, language):
    facts = person_facts(profile, other_facts)
    contexts = _contexts(covered, facts)
    base = build_prompt(facts, contexts, language)
    key = cache_key(base)
    hit = _cache().get(key)
    if hit:
        return hit["result"], {**hit["meta"], "cached": True}

    chosen, attempts, prompt, provider_error = {}, [], base, None
    for _ in (1, 2):
        try:
            raw, provider, model = u._generate(prompt)
        except u.UnderstandError as e:
            provider_error = str(e)
            break
        obj, parse_error = u._parse(raw)
        if obj is None:
            cleaned, errors, structural, stats = {}, {}, [parse_error], {"total": 0, "valid": 0}
        else:
            cleaned, errors, structural, stats = validate(obj, contexts, facts, profile, language)
        attempts.append({"provider": provider, "model": model, "structural_errors": structural,
                         "scheme_errors": {sid: e for sid, e in errors.items() if e}, "citations": stats})
        for sid, entry in cleaned.items():
            if entry is not None and sid not in chosen:
                chosen[sid] = entry
        problems = structural + [e for sid, errs in errors.items() if sid not in chosen for e in errs]
        if not problems and len(chosen) == len(contexts):
            break
        if not problems:            # nothing to report, but some scheme is still missing
            problems = [f"{x['sid']}: missing from the answer" for x in contexts if x["sid"] not in chosen]
        prompt = base + RETRY_SUFFIX.format(errors="\n".join(f"- {p}" for p in problems))

    entries = {}
    for x in contexts:
        r = x["result"]
        if x["sid"] in chosen:
            entries[r["slug"]] = _entry(r, **chosen[x["sid"]], source="llm", fallback_reason=None)
        else:
            text, citations = fallback_reason(r, language)
            why = f"no provider answered: {provider_error}" if provider_error else "no valid answer after the retry"
            entries[r["slug"]] = _entry(r, status=r["status"], reason=text, citations=citations,
                                        relevant_unverified=[], source="fallback", fallback_reason=why)
    meta = {"version": GENERATE_VERSION, "language": language, "attempts": attempts, "provider_error": provider_error,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    if not provider_error:
        _store({"key": key, "version": GENERATE_VERSION, "language": language, "result": entries, "meta": meta})
    return entries, {**meta, "cached": False}


def explain(profile, other_facts, match_results, language):
    """Explanations for match_results (matcher.match() output, best first).

    Returns {"language", "schemes": [...], "_meta"}: one entry per explained
    result, in rank order: the top TOP_N results that are not not_eligible
    (source "llm", or "fallback" when no valid answer came back) and every
    not_eligible result (source "code"). Each entry has rank, slug,
    scheme_name, matcher_status, status (matcher_status, or needs_checking
    after a relevant_unverified flag), reason, citations, relevant_unverified,
    source and fallback_reason. _meta is None when no LLM call was needed."""
    by_slug = {}
    for r in match_results:
        if r["status"] == "not_eligible":
            reason, failed = not_eligible_reason(r, language)
            citations = [c for c in (_span_citation(r["slug"], f["source_span"]) for f in failed) if c]
            by_slug[r["slug"]] = _entry(r, status="not_eligible", reason=reason, citations=citations,
                                        relevant_unverified=[], source="code", fallback_reason=None)
    covered = [r for r in match_results if r["status"] != "not_eligible"][:TOP_N]
    meta = None
    if covered:
        entries, meta = _explain_covered(profile, other_facts, covered, language)
        for r in covered:           # a cached answer may come from a list with other ranks
            by_slug[r["slug"]] = {**entries[r["slug"]], "rank": r.get("rank")}
    return {"language": language, "schemes": [by_slug[r["slug"]] for r in match_results if r["slug"] in by_slug],
            "_meta": meta}

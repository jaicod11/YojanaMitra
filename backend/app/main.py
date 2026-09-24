"""HTTP API over the frozen pipeline: understand -> retrieval -> match -> explain.

    uvicorn app.main:app --app-dir backend --port 8000

POST /match takes {"query", "language", "clarification"} and returns the shape
the frontend expects: profile_confidence, clarifying_question, notice and
results. A clarification is appended to the query and the whole pipeline runs
again on the combined text.

GET /health returns {"status": "ok"} and each LLM provider's key state
(keys available, cooling down, retired; last failure) without calling any
model.

Logging: the provider pool's lines (every failed attempt, every answer, key
cooldowns) and failures this module swallows go to the "yojanamitra" loggers,
written in uvicorn's format once the server starts (see lifespan).

Everything this module adds sits outside the frozen components
(understand.py, retrieval.py, matcher.py, generate.py), which it only calls:

- results carry the scheme record's documents (documents_text split into
  items), apply_url (its myScheme page), last_updated and source_url, and are
  sorted eligible, then needs_checking, then not_eligible, each group by how
  certain the verdict is (see sort_key)
- an eligible scheme that still has unverified conditions says so in its
  reason and lists them in "caveats"; when one of them states a stricter
  bound than the matcher checked and the person's answers do not settle it,
  the status drops to needs_checking (see app/postprocess.py)
- notice: when the person names a scheme ("... Yojana", "... Card") that no
  scheme name in the corpus matches, the answer says so instead of silently
  showing other schemes
- matched_clause: the clause the explanation cites, or the chunk retrieval
  matched on
- results beyond the five explain() covers get generate.py's code-written
  template, so every result has a reason (app/postprocess.collect)
- caveats_verbatim / unverified_verbatim flag, entry by entry, whether a
  condition appears verbatim in the scheme's eligibility_text (extraction
  asks for verbatim residuals but does not enforce it). The matcher's fixed
  "requires a dependent" line is not scheme text: it is left out of
  unverified_conditions and reported as requires_dependent_note instead

CORS: the browser calls POST /match directly, so the frontend's origin must be
listed in ALLOWED_ORIGINS (see .env.example).

Degraded mode: if no LLM provider answers, understand() and explain() fail
over to an empty profile and the code-written templates, so /match still
returns the matched schemes with reasons.
"""
import inspect
import json
import logging
import os
import re
import threading
from contextlib import asynccontextmanager
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from uvicorn.logging import DefaultFormatter

from app import generate
from app import matcher as matcher_module
from app.generate import explain
from app.postprocess import collect, finalize, is_grounded
from app.matcher import match, select_clarifying_field
from app.retrieval import default_retriever
from app.understand import PROFILE_KEYS, UnderstandError, phrase_question, provider_status, understand

ROOT = Path(__file__).resolve().parents[2]
SCHEMES_DIR = ROOT / "data" / "interim" / "schemes"
TOP_K = 10
MAX_QUERY_CHARS = 2000
LANGUAGES = ("en", "hi", "te", "ta", "bn")
log = logging.getLogger("yojanamitra.api")

# ALLOWED_ORIGINS may be set in the environment or in the repo-root .env; a
# variable already set in the environment wins.
load_dotenv(ROOT / ".env")
# CORS: exact origins only, comma-separated in ALLOWED_ORIGINS. The default is
# the Lovable dev server on this machine. At deploy time, add the deployed
# frontend's origin, e.g.
#   ALLOWED_ORIGINS=http://localhost:8080,https://<deployed-frontend-domain>   # <-- fill in at deploy time
DEFAULT_ORIGINS = "http://localhost:8080"
ALLOWED_ORIGINS = [o.strip().rstrip("/") for o in os.getenv("ALLOWED_ORIGINS", DEFAULT_ORIGINS).split(",")
                   if o.strip()]
if any("*" in o for o in ALLOWED_ORIGINS):
    raise ValueError("ALLOWED_ORIGINS must list exact origins; wildcards are not allowed")

# matcher.py adds this fixed sentence to unverified_conditions when a scheme
# requires a dependent. It is not scheme text, so it is reported as a flag
# instead. Fail at startup if matcher.py's wording ever drifts from this copy.
DEPENDENT_NOTE = "the scheme requires a dependent (see the scheme text)"
assert DEPENDENT_NOTE in inspect.getsource(matcher_module.evaluate_scheme), "matcher.py's dependent note changed"

# bge-m3 and the FAISS index are shared mutable state; retrieval and matching
# (which embeds occupations and other_facts) run one at a time.
_EMBED_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Scheme records
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def scheme_record(slug):
    path = SCHEMES_DIR / f"{slug}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


# "Aadhaar CardLandholding papers" (items run together), "Claim Form1. Death
# Certificate2. Aadhar Card" (numbered) and "Aadhar card.Address proof." (one
# sentence each) all occur in the corpus.
_DOC_NUMBER = re.compile(r"(?:(?<=[a-z.)])|(?<=^))\s*\d{1,2}[.)]\s*(?=[A-Z])")
_DOC_SENTENCE = re.compile(r"(?<!\bi\.e)(?<!\be\.g)(?<!\betc)(?<!\bviz)(?<!\bNo)(?<!\bMr)(?<!\bMrs)(?<!\bDr)"
                           r"(?<!\bSmt)(?<!\bRs)\.\s+(?=[A-Z0-9])")
# A period between two items ("...NDMC.Group photo"), but not inside initials
# such as "M.Sc"; or no period at all ("Aadhaar CardLandholding papers").
_DOC_RUN_ON = re.compile(r"(?<!\b[A-Z])(?<=[A-Za-z0-9)])\.(?=[A-Z][a-z]|[A-Z]{2,}[\s/])"
                         r"|(?<=[a-z)])(?=[A-Z][a-z]|[A-Z]{2,}[\s/])")
_DOC_BULLET = re.compile(r"[\r\n•▪●]+")
_DOC_HEADER = re.compile(r"^(?:indicative\s+|required\s+|necessary\s+)?documents?(?:\s+required|\s+needed|\s+list)?$"
                         r"|^list of documents$|documents required$", re.I)
# "TS Rythu Bheema Pathakam Documents Required Claim Form" -> "Claim Form"
_DOC_LEAD_IN = re.compile(r"^.{0,60}?documents?\s+(?:required|needed|list)\s*(?=[A-Z])", re.I)


def split_documents(text):
    """documents_text as a list of items."""
    if not text or not text.strip():
        return []
    parts = [text]
    for pattern in (_DOC_BULLET, _DOC_NUMBER, _DOC_SENTENCE, _DOC_RUN_ON):
        parts = [piece for part in parts for piece in pattern.split(part)]
    items = []
    for i, p in enumerate(parts):
        p = re.sub(r"\s+", " ", p or "").strip(" .;:,-–—")
        if i == 0:
            p = _DOC_LEAD_IN.sub("", p).strip(" .;:,-–—")
        if len(p) < 3 or _DOC_HEADER.search(p) or p.lower() in {"documents", "document"}:
            continue
        if p not in items:
            items.append(p)
    return items


# ---------------------------------------------------------------------------
# "We couldn't find a scheme called ..."
# ---------------------------------------------------------------------------

_SCHEME_WORD = r"Yojanas?|Yojnas?|Yojane|Schemes?|Cards?|Nidhi|Bima|Bheema|Beema"
_NAMED_SCHEME = re.compile(rf"((?:[A-Z][\w'’\-]*\s+){{1,6}}(?:{_SCHEME_WORD}))\b")
# Words too common to identify a scheme by.
_WEAK_TOKENS = {"scheme", "schemes", "yojana", "yojna", "yojanas", "yojane", "card", "cards", "nidhi", "bima",
                "bheema", "beema", "the", "of", "for", "and", "a", "an", "to", "in", "state", "government",
                "govt", "national", "pradhan", "mantri", "mukhya", "chief", "minister", "new", "india", "indian"}


def _name_tokens(name):
    return [t for t in re.findall(r"[\w’']+", (name or "").lower()) if t not in _WEAK_TOKENS and len(t) > 2]


@lru_cache(maxsize=1)
def _corpus_name_tokens():
    return [set(_name_tokens(scheme_record(p.stem).get("scheme_name"))) for p in SCHEMES_DIR.glob("*.json")]


def _matches_corpus(candidate):
    """True when some scheme name covers every distinctive word of candidate."""
    tokens = _name_tokens(candidate)
    if not tokens:
        return True                       # nothing distinctive to look up
    for name in _corpus_name_tokens():
        if all(any(t == n or SequenceMatcher(None, t, n).ratio() >= 0.82 for n in name) for t in tokens):
            return True
    return False


def unknown_scheme_name(*texts):
    """A scheme the person named that no scheme name in the corpus matches."""
    seen = []
    for text in texts:
        for m in _NAMED_SCHEME.finditer(text or ""):
            name = re.sub(r"\s+", " ", m.group(1)).strip()
            if name.lower() not in [s.lower() for s in seen]:
                seen.append(name)
    return next((name for name in seen if not _matches_corpus(name)), None)


# ---------------------------------------------------------------------------
# Request and response
# ---------------------------------------------------------------------------

class MatchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)
    language: str = "en"
    clarification: str | None = Field(default=None, max_length=MAX_QUERY_CHARS)


class ProfileField(BaseModel):
    field: str
    value: Any
    confidence: str | None


class Result(BaseModel):
    slug: str
    scheme_name: str | None
    level: str | None
    status: str
    reason: str
    matched_clause: str | None
    unverified_conditions: list[str]
    unverified_verbatim: list[bool]
    caveats: list[str]
    caveats_verbatim: list[bool]
    requires_dependent_note: bool
    documents: list[str]
    apply_url: str | None
    last_updated: str | None
    source_url: str | None


class MatchResponse(BaseModel):
    profile_confidence: list[ProfileField]
    clarifying_question: str | None
    notice: str | None
    results: list[Result]


EMPTY_UNDERSTANDING = {"search_query_en": None, "profile": {**dict.fromkeys(PROFILE_KEYS), "other_facts": []},
                       "confidence": {}, "evidence": {}, "clarifying_question": None, "status": "unavailable"}


def configure_server_logging():
    """Send the app's log lines (this module's and the provider pool's) to the
    server log in uvicorn's format. The pool's default handler is for the
    batch scripts' progress bars (scripts/label_categories.py), so it is
    removed here. Done at startup, not import, so scripts that import this
    module keep their own output."""
    handler = logging.StreamHandler()
    handler.setFormatter(DefaultFormatter("%(levelprefix)s %(message)s"))
    app_log = logging.getLogger("yojanamitra")
    app_log.handlers[:] = [handler]
    app_log.setLevel(logging.INFO)
    app_log.propagate = False
    providers_log = logging.getLogger("yojanamitra.providers")
    providers_log.handlers.clear()
    providers_log.propagate = True


@asynccontextmanager
async def lifespan(app):
    """Load the index, bge-m3 and the chunk texts once, before the first request."""
    configure_server_logging()
    default_retriever().search("government scheme for farmers", k=1, mode="hybrid")
    generate.scheme_chunks("pm-kisan")
    yield


app = FastAPI(title="YojanaMitra API", version="1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_methods=["POST", "OPTIONS"],
                   allow_headers=["Content-Type"], allow_credentials=False)


@app.get("/health")
def health():
    """Liveness, plus each provider's key state. Reads counters only; no model
    or provider is called."""
    return {"status": "ok", "providers": provider_status()}


@app.post("/match", response_model=MatchResponse)
def match_schemes(request: MatchRequest):
    language = request.language if request.language in LANGUAGES else "en"
    text = request.query.strip()
    if request.clarification and request.clarification.strip():
        text = f"{text} {request.clarification.strip()}"

    try:
        u = understand(text, language)
    except UnderstandError as e:
        log.warning("understand() failed, continuing without a profile: %s", e)
        u = EMPTY_UNDERSTANDING                      # no provider: keep going without a profile
    profile = u["profile"]
    facts = profile.get("other_facts") or []
    query_en = u["search_query_en"] or text

    with _EMBED_LOCK:
        hits = default_retriever().search(query_en, k=TOP_K, mode="hybrid")
        matched = match(profile, facts, hits, u["confidence"], u["evidence"])
        selection = select_clarifying_field(matched, top=TOP_K)
    explained = explain(profile, facts, matched, language)

    # -- clarifying question: understand()'s fallback for a nearly empty
    # profile, otherwise the field the matcher would ask about. Not asked
    # again once the person has answered one.
    question = None
    if not request.clarification:
        question = u.get("clarifying_question")
        if not question and selection and selection.get("field"):
            try:
                question = phrase_question(selection["field"], language, refine=selection["kind"] == "refine")
            except (UnderstandError, ValueError) as e:
                log.warning("phrase_question() failed, no clarifying question: %s", e)
                question = None

    evidence = {h["slug"]: h["evidence"]["text"] for h in hits}
    items = collect(matched, explained, language, lambda slug: scheme_record(slug).get("eligibility_text"))
    results = []
    for entry in finalize(items, profile, facts, confidence=u["confidence"], language=language):
        m, record = entry["match"], scheme_record(entry["match"]["slug"])
        scheme_text = [record.get("eligibility_text") or ""]
        unverified = [x for x in m["unverified_conditions"] if x != DEPENDENT_NOTE]
        results.append(Result(
            slug=m["slug"], scheme_name=m["scheme_name"] or record.get("scheme_name"), level=m["level"],
            status=entry["status"], reason=entry["reason"],
            matched_clause=(entry["citations"][0]["quote"] if entry["citations"] else evidence.get(m["slug"])),
            unverified_conditions=unverified,
            unverified_verbatim=[is_grounded(x, scheme_text) for x in unverified],
            caveats=entry["caveats"],
            caveats_verbatim=[is_grounded(x, scheme_text) for x in entry["caveats"]],
            requires_dependent_note=DEPENDENT_NOTE in m["unverified_conditions"],
            documents=split_documents(record.get("documents_text")),
            apply_url=record.get("source_url"), last_updated=record.get("last_updated"),
            source_url=record.get("source_url"),
        ))

    return MatchResponse(
        profile_confidence=[ProfileField(field=k, value=profile[k], confidence=u["confidence"].get(k))
                            for k in PROFILE_KEYS if profile.get(k) not in (None, [])],
        clarifying_question=question,
        notice=(f"We couldn't find a scheme called {name}. Here are schemes that may fit your situation."
                if (name := unknown_scheme_name(text, u["search_query_en"])) else None),
        results=results,
    )

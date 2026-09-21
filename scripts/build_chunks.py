"""Chunk the scheme corpus for retrieval.

Reads data/interim/schemes/*.json and writes data/processed/chunks.jsonl:

- one "overview" chunk per scheme: scheme_name + description + benefits_text
- eligibility_text split into clause-level chunks at sentence boundaries.
  Fragments under MIN_CHARS are merged into a neighbour, and clauses over
  MAX_CHARS are split at the last ";", "," or space before the cap. Clauses
  after an "Exclusions" header (the parser glues myScheme's Exclusions
  subsection into eligibility_text) get section "exclusions", so retrieval
  can tell a disqualifying clause from a qualifying one.

Every chunk carries chunk_id, slug, scheme_name, section, text, raw_text,
state, level and category (from data/interim/labels/predictions.json).

`text` is what gets indexed (embedded and BM25). For clause chunks it is
prefixed with the scheme context, "<scheme_name> — eligibility: <clause>" (or
"— exclusions:"), so identical boilerplate clauses in different schemes are
distinct and each clause says which scheme it belongs to. `raw_text` is the
bare clause, for display and citation. Overview chunks already start with the
scheme name, so their text is unprefixed and raw_text equals text. The
MIN_CHARS/MAX_CHARS limits apply to the raw clause.
"""
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCHEMES_DIR = ROOT / "data" / "interim" / "schemes"
PREDICTIONS_PATH = ROOT / "data" / "interim" / "labels" / "predictions.json"
OUT_PATH = ROOT / "data" / "processed" / "chunks.jsonl"

MIN_CHARS = 40
MAX_CHARS = 400

# Tokens that end in "." without ending a sentence ("Rs. 500", "Govt. of").
# Single letters are handled separately: initials, e.g./i.e., B.Tech, M.Sc.
ABBREVIATIONS = {
    "rs", "no", "nos", "dr", "mr", "mrs", "ms", "smt", "shri", "sri", "st",
    "sr", "jr", "vs", "viz", "approx", "govt", "ltd", "pvt", "co", "dept",
    "deptt", "ph", "sq", "hon", "prof", "ref", "yrs", "yr",
}
EMPTY_CLAUSES = {"na", "n a", "nil", "none", "not applicable"}

LIST_MARKER = r"(?:\d{1,2}|[a-hA-H]|[ivx]{1,4})[.)]\s?(?=[A-Z])|\((?:\d{1,2}|[a-h]|[ivx]{1,4})\)\s?(?=[A-Z])"
# A list marker starting a clause: after start of text, ".", ":", ";" or a bullet.
MARKER_BOUNDARY = re.compile(rf"(?:(?<=^)|(?<=[.:;•]))\s*(?={LIST_MARKER})")
# The parser sometimes drops the separator between list items or after a
# heading ("discontinuedIn case of", "EligibilityThe applicant"). Split such
# joins only before common clause openers, so fused names are left alone.
GLUED_START = re.compile(
    r"(?<=[a-z]{2})(?=(?:The|If|All|Any|Only|No|For|In|Once|Those|Where|When|Women|Children|Families|"
    r"Applicants?|Candidates?|Students?|Persons?|Beneficiar(?:y|ies)|Farmers?|Individuals?|"
    r"Minimum|Maximum|Preference|Income|Eligibility|Note)\b)")
CLAUSE_OPENER = re.compile(
    r"(?:The|A|An|If|All|Any|Only|No|For|In|Once|Those|Where|When|They|This|These|Such|Women|Children|"
    r"Applicants?|Candidates?|Students?|Persons?|Beneficiar(?:y|ies)|Farmers?|Individuals?|Eligibility)\b")
EXCLUSIONS_HEADER = re.compile(r"Exclusions?\s*:?\s*(?=[A-Z])|Exclusions\s*:?\s*$")
SENTENCE_END = re.compile(r"[.!?]['\")\]]?")


def normalize(text):
    return re.sub(r"\s+", " ", text or "").strip()   # \s includes non-breaking spaces


def is_sentence_end(text, i, j):
    """Whether the terminator text[i:j] ends a sentence."""
    rest = text[j:].lstrip()
    if not rest:
        return True
    nxt = rest[0]
    if not (nxt.isupper() or nxt in "•(" or re.match(LIST_MARKER, rest)):
        return False
    if text[i] != ".":
        return True
    token = re.search(r"([A-Za-z0-9/]+)$", text[:i])
    if not token:
        return True
    word = token.group(1)
    if len(word) == 1 and word.isalpha():          # initials, B.Tech, e.g.
        return False
    if word.lower() in ABBREVIATIONS:
        return False
    # Last part of a dotted abbreviation (B.Sc., M.Tech., M.F.Sc.): a sentence
    # end only if a clause opener follows ("M.Sc. Agriculture" stays whole).
    if re.search(r"(?:^|[^A-Za-z])[A-Za-z]\.$", text[:token.start()]) and not CLAUSE_OPENER.match(rest):
        return False
    # "2." opening a numbered list is a marker, not a sentence end.
    if word.isdigit() and len(word) <= 2 and re.search(r"(?:^|[.:;•]\s?)$", text[:token.start()]):
        return False
    return True


def split_sentences(text):
    """Split one block of text into sentences and list items."""
    pieces = []
    for block in re.split(r"\s*•\s*", text):
        # A marker right after a one-letter abbreviation is not a list item:
        # "e.g. PMEGP" must not become "e." + "g. PMEGP".
        starts = {0} | {m.end() for m in MARKER_BOUNDARY.finditer(block)
                        if m.end() > 0 and not re.search(r"\b[A-Za-z]\.$", block[:m.start()])}
        starts |= {m.start() for m in GLUED_START.finditer(block)}
        for m in SENTENCE_END.finditer(block):
            if is_sentence_end(block, m.start(), m.end()):
                starts.add(m.end())
        cuts = sorted(starts) + [len(block)]
        pieces += [block[a:b].strip() for a, b in zip(cuts, cuts[1:])]
    return [p for p in pieces if p]


def cap_length(piece):
    """Split a clause longer than MAX_CHARS at the last ; , or space before the cap."""
    out = []
    while len(piece) > MAX_CHARS:
        window = piece[:MAX_CHARS]
        cut = max(window.rfind("; "), window.rfind(", "))
        if cut < MAX_CHARS // 2:
            cut = window.rfind(" ")
        if cut <= 0:
            cut = MAX_CHARS - 1
        out.append(piece[:cut + 1].strip())
        piece = piece[cut + 1:].strip()
    return out + [piece] if piece else out


def merge_short(pieces):
    """Merge fragments under MIN_CHARS into the next piece (the previous one at the end)."""
    merged, carry = [], ""
    for p in pieces:
        p = f"{carry} {p}".strip() if carry else p
        carry = ""
        if len(p) < MIN_CHARS:
            carry = p
        else:
            merged.append(p)
    if carry:
        if merged:
            merged[-1] = f"{merged[-1]} {carry}"
        else:
            merged.append(carry)
    return merged


def eligibility_clauses(text):
    """Yield (section, clause) pairs for one scheme's eligibility_text."""
    text = normalize(text)
    if not text:
        return []
    header = EXCLUSIONS_HEADER.search(text)
    parts = [("eligibility", text[:header.start()] if header else text)]
    if header:
        parts.append(("exclusions", text[header.end():]))
    out = []
    for section, body in parts:
        pieces = [c for p in split_sentences(body) for c in cap_length(p)]
        pieces = [p for p in pieces if re.sub(r"[^a-z]+", " ", p.lower()).strip() not in EMPTY_CLAUSES]
        out += [(section, p) for p in merge_short(pieces)]
    return out


def build(schemes, categories):
    chunks = []
    for d in schemes:
        slug = d["slug"]
        base = {"slug": slug, "scheme_name": normalize(d["scheme_name"]),
                "state": d.get("state"), "level": d.get("level"), "category": categories[slug]}
        overview = " ".join(x for x in (normalize(d["scheme_name"]) + ".",
                                        normalize(d.get("description")),
                                        normalize(d.get("benefits_text"))) if x)
        chunks.append({"chunk_id": f"{slug}::overview", "section": "overview",
                       "text": overview, "raw_text": overview, **base})
        for i, (section, clause) in enumerate(eligibility_clauses(d.get("eligibility_text"))):
            chunks.append({"chunk_id": f"{slug}::{section}::{i:03d}", "section": section,
                           "text": f"{base['scheme_name']} — {section}: {clause}", "raw_text": clause, **base})
    fields = ("chunk_id", "slug", "scheme_name", "section", "text", "raw_text", "state", "level", "category")
    return [{k: c[k] for k in fields} for c in chunks]


def quantiles(xs):
    xs = sorted(xs)
    pick = lambda p: xs[round(p * (len(xs) - 1))]
    return (f"n={len(xs):5d}  min={xs[0]:4d}  p10={pick(.1):4d}  median={pick(.5):4d}  "
            f"p90={pick(.9):5d}  p99={pick(.99):5d}  max={xs[-1]:5d}  mean={statistics.mean(xs):.0f}")


def main():
    schemes = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(SCHEMES_DIR.glob("*.json"))]
    predictions = json.loads(PREDICTIONS_PATH.read_text(encoding="utf-8"))
    missing = [d["slug"] for d in schemes if d["slug"] not in predictions]
    if missing:
        sys.exit(f"error: {len(missing)} schemes have no category in {PREDICTIONS_PATH}: {missing[:5]}")
    chunks = build(schemes, {s: v["category"] for s, v in predictions.items()})

    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids)), "duplicate chunk_id"
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in chunks), encoding="utf-8")

    print(f"wrote {len(chunks)} chunks for {len(schemes)} schemes -> {OUT_PATH.relative_to(ROOT)}")
    print("\nchunks by section:", dict(Counter(c["section"] for c in chunks)))
    print("\nlength in characters of indexed text (clauses include the scheme prefix)")
    for section in ("overview", "eligibility", "exclusions"):
        print(f"  {section:12s} {quantiles([len(c['text']) for c in chunks if c['section'] == section])}")
    print("raw clause length (raw_text)")
    for section in ("eligibility", "exclusions"):
        print(f"  {section:12s} {quantiles([len(c['raw_text']) for c in chunks if c['section'] == section])}")
    clause_lens = [len(c["raw_text"]) for c in chunks if c["section"] != "overview"]
    print(f"  clauses over {MAX_CHARS + MIN_CHARS} chars (a short fragment merged into a capped clause): "
          f"{sum(n > MAX_CHARS + MIN_CHARS for n in clause_lens)}; under {MIN_CHARS}: {sum(n < MIN_CHARS for n in clause_lens)}")
    per = Counter(c["slug"] for c in chunks)
    print("\nchunks per scheme (overview included):", quantiles(list(per.values())))
    print("  schemes with no eligibility/exclusion chunks:", sum(n == 1 for n in per.values()))
    print("  schemes with an exclusions section:", len({c["slug"] for c in chunks if c["section"] == "exclusions"}))
    for field in ("raw_text", "text"):
        dup = Counter(c[field] for c in chunks)
        shared = {t: n for t, n in dup.items() if n > 1}
        print(f"\nduplicate {field} values: {len(shared)} distinct texts repeated, "
              f"{sum(shared.values())} chunks involved, {sum(shared.values()) - len(shared)} redundant copies")
        for t, n in [(t, n) for t, n in dup.most_common(5) if n > 1]:
            print(f"  {n:4d} x {t[:100]}")


if __name__ == "__main__":
    main()

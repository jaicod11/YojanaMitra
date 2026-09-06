#!/usr/bin/env python3
"""Convert scraped myscheme.gov.in PDFs into structured per-scheme JSON records.

Raw section splitting only -- no constraint extraction, no LLM calls, no
embedding, no normalization of eligibility semantics.
"""
import argparse
import hashlib
import json
import re
import statistics
import sys
from pathlib import Path

import ftfy
import pymupdf
from tqdm import tqdm

INPUT_DIR = Path("data/external/gov_myscheme/text_data")
OUTPUT_DIR = Path("data/interim/schemes")
REPORT_PATH = Path("data/interim/parse_report.json")

# ---------------------------------------------------------------------------
# Heading model
# ---------------------------------------------------------------------------

HEADINGS = [
    "Details",
    "Benefits",
    "Eligibility",
    "Application Process",
    "Documents Required",
    "Frequently Asked Questions",
    "Sources And References",
]
CANON = {h: i for i, h in enumerate(HEADINGS)}
FIELD_FOR_HEADING = {
    "Details": "description",
    "Benefits": "benefits_text",
    "Eligibility": "eligibility_text",
    "Application Process": "application_process_text",
    "Documents Required": "documents_text",
    "Frequently Asked Questions": "faq_text",
}
ALL_TEXT_FIELDS = list(FIELD_FOR_HEADING.values())


def _flex(s):
    """Escape a phrase but tolerate 0-or-1 whitespace wherever the phrase has a space.

    The PDFs sometimes wrap a literal space at a concatenation boundary
    ("sign out?CancelSign Out" vs "sign out? CancelSign Out") -- pymupdf's
    line-wrap newlines get collapsed to spaces upstream, so allowing an
    optional space here makes matching robust to both variants.
    """
    return re.escape(s).replace(r"\ ", r"\s?").replace(r"\-", r"\s?-\s?")


HEAD_RE = re.compile("|".join(_flex(h) for h in sorted(HEADINGS, key=len, reverse=True)))

# Fixed, position-independent chrome observed in every sampled file.
SIGN_OUT_MODAL_RE = re.compile(r"Are\s?you\s?sure\s?you\s?want\s?to\s?sign\s?out\?\s?Cancel\s?Sign\s?Out")
LANG_SWITCH_RE = re.compile(r"Eng\s?English/\S+\s?Sign\s?InBack")
ERROR_TOAST_RE = re.compile(r"Something\s?went\s?wrong\.\s?Please\s?try\s?again\s?later\.\s?Ok")
SIGNIN_PROMPT_RE = re.compile(r"You\s?need\s?to\s?sign\s?in\s?before\s?applying\s?for\s?schemes\s?Cancel\s?Sign\s?In")
ALREADY_APPLIED_RE = re.compile(
    r"It\s?seems\s?you\s?have\s?already\s?initiated\s?your\s?application\s?earlier\.\s?"
    r"To\s?know\s?more\s?please\s?visit\s?Cancel\s?Apply\s?Now\s?Check\s?Eligibility"
)
FEEDBACK_WORD_RE = re.compile(r"^\s?Feedback")
LAST_UPDATED_RE = re.compile(r"Last\s?Updated\s?On\s*:?\s*(\d{2})/(\d{2})/(\d{4})")
WAS_HELPFUL_MARKER = "Was this helpful?"
SIGN_OUT_MARKER = "Are you sure you want to sign out?"
LANG_MARKER = "EngEnglish"

DUPLICATE_DOWNLOAD_RE = re.compile(r"^(.*)\(\d+\)$")


def list_input_files():
    """Sorted input PDFs, excluding accidental re-download copies.

    ~90 files in this corpus are byte-identical re-downloads saved by the
    browser as "<slug>(1).pdf", "<slug>(2).pdf" alongside "<slug>.pdf".
    Parsing them would mint bogus extra slugs / source_urls for schemes
    that already have a proper record, so they're skipped whenever the
    un-suffixed base file is also present.
    """
    all_files = sorted(INPUT_DIR.glob("*.pdf"))
    stems = {p.stem for p in all_files}
    kept, skipped = [], []
    for p in all_files:
        m = DUPLICATE_DOWNLOAD_RE.match(p.stem)
        if m and m.group(1) in stems:
            skipped.append(p.stem)
        else:
            kept.append(p)
    return kept, skipped

# ---------------------------------------------------------------------------
# Tag blob parsing
# ---------------------------------------------------------------------------

STATES_UTS = [
    "Andaman and Nicobar Islands", "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar",
    "Chandigarh", "Chhattisgarh", "Dadra and Nagar Haveli and Daman and Diu",
    "National Capital Territory of Delhi", "NCT of Delhi", "Delhi", "Goa", "Gujarat",
    "Haryana", "Himachal Pradesh", "Jammu and Kashmir", "Jharkhand", "Karnataka", "Kerala",
    "Ladakh", "Lakshadweep", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya",
    "Mizoram", "Nagaland", "Odisha", "Puducherry", "Punjab", "Rajasthan", "Sikkim",
    "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal",
]
STATES_UTS_SET = set(STATES_UTS)

# Split "AmbedkarEntrepreneurLoanMSMEScheduled CasteScheduled TribeSubsidy" into
# ["Ambedkar","Entrepreneur","Loan","MSME","Scheduled Caste","Scheduled Tribe","Subsidy"].
# A space is only ever *within* a tag/name, never *between* them, so splitting on
# camel-case-like boundaries (lower/digit -> upper, or ACRONYM -> Titlecase) recovers
# tag boundaries while leaving genuine multi-word tags intact.
TOKEN_SPLIT_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def tokenize_blob(blob):
    return [p.strip() for p in TOKEN_SPLIT_RE.split(blob) if p.strip()]


def looks_like_tag(token):
    """Real tag chips are rendered in full Title Case (every word capitalized);
    stray UI sentences leaking into the blob ("Sign in to apply") are not."""
    words = [w for w in re.split(r"[\s-]+", token) if w]
    if not words:
        return False
    return all(not w[0].isalpha() or w[0].isupper() for w in words)


def parse_tag_blob(preamble, scheme_name):
    flags = []
    blob = preamble.strip()
    if not blob:
        return None, None, [], ["tag_blob_empty"]

    # The page's own <title> text (a verbatim copy of scheme_name) always
    # precedes the tag-blob proper, since we only strip the modals/nav
    # *between* them, not the title itself. Titles can contain punctuation
    # ("M.V.Sc.", "Ph.D.") that defeats the camel-case tokenizer below, so
    # locate both copies by direct (whitespace-tolerant) search instead.
    # The tag blob's own copy occasionally differs by a stray space next to
    # a hyphen ("Scheme-B" vs "Scheme- B"), hence \s? rather than a literal
    # match.
    name_re = re.compile(_flex(scheme_name)) if scheme_name else None
    m0 = name_re.match(blob) if name_re else None
    if m0:
        blob = blob[m0.end():]

    m1 = name_re.search(blob) if name_re else None
    if not m1:
        flags.append("tag_blob_name_not_found")
        prefix_text, suffix_text = "", blob
    else:
        prefix_text = blob[:m1.start()].strip()
        suffix_text = blob[m1.end():]

    state = None
    level = None
    if prefix_text in STATES_UTS_SET:
        state = prefix_text
        level = "state"
    elif prefix_text:
        level = "central"
    else:
        flags.append("level_uncertain")

    rest = tokenize_blob(suffix_text)
    if rest and re.fullmatch(r"[A-Z0-9]{2,12}", rest[0]):
        rest = rest[1:]  # drop the scheme acronym token, e.g. "AABCS"
    tags = [t for t in rest if looks_like_tag(t)]
    if len(tags) != len(rest):
        flags.append("tag_blob_noise_filtered")
    return state, level, tags, flags


# ---------------------------------------------------------------------------
# Page extraction / cleaning
# ---------------------------------------------------------------------------

def extract_deduped_pages(doc):
    """Hash each page's raw text and keep only the first occurrence of each."""
    seen = set()
    kept = []
    dropped = 0
    for page in doc:
        raw = page.get_text()
        h = hashlib.md5(raw.encode("utf-8", "ignore")).hexdigest()
        if h in seen:
            dropped += 1
            continue
        seen.add(h)
        kept.append(raw)
    return kept, dropped


def clean_join(pages):
    text = " ".join(pages)
    text = re.sub(r"\s+", " ", text).strip()
    text = ftfy.fix_text(text)
    text = text.replace("â€​", '"')  # residual glued mojibake, see notes
    text = text.replace("​", "")
    return text


def strip_front_boilerplate(text):
    flags = []
    text = SIGN_OUT_MODAL_RE.sub("", text)

    m = LANG_SWITCH_RE.search(text)
    if m:
        cursor = m.end()
        current_idx = -1
        while True:
            hm = HEAD_RE.match(text, cursor)
            if not hm:
                # tolerate a short extra tab word between headings, e.g. "Exclusions"
                hm2 = HEAD_RE.search(text, cursor)
                if hm2 and hm2.start() - cursor <= 20 and re.fullmatch(r"[A-Za-z ]*", text[cursor:hm2.start()]):
                    hm = hm2
                else:
                    break
            c = CANON[hm.group()]
            if c <= current_idx:
                break
            current_idx = c
            cursor = hm.end()
        fb = FEEDBACK_WORD_RE.match(text[cursor:cursor + 15])
        if fb:
            cursor += fb.end()
        text = text[:m.start()] + text[cursor:]
    else:
        flags.append("nav1_marker_not_found")

    text = ERROR_TOAST_RE.sub("", text)
    text = SIGNIN_PROMPT_RE.sub("", text)
    text = ALREADY_APPLIED_RE.sub("", text)
    return text, flags


def split_sections(content_text):
    """Monotonic forward-index heading walk.

    Only a match whose canonical heading index strictly exceeds the current
    section index counts as a new section boundary. This naturally ignores
    same-heading repeats (e.g. a tab-style "Application Process...Offline
    ...Application Process" sub-heading) and false-positive substrings of a
    heading word inside ordinary prose (e.g. "Check Eligibility",
    "1. Details of Aadhaar card"), since neither advances the index.
    """
    boundaries = []
    current_idx = -1
    started = False
    for m in HEAD_RE.finditer(content_text):
        c = CANON[m.group()]
        if not started:
            # The tag blob (state/name/tags) can itself contain a heading word
            # as a substring of a tag, e.g. "Death Benefits" or "Contact
            # Details". Ignore everything until the real body's first
            # "Details" heading so such noise can't be mistaken for a
            # section boundary before real content has even started.
            if c != 0:
                continue
            started = True
        if c > current_idx:
            boundaries.append((c, m.start(), m.end()))
            current_idx = c

    sections = {}
    for i, (c, start, end) in enumerate(boundaries):
        stop = boundaries[i + 1][1] if i + 1 < len(boundaries) else len(content_text)
        sections[c] = content_text[end:stop].strip()

    preamble_end = boundaries[0][1] if boundaries else 0
    preamble = content_text[:preamble_end]
    return sections, preamble, [CANON[h] for h in HEADINGS if CANON[h] not in sections]


# ---------------------------------------------------------------------------
# Per-file parse
# ---------------------------------------------------------------------------

def parse_pdf(path):
    slug = path.stem
    source_url = f"https://www.myscheme.gov.in/schemes/{slug}"
    flags = []

    doc = pymupdf.open(path)
    pages, dropped = extract_deduped_pages(doc)
    if dropped:
        flags.append("duplicate_pages_dropped")
    doc.close()

    text = clean_join(pages)

    lu = LAST_UPDATED_RE.search(text)
    if lu:
        dd, mm, yyyy = lu.groups()
        last_updated = f"{yyyy}-{mm}-{dd}"
    else:
        last_updated = None
        flags.append("no_last_updated")

    sign_out_idx = text.find(SIGN_OUT_MARKER)
    lang_idx = text.find(LANG_MARKER)
    title_end = sign_out_idx if sign_out_idx != -1 else lang_idx
    if title_end == -1:
        scheme_name = ""
        flags.append("title_extraction_failed")
    else:
        scheme_name = text[:title_end].strip()

    helpful_idx = text.find(WAS_HELPFUL_MARKER)
    if helpful_idx == -1:
        content_text = text
        flags.append("no_feedback_marker")
    else:
        content_text = text[:helpful_idx]

    content_text, front_flags = strip_front_boilerplate(content_text)
    flags.extend(front_flags)

    sections, preamble, _missing_canon = split_sections(content_text)
    state, level, tags, tag_flags = parse_tag_blob(preamble, scheme_name)
    flags.extend(tag_flags)

    record = {
        "slug": slug,
        "scheme_name": scheme_name,
        "state": state,
        "level": level,
        "tags": tags,
        "description": sections.get(CANON["Details"], ""),
        "benefits_text": sections.get(CANON["Benefits"], ""),
        "eligibility_text": sections.get(CANON["Eligibility"], ""),
        "application_process_text": sections.get(CANON["Application Process"], ""),
        "documents_text": sections.get(CANON["Documents Required"], ""),
        "faq_text": sections.get(CANON["Frequently Asked Questions"], ""),
        "last_updated": last_updated,
        "source_url": source_url,
        "parse_flags": flags,
    }

    for canon_idx, field in ((CANON[h], f) for h, f in FIELD_FOR_HEADING.items()):
        if not record[field]:
            record["parse_flags"].append(f"missing_{field.replace('_text','').replace('description','details')}")

    total_len = sum(len(record[f]) for f in ALL_TEXT_FIELDS)
    if total_len < 200:
        record["parse_flags"].append("suspiciously_short")

    return record


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def build_report(records, total_files, failed):
    report = {
        "total_files": total_files,
        "parsed": len(records),
        "failed": len(failed),
        "failed_slugs": failed,
        "section_presence_rate": {},
        "eligibility_length_distribution": {},
        "parse_flag_counts": {},
        "shortest_eligibility_slugs": [],
    }

    for field in ALL_TEXT_FIELDS:
        non_empty = sum(1 for r in records if r.get(field))
        report["section_presence_rate"][field] = round(non_empty / len(records), 4) if records else 0.0

    elig_lengths = [len(r["eligibility_text"]) for r in records if r["eligibility_text"]]
    if elig_lengths:
        elig_lengths_sorted = sorted(elig_lengths)
        n = len(elig_lengths_sorted)
        p90_idx = min(n - 1, int(round(0.9 * (n - 1))))
        report["eligibility_length_distribution"] = {
            "min": elig_lengths_sorted[0],
            "median": statistics.median(elig_lengths_sorted),
            "p90": elig_lengths_sorted[p90_idx],
            "max": elig_lengths_sorted[-1],
            "count": n,
        }

    flag_counts = {}
    for r in records:
        for f in r["parse_flags"]:
            flag_counts[f] = flag_counts.get(f, 0) + 1
    report["parse_flag_counts"] = dict(sorted(flag_counts.items(), key=lambda kv: -kv[1]))

    non_empty_elig = [(len(r["eligibility_text"]), r["slug"]) for r in records if r["eligibility_text"]]
    non_empty_elig.sort()
    report["shortest_eligibility_slugs"] = [
        {"slug": s, "length": l} for l, s in non_empty_elig[:10]
    ]

    return report


def print_report(report):
    print()
    print("=" * 70)
    print("PARSE REPORT")
    print("=" * 70)
    print(f"total files : {report['total_files']}")
    print(f"parsed      : {report['parsed']}")
    print(f"failed      : {report['failed']}")
    if "skipped_duplicate_downloads" in report:
        print(f"skipped     : {report['skipped_duplicate_downloads']} (duplicate re-downloads, e.g. slug(1).pdf)")
    if report["failed_slugs"]:
        print(f"  failed slugs: {report['failed_slugs']}")
    print()
    print("section presence rate:")
    for field, rate in report["section_presence_rate"].items():
        print(f"  {field:28s} {rate*100:6.2f}%")
    print()
    if report["eligibility_length_distribution"]:
        d = report["eligibility_length_distribution"]
        print("eligibility_text length distribution (chars, non-empty only):")
        print(f"  count={d['count']}  min={d['min']}  median={d['median']}  p90={d['p90']}  max={d['max']}")
    print()
    print("parse_flag counts:")
    for flag, count in report["parse_flag_counts"].items():
        print(f"  {flag:32s} {count}")
    print()
    print("10 shortest non-empty eligibility_text (likely bad parses):")
    for entry in report["shortest_eligibility_slugs"]:
        print(f"  {entry['length']:5d}  {entry['slug']}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None, help="only process the first N files")
    ap.add_argument("--force", action="store_true", help="reparse and overwrite existing outputs")
    ap.add_argument("--slug", type=str, default=None, help="process a single slug and print debug info")
    args = ap.parse_args()

    if args.slug:
        path = INPUT_DIR / f"{args.slug}.pdf"
        if not path.exists():
            print(f"no such file: {path}", file=sys.stderr)
            sys.exit(1)
        record = parse_pdf(path)
        print(json.dumps(record, indent=2, ensure_ascii=False))
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    files, skipped_duplicates = list_input_files()
    if args.limit:
        files = files[: args.limit]

    records = []
    failed = []
    for path in tqdm(files, desc="parsing"):
        out_path = OUTPUT_DIR / f"{path.stem}.json"
        if out_path.exists() and not args.force:
            try:
                records.append(json.loads(out_path.read_text(encoding="utf-8")))
                continue
            except (json.JSONDecodeError, OSError):
                pass  # fall through and reparse
        try:
            record = parse_pdf(path)
        except Exception as e:  # noqa: BLE001 - report and continue
            failed.append(path.stem)
            tqdm.write(f"FAILED {path.stem}: {e}")
            continue
        out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        records.append(record)

    report = build_report(records, len(files), failed)
    report["skipped_duplicate_downloads"] = len(skipped_duplicates)
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print_report(report)


if __name__ == "__main__":
    main()

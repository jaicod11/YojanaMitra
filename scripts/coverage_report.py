#!/usr/bin/env python3
"""How machine-checkable is the constraint corpus?

Answers one question: for how many schemes can deterministic code actually
decide anything from typed fields, and where does it have to fall back to
residual prose? Also surfaces which residual conditions recur often enough to
be worth promoting to typed fields.

Reads data/interim/constraints/<slug>.json (whatever has been extracted so
far) plus the category labels in data/interim/labels/predictions.json, and
writes data/interim/constraints/coverage_report.json.

Calls no APIs -- pure analysis over files on disk, safe to run mid-extraction.
"""
import argparse
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extract_constraints as ec  # noqa: E402  (schema + field semantics)
import label_categories as lc  # noqa: E402

ROOT = ec.ROOT
CONSTRAINTS_DIR = ROOT / "data/interim/constraints"
PREDICTIONS_PATH = ROOT / "data/interim/labels/predictions.json"
OUT_NAME = "coverage_report.json"

# Residual strings are long verbatim sentences, so exact matches are rare even
# when the same requirement recurs. Phrase frequency across schemes is what
# actually reveals a promotion candidate.
NGRAM_SIZES = (4, 5, 6)
MIN_SCHEMES_FOR_PHRASE = 5
# Scheme prose opens the same way everywhere ("the applicant should be ..."),
# so an n-gram is only interesting if it carries real content words. Requiring
# two keeps "availing any other scheme" and drops "the applicant should be".
BOILERPLATE_WORDS = {
    "the", "a", "an", "of", "to", "in", "for", "and", "or", "be", "is", "are", "was",
    "should", "must", "shall", "not", "any", "under", "this", "that", "with", "by",
    "as", "at", "on", "from", "his", "her", "their", "he", "she", "they", "it",
    "applicant", "applicants", "candidate", "candidates", "beneficiary", "beneficiaries",
    "person", "persons", "who", "whose", "has", "have", "had", "been", "being", "will",
    "would", "may", "can", "only", "also", "such", "same", "there", "which", "if",
    "case", "per", "no",
}
MIN_CONTENT_WORDS = 2


def is_informative(gram):
    content = [w for w in gram.split() if w not in BOILERPLATE_WORDS]
    return len(content) >= MIN_CONTENT_WORDS


def normalise(text):
    """Loose normalisation: lowercase, drop punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_constraints(constraints_dir):
    records = {}
    for path in sorted(Path(constraints_dir).glob("*.json")):
        if path.name in (ec.REPORT_FILENAME, OUT_NAME):
            continue
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if "constraints" in rec:
            records[rec["slug"]] = rec
    return records


def binding_count(constraints):
    """Typed fields that actually narrow who qualifies ('any' does not)."""
    return sum(1 for f in ec.TYPED_FIELDS if ec.field_is_constraining(constraints, f))


def bucket(n):
    return str(n) if n < 3 else "3+"


def summarise(counts):
    """counts: list of binding-field counts -> distribution + cumulative shares."""
    n = len(counts)
    if not n:
        return {}
    dist = collections.Counter(bucket(c) for c in counts)
    return {
        "schemes": n,
        "distribution": {k: dist.get(k, 0) for k in ("0", "1", "2", "3+")},
        "distribution_pct": {k: round(dist.get(k, 0) / n * 100, 1) for k in ("0", "1", "2", "3+")},
        "at_least_1_pct": round(sum(1 for c in counts if c >= 1) / n * 100, 1),
        "at_least_2_pct": round(sum(1 for c in counts if c >= 2) / n * 100, 1),
        "at_least_3_pct": round(sum(1 for c in counts if c >= 3) / n * 100, 1),
        "mean_binding_fields": round(sum(counts) / n, 2),
    }


def phrase_candidates(residual_sets):
    """Recurring phrases across schemes, by number of DISTINCT schemes.

    A promotion candidate is a requirement many schemes impose, so schemes-
    containing is the right denominator, not raw occurrences.
    """
    schemes_with_ngram = collections.defaultdict(set)
    for slug, residuals in residual_sets.items():
        seen = set()
        for text in residuals:
            words = text.split()
            for size in NGRAM_SIZES:
                for i in range(len(words) - size + 1):
                    seen.add(" ".join(words[i:i + size]))
        for gram in seen:
            schemes_with_ngram[gram].add(slug)

    scored = [(len(slugs), gram) for gram, slugs in schemes_with_ngram.items()
              if len(slugs) >= MIN_SCHEMES_FOR_PHRASE and is_informative(gram)]
    scored.sort(key=lambda kv: (-kv[0], -len(kv[1])))

    # One recurring idea generates many overlapping n-grams ("be a permanent
    # resident", "should be a permanent resident of", ...). Collapse them by
    # content-word overlap so the list shows distinct candidates, not variants
    # of one.
    kept = []
    for count, gram in scored:
        words = {w for w in gram.split() if w not in BOILERPLATE_WORDS}
        if any(words and kwords and
               len(words & kwords) / min(len(words), len(kwords)) >= 0.75
               for _, _, kwords in kept):
            continue
        kept.append((count, gram, words))
        if len(kept) >= 25:
            break
    kept = [(c, g) for c, g, _ in kept]
    return [{"phrase": g, "schemes": c,
             "pct_of_schemes": round(c / max(1, len(residual_sets)) * 100, 1)}
            for c, g in kept]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--constraints-dir", type=Path, default=CONSTRAINTS_DIR)
    ap.add_argument("--predictions", type=Path, default=PREDICTIONS_PATH)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--baseline", type=Path, default=None,
                    help="constraints dir snapshotted before a pass; adds a before/after "
                         "comparison, including schemes that moved from 0 binding fields to >=1")
    args = ap.parse_args()

    records = load_constraints(args.constraints_dir)
    if not records:
        print(f"no constraint records found in {args.constraints_dir}", file=sys.stderr)
        sys.exit(1)

    labels = {}
    if args.predictions.exists():
        labels = json.loads(args.predictions.read_text(encoding="utf-8"))
    schemes = lc.load_records()

    overall, by_category, by_level, by_field = [], collections.defaultdict(list), \
        collections.defaultdict(list), collections.Counter()
    residual_sets, residual_counts = {}, []
    exact_counter = collections.Counter()
    zero_binding, failed = [], []

    for slug, rec in records.items():
        c = rec["constraints"]
        n = binding_count(c)
        overall.append(n)
        if n == 0:
            zero_binding.append(slug)
        for f in ec.TYPED_FIELDS:
            if ec.field_is_constraining(c, f):
                by_field[f] += 1

        cat = labels.get(slug, {}).get("category")
        if cat:
            by_category[cat].append(n)
        level = (schemes.get(slug) or {}).get("level")
        if level:
            by_level[level].append(n)

        if rec.get("extraction_failed"):
            # All-null, so it already counts above as zero binding (manual
            # review). Its residuals are unknown rather than empty, so it stays
            # out of the residual stats.
            failed.append(slug)
            continue
        residuals = [normalise(r) for r in c["residual_conditions"]]
        residuals = [r for r in residuals if r]
        residual_sets[slug] = residuals
        residual_counts.append(len(residuals))
        exact_counter.update(set(residuals))  # set(): once per scheme

    n = len(records)
    n_residual = max(1, len(residual_sets))
    report = {
        "constraint_records_analysed": n,
        "note": ("'Binding' counts typed fields that actually narrow eligibility; "
                 "gender/residence set to 'any' assert no restriction and are excluded."),
        "overall": summarise(overall),
        "per_field_binding_counts": {f: by_field.get(f, 0) for f in ec.TYPED_FIELDS},
        "per_field_binding_pct": {f: round(by_field.get(f, 0) / n * 100, 1) for f in ec.TYPED_FIELDS},
        "by_category": {cat: summarise(v) for cat, v in
                        sorted(by_category.items(), key=lambda kv: -len(kv[1]))},
        "by_level": {lvl: summarise(v) for lvl, v in sorted(by_level.items())},
        "schemes_with_zero_binding_constraints": {
            "count": len(zero_binding),
            "pct": round(len(zero_binding) / n * 100, 1),
            "examples": sorted(zero_binding)[:25],
            "of_which_extraction_failed": len(failed),
        },
        "extraction_failed": {
            "count": len(failed),
            "slugs": sorted(failed),
            "note": ("No validated extraction after all attempts: every typed field is null, "
                     "so these count as zero binding (manual review) and are excluded "
                     "from residual_conditions statistics."),
        },
        "residual_conditions": {
            "schemes": len(residual_sets),
            "total": sum(residual_counts),
            "mean_per_scheme": round(sum(residual_counts) / n_residual, 2),
            "top_20_exact_normalised": [
                {"text": t, "schemes": c, "pct_of_schemes": round(c / n_residual * 100, 1)}
                for t, c in exact_counter.most_common(20)
            ],
            "exact_match_note": ("Residuals are long verbatim sentences, so exact matches "
                                  "undercount recurring requirements phrased differently. "
                                  "recurring_phrases below is the signal for promotion."),
            "recurring_phrases": phrase_candidates(residual_sets),
            "phrase_note": (f"n-grams of {NGRAM_SIZES} words appearing in >= "
                            f"{MIN_SCHEMES_FOR_PHRASE} distinct schemes, longest phrasing kept."),
        },
    }

    if args.baseline:
        base = load_constraints(args.baseline)
        common = [s for s in records if s in base]
        before = {s: binding_count(base[s]["constraints"]) for s in common}
        after = {s: binding_count(records[s]["constraints"]) for s in common}
        zero_to_some = sorted(s for s in common if before[s] == 0 and after[s] >= 1)
        report["vs_baseline"] = {
            "baseline_dir": str(args.baseline),
            "schemes_compared": len(common),
            "before": summarise(list(before.values())),
            "after": summarise(list(after.values())),
            "zero_to_at_least_1": len(zero_to_some),
            "zero_to_at_least_1_slugs": zero_to_some,
            "schemes_gaining_binding_fields": sum(1 for s in common if after[s] > before[s]),
        }

    out_path = args.out or (Path(args.constraints_dir) / OUT_NAME)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    o = report["overall"]
    print("=" * 74)
    print(f"CONSTRAINT COVERAGE  ({n} schemes analysed)")
    print("=" * 74)
    print("binding typed fields per scheme:")
    for k in ("0", "1", "2", "3+"):
        print(f"  {k:>2s} fields : {o['distribution'][k]:5d}  ({o['distribution_pct'][k]:5.1f}%)")
    print(f"  >=1 binding: {o['at_least_1_pct']}%   >=2: {o['at_least_2_pct']}%   "
          f">=3: {o['at_least_3_pct']}%   mean: {o['mean_binding_fields']}")
    if failed:
        print(f"  (0 fields includes {len(failed)} extraction_failed -> manual review)")
    print()
    print("binding rate per field:")
    for f, pct in sorted(report["per_field_binding_pct"].items(), key=lambda kv: -kv[1]):
        print(f"  {f:16s} {pct:5.1f}%  {'#' * int(round(pct / 2.5))}")
    print()
    print(f"{'category':42s} {'n':>5s} {'>=1':>6s} {'>=2':>6s} {'0 fields':>9s}")
    for cat, s in report["by_category"].items():
        print(f"  {cat:40s} {s['schemes']:5d} {s['at_least_1_pct']:5.1f}% "
              f"{s['at_least_2_pct']:5.1f}% {s['distribution_pct']['0']:8.1f}%")
    print()
    print(f"{'level':42s} {'n':>5s} {'>=1':>6s} {'>=2':>6s} {'0 fields':>9s}")
    for lvl, s in report["by_level"].items():
        print(f"  {lvl:40s} {s['schemes']:5d} {s['at_least_1_pct']:5.1f}% "
              f"{s['at_least_2_pct']:5.1f}% {s['distribution_pct']['0']:8.1f}%")
    print()
    print("top exact residual strings (normalised, counted once per scheme):")
    for e in report["residual_conditions"]["top_20_exact_normalised"][:20]:
        print(f"  {e['schemes']:4d} ({e['pct_of_schemes']:4.1f}%)  {e['text'][:88]}")
    print()
    print("recurring phrases -- candidates to promote from residual to typed:")
    for e in report["residual_conditions"]["recurring_phrases"]:
        print(f"  {e['schemes']:4d} ({e['pct_of_schemes']:4.1f}%)  {e['phrase']}")
    vb = report.get("vs_baseline")
    if vb:
        print()
        print(f"vs baseline {vb['baseline_dir']}  ({vb['schemes_compared']} schemes):")
        for label in ("before", "after"):
            s = vb[label]
            print(f"  {label:6s}  0 fields: {s['distribution']['0']:5d}   >=1: {s['at_least_1_pct']}%   "
                  f">=2: {s['at_least_2_pct']}%   >=3: {s['at_least_3_pct']}%   mean: {s['mean_binding_fields']}")
        print(f"  moved from 0 fields to >=1: {vb['zero_to_at_least_1']}   "
              f"gained any binding field: {vb['schemes_gaining_binding_fields']}")
    print("=" * 74)
    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()

"""Evaluate scheme retrieval against the gold set.

    python scripts/evaluate.py                        # dev split, all modes
    python scripts/evaluate.py --split test --final   # once, at the end

Reads data/gold/gold_set.jsonl, drops skip_scoring rows, keeps the requested
split, and runs each row's description through backend/app/retrieval.search
in every mode (bm25, dense, hybrid), taking the top 10 schemes.

- positive rows:  Recall@5 and Recall@10 (per row, the fraction of expected
                  schemes in the top k, averaged over rows) and MRR@10 (the
                  reciprocal rank of the first expected scheme in the top 10,
                  or 0 if none).
- exclusion rows: false-inclusion rate, the share of rows with any excluded
                  scheme in the top 10. As a diagnostic, also how many of
                  those inclusions were matched through an "exclusions" chunk,
                  i.e. the clause that rules the user out.
- clarify and no_match rows are not scored here: retrieval always returns
  schemes, and asking or declining is the job of a later stage. They are
  counted in the output.

Every metric is also broken down by language. Results, including each row's
top 10 with evidence, go to data/eval/<split>_<timestamp>.json.

Tuning may only look at dev; test is run once, at the end (data/gold/README.md).
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
from app.retrieval import CANDIDATES, MODES, RRF_K, default_retriever, file_sha256  # noqa: E402

GOLD_PATH = ROOT / "data" / "gold" / "gold_set.jsonl"
EVAL_DIR = ROOT / "data" / "eval"
TOP_K = 10
EVAL_MODES = ("bm25", "dense", "hybrid")
assert set(EVAL_MODES) == set(MODES)


def score_positive(expected, slugs):
    first = next((i for i, s in enumerate(slugs[:TOP_K], 1) if s in expected), None)
    return {
        "recall@5": len(expected & set(slugs[:5])) / len(expected),
        "recall@10": len(expected & set(slugs[:10])) / len(expected),
        "rr@10": 1 / first if first else 0.0,
        "first_hit_rank": first,
    }


def summarize(rows, mode):
    pos = [r["modes"][mode] for r in rows if r["test_type"] == "positive"]
    exc = [r["modes"][mode] for r in rows if r["test_type"] == "exclusion"]
    mean = lambda xs: round(sum(xs) / len(xs), 4) if xs else None
    included = [m for m in exc if m["excluded_in_top10"]]
    return {
        "positive": {"n": len(pos),
                     "recall@5": mean([m["recall@5"] for m in pos]),
                     "recall@10": mean([m["recall@10"] for m in pos]),
                     "mrr@10": mean([m["rr@10"] for m in pos])},
        "exclusion": {"n": len(exc),
                      "false_inclusion_rate": mean([1.0 if m["excluded_in_top10"] else 0.0 for m in exc]),
                      "inclusions_via_exclusions_chunk": sum(
                          any(hit["section"] == "exclusions" for hit in m["excluded_in_top10"]) for m in included)},
    }


def fmt(x):
    return "   —  " if x is None else f"{x:6.3f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--final", action="store_true", help="required to run the test split")
    args = ap.parse_args()
    if args.split == "test" and not args.final:
        ap.error("the test split is run once, at the end (see data/gold/README.md); pass --final to run it")

    gold = [json.loads(line) for line in open(GOLD_PATH, encoding="utf-8") if line.strip()]
    if any("split" not in r for r in gold):
        sys.exit(f"error: {GOLD_PATH} has rows without a split; run scripts/split_gold.py")
    rows = [r for r in gold if not r.get("skip_scoring") and r["split"] == args.split]
    counts = defaultdict(int)
    for r in rows:
        counts[r["test_type"]] += 1
    print(f"{args.split}: {len(rows)} rows {dict(counts)} "
          f"({sum(1 for r in gold if r.get('skip_scoring'))} skip_scoring rows excluded)")

    retriever = default_retriever()
    latency = defaultdict(list)
    results = []
    for r in rows:
        entry = {"id": r["id"], "language": r["language"], "test_type": r["test_type"],
                 "expected_schemes": r["expected_schemes"], "excluded_schemes": r.get("excluded_schemes", []),
                 "modes": {}}
        for mode in EVAL_MODES:
            t0 = time.time()
            hits = retriever.search(r["description"], k=TOP_K, mode=mode)
            latency[mode].append(time.time() - t0)
            slugs = [h["slug"] for h in hits]
            m = {"top10": [{"slug": h["slug"], "score": round(h["score"], 5), "section": h["evidence"]["section"],
                            "chunk_id": h["evidence"]["chunk_id"]} for h in hits]}
            if r["test_type"] == "positive":
                m.update(score_positive(set(r["expected_schemes"]), slugs))
            if r["test_type"] == "exclusion":
                excluded = set(r["excluded_schemes"])
                m["excluded_in_top10"] = [{"slug": h["slug"], "rank": i, "section": h["evidence"]["section"]}
                                          for i, h in enumerate(hits[:TOP_K], 1) if h["slug"] in excluded]
            entry["modes"][mode] = m
        results.append(entry)

    languages = sorted({r["language"] for r in results})
    metrics = {}
    for mode in EVAL_MODES:
        metrics[mode] = summarize(results, mode)
        metrics[mode]["by_language"] = {lang: summarize([r for r in results if r["language"] == lang], mode)
                                        for lang in languages}
        # The first dense query includes loading the model; report the median.
        metrics[mode]["median_latency_s"] = round(sorted(latency[mode])[len(latency[mode]) // 2], 3)

    print(f"\n{'mode':7s} | {'positive (n=' + str(counts['positive']) + ')':^31s} | "
          f"{'exclusion (n=' + str(counts['exclusion']) + ')':^30s}")
    print(f"{'':7s} | {'R@5':>6s} {'R@10':>6s} {'MRR@10':>7s}          | {'false-incl.':>11s} {'via excl. chunk':>16s}")
    for mode in EVAL_MODES:
        p, e = metrics[mode]["positive"], metrics[mode]["exclusion"]
        print(f"{mode:7s} | {fmt(p['recall@5'])} {fmt(p['recall@10'])} {fmt(p['mrr@10']):>7s}          | "
              f"{fmt(e['false_inclusion_rate']):>11s} {e['inclusions_via_exclusions_chunk']:>16d}")

    print("\nby language")
    for lang in languages:
        for mode in EVAL_MODES:
            p = metrics[mode]["by_language"][lang]["positive"]
            e = metrics[mode]["by_language"][lang]["exclusion"]
            print(f"  {lang:3s} {mode:7s} positive n={p['n']:2d}  R@5 {fmt(p['recall@5'])}  "
                  f"R@10 {fmt(p['recall@10'])}  MRR@10 {fmt(p['mrr@10'])}   "
                  f"| exclusion n={e['n']:2d}  false-incl. {fmt(e['false_inclusion_rate'])}")

    print("\npositive rows with no expected scheme in the top 10")
    for mode in EVAL_MODES:
        missed = [f"{r['id']}({r['language']})" for r in results
                  if r["test_type"] == "positive" and not r["modes"][mode]["first_hit_rank"]]
        print(f"  {mode:7s} {', '.join(missed) or '(none)'}")

    manifest = retriever.manifest
    out = {
        "split": args.split,
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "gold_sha256": file_sha256(GOLD_PATH),
        "index": {"chunks_sha256": manifest["chunks_sha256"], "chunk_count": manifest["chunk_count"],
                  "dense": manifest["dense"], "build_date": manifest["build_date"]},
        "retrieval": {"top_k": TOP_K, "candidates_per_list": CANDIDATES, "rrf_k": RRF_K},
        "counts": {"rows": len(rows), **counts, "not_scored": counts["clarify"] + counts["no_match"]},
        "metrics": metrics,
        "rows": results,
    }
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    path = EVAL_DIR / f"{args.split}_{datetime.now():%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsaved {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

"""End-to-end evaluation on a gold split: understand -> hybrid retrieval -> match.

    python scripts/evaluate_e2e.py                        # dev
    python scripts/evaluate_e2e.py --split test --final   # once, at the end

For every scoreable row, runs understand() (cached), retrieves the top 10
schemes with hybrid search on search_query_en, and matches them with
backend/app/matcher.py. Target schemes (expected for positive rows, excluded
for exclusion rows) are also matched directly, so their status is known even
when retrieval missed them.

- positive rows:  status distribution of each expected scheme; the
                  false-exclusion rate (marked not_eligible); and the
                  end-to-end rate (retrieved in the top 10 and not marked
                  not_eligible)
- exclusion rows: each excluded scheme marked not_eligible (correct),
                  eligible (wrongly included) or needs_checking (not ruled out)
- clarify rows:   the field select_clarifying_field() chooses, next to the
                  gold notes, and that field phrased in the row's language
- every row:      a trace of each pass/fail/unknown for the target schemes

Also reports profile extraction on the gold set's seven scored fields, and
the fill rate of the fields understand-v3 added. Results go to
data/eval/e2e_<split>_<timestamp>.json.

Tuning may only look at dev; test is run once, at the end (data/gold/README.md).
"""
import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))
from app.matcher import OCCUPATION_THRESHOLD, match, select_clarifying_field  # noqa: E402
from app.retrieval import default_retriever, file_sha256  # noqa: E402
from app.understand import PROMPT_VERSION, SCHEMA_KEYS, UnderstandError, phrase_question, understand  # noqa: E402
from evaluate import GOLD_PATH, score_profiles  # noqa: E402

EVAL_DIR = ROOT / "data" / "eval"
STATUSES = ("eligible", "needs_checking", "not_eligible")


def rate(n, d):
    return round(n / d, 4) if d else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--final", action="store_true", help="required to run the test split")
    ap.add_argument("--understand-version", default=None, help="default: the current version")
    ap.add_argument("--top", type=int, default=10, help="candidates retrieved and matched per row")
    args = ap.parse_args()
    if args.split == "test" and not args.final:
        ap.error("the test split is run once, at the end (see data/gold/README.md); pass --final to run it")
    version = args.understand_version or PROMPT_VERSION
    if not version.startswith("understand-"):
        version = f"understand-{version}"

    gold = [json.loads(line) for line in open(GOLD_PATH, encoding="utf-8") if line.strip()]
    rows = [r for r in gold if not r.get("skip_scoring") and r["split"] == args.split]
    print(f"{args.split}: {len(rows)} rows, {version}, hybrid top {args.top}, occupation threshold {OCCUPATION_THRESHOLD}")
    retriever = default_retriever()

    results = []
    for i, r in enumerate(rows, 1):
        try:
            u = understand(r["description"], r["language"], version=version)
        except UnderstandError as e:
            sys.exit(f"error: understand() failed on {r['id']}: {e}\n{i - 1} rows are cached; rerun to continue")
        meta = u["_meta"]
        print(f"  {i:2d}/{len(rows)} {r['id']} [{r['language']}] {u['status']:7s} "
              f"{'cached' if meta['cached'] else meta['provider'] + ', ' + str(len(meta['attempts'])) + ' call(s)'}")
        query = u["search_query_en"] or r["description"]
        profile, facts = u["profile"], u["profile"].get("other_facts", [])
        hits = retriever.search(query, k=args.top, mode="hybrid")
        matched = match(profile, facts, hits, u["confidence"], u["evidence"])
        selection = select_clarifying_field(matched, top=args.top)
        entry = {
            "id": r["id"], "language": r["language"], "test_type": r["test_type"], "notes": r["notes"],
            "query": query, "understand_status": u["status"], "understand_notes": u.get("notes", []),
            "profile": profile, "confidence": u["confidence"],
            "fallback_question": u.get("clarifying_question"),
            "candidates": [{"rank": m["rank"], "slug": m["slug"], "status": m["status"],
                            "blocking_fields": m["blocking_fields"]} for m in matched],
            "clarifying_selection": selection, "targets": [],
        }
        if r["test_type"] == "clarify" and selection:
            entry["clarifying_question"] = phrase_question(selection["field"], r["language"])
        targets = r["expected_schemes"] if r["test_type"] == "positive" else \
            r.get("excluded_schemes", []) if r["test_type"] == "exclusion" else []
        retrieved = {m["slug"]: m["rank"] for m in matched}
        for slug in targets:
            direct = match(profile, facts, [slug], u["confidence"], u["evidence"])[0]
            entry["targets"].append({"slug": slug, "retrieved_rank": retrieved.get(slug), **{
                k: direct[k] for k in ("status", "conditions", "unverified_conditions", "to_confirm",
                                       "blocking_fields", "note")}})
        results.append(entry)

    # -- metrics ------------------------------------------------------------
    pos = [t for e in results if e["test_type"] == "positive" for t in e["targets"]]
    exc = [t for e in results if e["test_type"] == "exclusion" for t in e["targets"]]
    pos_status, exc_status = Counter(t["status"] for t in pos), Counter(t["status"] for t in exc)
    metrics = {
        "positive": {"n": len(pos),
                     "status": {s: pos_status[s] for s in STATUSES},
                     "false_exclusion_rate": rate(pos_status["not_eligible"], len(pos)),
                     "retrieved_top": sum(t["retrieved_rank"] is not None for t in pos),
                     "end_to_end_rate": rate(sum(t["retrieved_rank"] is not None and t["status"] != "not_eligible"
                                                 for t in pos), len(pos))},
        "exclusion": {"n": len(exc),
                      "correct_not_eligible": rate(exc_status["not_eligible"], len(exc)),
                      "wrongly_included_eligible": rate(exc_status["eligible"], len(exc)),
                      "not_ruled_out_needs_checking": rate(exc_status["needs_checking"], len(exc)),
                      "retrieved_top": sum(t["retrieved_rank"] is not None for t in exc)},
        "candidate_status": dict(Counter(c["status"] for e in results for c in e["candidates"])),
    }
    gold_by_id = {r["id"]: r for r in rows}
    extraction = score_profiles([(e["profile"], gold_by_id[e["id"]]["expected_profile"]) for e in results])
    fill = {k: {"filled": sum(e["profile"].get(k) not in (None, []) for e in results), "rows": len(results),
                "ids": [e["id"] for e in results if e["profile"].get(k) not in (None, [])]} for k in SCHEMA_KEYS}

    # -- report -------------------------------------------------------------
    p, x = metrics["positive"], metrics["exclusion"]
    print(f"\npositive ({p['n']} expected schemes): eligible {p['status']['eligible']}, needs_checking "
          f"{p['status']['needs_checking']}, not_eligible {p['status']['not_eligible']}  | false-exclusion rate "
          f"{p['false_exclusion_rate']}  | retrieved in top {args.top}: {p['retrieved_top']}  | end-to-end "
          f"(retrieved and not excluded) {p['end_to_end_rate']}")
    print(f"exclusion ({x['n']} excluded schemes): not_eligible (correct) {x['correct_not_eligible']}, eligible "
          f"(wrongly included) {x['wrongly_included_eligible']}, needs_checking (not ruled out) "
          f"{x['not_ruled_out_needs_checking']}  | retrieved in top {args.top}: {x['retrieved_top']}")
    print(f"all top-{args.top} candidates: {metrics['candidate_status']}")

    print("\nprofile extraction (gold's seven scored fields; occupation approximate)")
    for key, f in extraction["fields"].items():
        print(f"  {key:18s} accuracy {f['accuracy']} (n={f['gold_non_null']})  false-fill {f['false_fill_rate']} "
              f"(n={f['gold_null']})")
    o = extraction["overall"]
    print(f"  {'all':18s} accuracy {o['accuracy']} (n={o['gold_non_null']})  false-fill {o['false_fill_rate']} "
          f"(n={o['gold_null']})")
    print("fill rate of the fields understand-v3 added:")
    for k, f in fill.items():
        print(f"  {k:32s} {f['filled']:2d}/{f['rows']}  {f['ids']}")

    print("\nclarify rows")
    for e in results:
        if e["test_type"] == "clarify":
            s = e["clarifying_selection"]
            print(f"  {e['id']} [{e['language']}] selector: {s['field'] if s else None} "
                  f"{'(scores ' + json.dumps(s['scores']) + ')' if s else ''}")
            print(f"      asks: {e.get('clarifying_question')}   | understand fallback: {e['fallback_question']}")
            print(f"      gold notes: {e['notes']}")

    print("\ntraces (target scheme per row)")
    for e in results:
        head = f"{e['id']} [{e['language']}] {e['test_type']}"
        if not e["targets"]:
            s = e["clarifying_selection"]
            print(f"\n{head}: no target; selector would ask about {s['field'] if s else None}")
            continue
        for t in e["targets"]:
            print(f"\n{head} -> {t['slug']}: {t['status'].upper()}  (retrieved rank {t['retrieved_rank']})"
                  + (f"  note: {t['note']}" if t["note"] else ""))
            for c in t["conditions"]:
                print(f"    {c['result']:7s}{'' if c['decisive'] else ' (non-decisive)'} {c['field']}: "
                      f"rule={json.dumps(c['constraint'], ensure_ascii=False)[:70]} profile={c['profile_value']!r}"
                      f"{' [' + c['confidence'] + ']' if c['confidence'] else ''}"
                      + (f"  ({c['note']})" if c["note"] else ""))
                if c["source_span"]:
                    print(f"             span: {c['source_span'][:110]}")
            if t["to_confirm"]:
                print(f"    to confirm: {t['to_confirm']}")
            print(f"    unverified conditions: {len(t['unverified_conditions'])}"
                  + (f", e.g. {t['unverified_conditions'][0][:90]!r}" if t["unverified_conditions"] else ""))

    out = {
        "split": args.split, "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "understand_prompt_version": version, "top": args.top, "retrieval_mode": "hybrid",
        "occupation_threshold": OCCUPATION_THRESHOLD, "gold_sha256": file_sha256(GOLD_PATH),
        "index": {k: retriever.manifest[k] for k in ("chunks_sha256", "chunk_count", "build_date")},
        "metrics": metrics, "profile_extraction": extraction, "new_field_fill": fill, "rows": results,
    }
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    path = EVAL_DIR / f"e2e_{args.split}_{datetime.now():%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\nsaved {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

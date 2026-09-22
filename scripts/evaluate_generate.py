"""End-to-end evaluation of the generation step on a gold split:
understand -> hybrid retrieval (top 10) -> match -> explain.

    python scripts/evaluate_generate.py                        # dev
    python scripts/evaluate_generate.py --split test --final   # once, at the end

Reports
- hallucinated eligibility: reasons that claim eligibility while the scheme's
  status is not eligible (after any relevance downgrade), and separately
  while the matcher's status is not eligible. The check is
  generate.claims_eligibility(), the same detector validation uses, so the
  printed reasons are for reading too.
- citation validity: the share of the model's citations that were exact
  quotes of a chunk of the right scheme, on the first attempt and over all
  attempts; and a re-check of every citation in the final output
- how often the fallback template was used, and why
- the relevance flags (relevant_unverified), the eligible schemes they
  downgraded, and every flag on gold_083 and gold_084
- no_match rows: whether any reason names the invented scheme
- every reason with its citations

Results go to data/eval/generate_<split>_<timestamp>.json. Tuning may only
look at dev; test is run once, at the end (data/gold/README.md).
"""
import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))
from app import generate  # noqa: E402
from app.generate import GENERATE_VERSION, claims_eligibility, explain, scheme_chunks  # noqa: E402
from app.matcher import match  # noqa: E402
from app.retrieval import default_retriever, file_sha256  # noqa: E402
from app.understand import PROMPT_VERSION, UnderstandError, understand  # noqa: E402
from evaluate import GOLD_PATH  # noqa: E402

EVAL_DIR = ROOT / "data" / "eval"
RELEVANCE_ROWS = ("gold_083", "gold_084")
_GENERIC_NAME_WORDS = {"yojana", "yojna", "scheme", "card", "kalyan", "samman", "the", "of", "for"}


def rate(n, d):
    return round(n / d, 4) if d else None


def invented_names(description):
    """Scheme names the person put in quotes."""
    return [m.strip() for m in re.findall(r"['‘’\"“”]([^'‘’\"“”]{4,80})['‘’\"“”]", description)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--final", action="store_true", help="required to run the test split")
    ap.add_argument("--top", type=int, default=10, help="candidates retrieved and matched per row")
    args = ap.parse_args()
    if args.split == "test" and not args.final:
        ap.error("the test split is run once, at the end (see data/gold/README.md); pass --final to run it")

    gold = [json.loads(line) for line in open(GOLD_PATH, encoding="utf-8") if line.strip()]
    rows = [r for r in gold if not r.get("skip_scoring") and r["split"] == args.split]
    print(f"{args.split}: {len(rows)} rows, {PROMPT_VERSION}, {GENERATE_VERSION}, hybrid top {args.top}, "
          f"explains the top {generate.TOP_N} not marked not_eligible")
    retriever = default_retriever()

    results = []
    for i, r in enumerate(rows, 1):
        try:
            u = understand(r["description"], r["language"])
        except UnderstandError as e:
            sys.exit(f"error: understand() failed on {r['id']}: {e}")
        query = u["search_query_en"] or r["description"]
        profile, facts = u["profile"], u["profile"].get("other_facts", [])
        matched = match(profile, facts, retriever.search(query, k=args.top, mode="hybrid"), u["confidence"],
                        u["evidence"])
        out = explain(profile, facts, matched, r["language"])
        meta = out["_meta"]
        calls = ("no call" if meta is None else "cached" if meta["cached"] else
                 "no provider answered" if meta.get("provider_error") and not meta["attempts"] else
                 f"{len(meta['attempts'])} call(s)")
        print(f"  {i:2d}/{len(rows)} {r['id']} [{r['language']}] {calls}")
        results.append({"id": r["id"], "language": r["language"], "test_type": r["test_type"],
                        "description": r["description"], "notes": r["notes"], "query": query,
                        "facts": generate.person_facts(profile, facts), "schemes": out["schemes"], "meta": meta})

    # -- metrics ------------------------------------------------------------
    explained = [(e, s) for e in results for s in e["schemes"] if s["source"] != "code"]
    code = [(e, s) for e in results for s in e["schemes"] if s["source"] == "code"]
    hallucinated = [(e["id"], s["slug"], s["status"], claims_eligibility(s["reason"], e["language"]))
                    for e, s in explained if s["status"] != "eligible" and claims_eligibility(s["reason"], e["language"])]
    hallucinated_vs_matcher = [(e["id"], s["slug"]) for e, s in explained
                               if s["matcher_status"] != "eligible" and claims_eligibility(s["reason"], e["language"])]
    caught = [(e["id"], sid, err) for e in results if e["meta"] for a in e["meta"]["attempts"]
              for sid, errs in a["scheme_errors"].items() for err in errs if "must not say they qualify" in err]

    attempts = [a for e in results if e["meta"] for a in e["meta"]["attempts"]]
    first = [e["meta"]["attempts"][0] for e in results if e["meta"] and e["meta"]["attempts"]]
    cite_first = (sum(a["citations"]["valid"] for a in first), sum(a["citations"]["total"] for a in first))
    cite_all = (sum(a["citations"]["valid"] for a in attempts), sum(a["citations"]["total"] for a in attempts))
    chunk_text = lambda slug: {c["chunk_id"]: c["raw_text"] for c in scheme_chunks(slug)}
    final_cites = [(e["id"], s["slug"], c) for e in results for s in e["schemes"] for c in s["citations"]]
    final_bad = [(i, slug, c["chunk_id"]) for i, slug, c in final_cites
                 if c["quote"] not in chunk_text(slug).get(c["chunk_id"], "")]
    fallbacks = [(e["id"], s["slug"], s["fallback_reason"]) for e, s in explained if s["source"] == "fallback"]

    error_kinds = Counter()
    for a in attempts:
        for errs in a["scheme_errors"].values():
            for err in errs:
                error_kinds[re.sub(r"^S\d+: ", "", err).split(":")[0].split(";")[0][:60]] += 1
        for err in a["structural_errors"]:
            error_kinds["structural: " + err[:50]] += 1

    flags = [(e["id"], s["slug"], s["matcher_status"], s["status"], f) for e, s in explained
             for f in s["relevant_unverified"]]
    downgraded = [(e["id"], s["slug"]) for e, s in explained
                  if s["matcher_status"] == "eligible" and s["status"] == "needs_checking"]
    no_match = []
    for e in results:
        if e["test_type"] != "no_match":
            continue
        for name in invented_names(e["description"]):
            core = " ".join(w for w in name.split() if w.lower() not in _GENERIC_NAME_WORDS)
            for s in e["schemes"]:
                text = s["reason"].casefold()
                if name.casefold() in text or (core and core.casefold() in text):
                    no_match.append((e["id"], name, s["slug"], s["reason"]))

    metrics = {
        "rows": len(results), "llm_calls": sum(len(e["meta"]["attempts"]) for e in results if e["meta"]),
        "rows_without_call": sum(e["meta"] is None for e in results),
        "schemes_explained": len(explained), "not_eligible_code_reasons": len(code),
        "hallucinated_eligibility": len(hallucinated), "hallucinated_vs_matcher_status": len(hallucinated_vs_matcher),
        "claims_caught_by_validation": len(caught),
        "citation_validity_first_attempt": {"valid": cite_first[0], "total": cite_first[1],
                                            "rate": rate(*cite_first)},
        "citation_validity_all_attempts": {"valid": cite_all[0], "total": cite_all[1], "rate": rate(*cite_all)},
        "final_citations": {"total": len(final_cites), "invalid": len(final_bad)},
        "fallback": {"schemes": len(fallbacks), "of": len(explained), "rate": rate(len(fallbacks), len(explained)),
                     "rows": len({i for i, _, _ in fallbacks})},
        "retried_rows": sum(len(e["meta"]["attempts"]) > 1 for e in results if e["meta"]),
        "relevance_flags": len(flags), "downgraded_eligible": len(downgraded),
        "no_match_rows_naming_invented_scheme": len({i for i, *_ in no_match}),
    }

    # -- report -------------------------------------------------------------
    m = metrics
    print(f"\nschemes explained by the model: {m['schemes_explained']} ({m['llm_calls']} calls, "
          f"{m['retried_rows']} rows retried, {m['rows_without_call']} rows with nothing to explain); "
          f"not_eligible reasons written in code: {m['not_eligible_code_reasons']}")
    print(f"hallucinated eligibility: {m['hallucinated_eligibility']} (vs final status), "
          f"{m['hallucinated_vs_matcher_status']} (vs matcher status); "
          f"claims caught and retried by validation: {m['claims_caught_by_validation']}")
    for h in hallucinated:
        print(f"  {h}")
    for c in caught:
        print(f"  caught: {c}")
    print(f"citation validity: first attempt {cite_first[0]}/{cite_first[1]} = {rate(*cite_first)}, "
          f"all attempts {cite_all[0]}/{cite_all[1]} = {rate(*cite_all)}; final output {len(final_cites)} "
          f"citations, {len(final_bad)} invalid {final_bad or ''}")
    print(f"fallback: {len(fallbacks)}/{len(explained)} schemes = {m['fallback']['rate']} in {m['fallback']['rows']} rows")
    for f in fallbacks:
        print(f"  {f}")
    print("validation problems, all attempts:")
    for k, n in error_kinds.most_common():
        print(f"  {n:3d}  {k}")
    print(f"relevance flags: {len(flags)} on {len({(i, s) for i, s, *_ in flags})} schemes; eligible schemes "
          f"downgraded to needs_checking: {downgraded}")
    for rid in RELEVANCE_ROWS:
        e = next((x for x in results if x["id"] == rid), None)
        if not e:
            continue
        print(f"  {rid}: {e['description']}")
        for s in e["schemes"]:
            if s["source"] == "code":
                continue
            print(f"    #{s['rank']} {s['slug']}: {s['matcher_status']} -> {s['status']}, "
                  f"{len(s['relevant_unverified'])} flag(s)")
            for f in s["relevant_unverified"]:
                print(f"        {f['fact']!r}  ~  {f['condition'][:110]!r}")
    print(f"no_match rows: {sum(e['test_type'] == 'no_match' for e in results)}; reasons naming the invented "
          f"scheme: {no_match or 'none'}")

    print("\nevery reason (rank, slug, matcher status -> final status, source)")
    for e in results:
        print(f"\n{e['id']} [{e['language']}] {e['test_type']}: {e['description']}")
        if not e["schemes"]:
            print("    (nothing to explain)")
        for s in e["schemes"]:
            arrow = s["matcher_status"] if s["matcher_status"] == s["status"] else f"{s['matcher_status']} -> {s['status']}"
            print(f"  #{s['rank']} {s['slug']} [{arrow}, {s['source']}]")
            print(f"      {s['reason']}")
            for f in s["relevant_unverified"]:
                print(f"      flag: {f['fact']!r} ~ {f['condition'][:100]!r}")
            for c in s["citations"]:
                print(f"      cite {c['chunk_id']}: {c['quote'][:160]!r}")

    out = {
        "split": args.split, "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "understand_prompt_version": PROMPT_VERSION, "generate_version": GENERATE_VERSION, "top": args.top,
        "explained_top_n": generate.TOP_N, "retrieval_mode": "hybrid", "gold_sha256": file_sha256(GOLD_PATH),
        "index": {k: retriever.manifest[k] for k in ("chunks_sha256", "chunk_count", "build_date")},
        "metrics": metrics, "hallucinated": hallucinated, "caught": caught, "fallbacks": fallbacks,
        "validation_problems": dict(error_kinds), "flags": flags, "no_match_hits": no_match, "rows": results,
    }
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    path = EVAL_DIR / f"generate_{args.split}_{datetime.now():%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\nsaved {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

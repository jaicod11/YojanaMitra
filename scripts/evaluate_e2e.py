"""End-to-end evaluation on a gold split: understand -> hybrid retrieval -> match.

    python scripts/evaluate_e2e.py                                   # dev
    python scripts/evaluate_e2e.py --compare data/eval/e2e_dev_<ts>.json   # and compare with an earlier run
    python scripts/evaluate_e2e.py --split test --final              # once, at the end

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

Also reports profile extraction on the gold set's seven scored fields and the
fill rate of the fields added to the profile after the gold set was made.

--compare takes an earlier e2e result (matcher v1 on understand-v3) and adds
a side-by-side of the metrics and a list of every verdict that changed, with
the step that changed it. Each row is replayed cumulatively: the v1 profile
with no v2 rules (which must reproduce the earlier run), then the current
profile, then each matcher rule in turn (matcher.RULES order). A verdict is
credited to the first step at which it changed.

Results go to data/eval/e2e_<split>_<timestamp>.json. Tuning may only look at
dev; test is run once, at the end (data/gold/README.md).
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
from app import matcher  # noqa: E402
from app.matcher import match, select_clarifying_field  # noqa: E402
from app.retrieval import default_retriever, file_sha256  # noqa: E402
from app.understand import PROMPT_VERSION, UnderstandError, phrase_question, understand  # noqa: E402
from evaluate import GOLD_PATH, SCORED_KEYS, score_profiles  # noqa: E402

EVAL_DIR = ROOT / "data" / "eval"
STATUSES = ("eligible", "needs_checking", "not_eligible")
NEW_FIELDS = ("education_class", "education_stage", "residence", "marital_status", "disability_percent",
              "is_minority", "bpl_household", "registered_construction_worker", "prior_benefit_schemes", "applying_for")


def rate(n, d):
    return round(n / d, 4) if d else None


def version_name(v):
    return v if v.startswith("understand-") else f"understand-{v}"


def compute_metrics(results):
    pos = [t for e in results if e["test_type"] == "positive" for t in e["targets"]]
    exc = [t for e in results if e["test_type"] == "exclusion" for t in e["targets"]]
    ps, xs = Counter(t["status"] for t in pos), Counter(t["status"] for t in exc)
    return {
        "positive": {"n": len(pos), "status": {s: ps[s] for s in STATUSES},
                     "false_exclusion_rate": rate(ps["not_eligible"], len(pos)),
                     "retrieved_top": sum(t["retrieved_rank"] is not None for t in pos),
                     "end_to_end_rate": rate(sum(t["retrieved_rank"] is not None and t["status"] != "not_eligible"
                                                 for t in pos), len(pos))},
        "exclusion": {"n": len(exc),
                      "correct_not_eligible": rate(xs["not_eligible"], len(exc)),
                      "wrongly_included_eligible": rate(xs["eligible"], len(exc)),
                      "not_ruled_out_needs_checking": rate(xs["needs_checking"], len(exc)),
                      "retrieved_top": sum(t["retrieved_rank"] is not None for t in exc)},
        "candidate_status": {s: sum(c["status"] == s for e in results for c in e["candidates"]) for s in STATUSES},
    }


def attribution_steps(v1_version, current_version):
    steps = [("v1", v1_version, ()), (current_version, current_version, ())]
    for i, rule in enumerate(matcher.RULES):
        steps.append((f"+{rule}", current_version, matcher.RULES[:i + 1]))
    return steps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--final", action="store_true", help="required to run the test split")
    ap.add_argument("--understand-version", default=None, help="default: the current version")
    ap.add_argument("--top", type=int, default=10, help="candidates retrieved and matched per row")
    ap.add_argument("--compare", type=Path, help="an earlier e2e result to compare with and attribute changes against")
    args = ap.parse_args()
    if args.split == "test" and not args.final:
        ap.error("the test split is run once, at the end (see data/gold/README.md); pass --final to run it")
    version = version_name(args.understand_version or PROMPT_VERSION)
    old = json.loads(args.compare.read_text(encoding="utf-8")) if args.compare else None

    gold = [json.loads(line) for line in open(GOLD_PATH, encoding="utf-8") if line.strip()]
    rows = [r for r in gold if not r.get("skip_scoring") and r["split"] == args.split]
    print(f"{args.split}: {len(rows)} rows, {version}, hybrid top {args.top}, occupation threshold "
          f"{matcher.OCCUPATION_THRESHOLD}, residual-touch threshold {matcher.RESIDUAL_TOUCH_THRESHOLD}, "
          f"min clarify score {matcher.MIN_CLARIFY_SCORE}")
    retriever = default_retriever()

    def understood(r, v):
        try:
            return understand(r["description"], r["language"], version=v)
        except UnderstandError as e:
            sys.exit(f"error: understand() failed on {r['id']} ({v}): {e}")

    results = []
    for i, r in enumerate(rows, 1):
        u = understood(r, version)
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
            "profile": profile, "confidence": u["confidence"], "fallback_question": u.get("clarifying_question"),
            "candidates": [{"rank": m["rank"], "slug": m["slug"], "status": m["status"], "reason": m["reason"],
                            "blocking_fields": m["blocking_fields"], "preferences": m["preferences"],
                            "residual_touches": m["residual_touches"]} for m in matched],
            "clarifying_selection": selection, "targets": [],
        }
        if r["test_type"] == "clarify" and selection and selection.get("field"):
            entry["clarifying_question"] = phrase_question(selection["field"], r["language"],
                                                           refine=selection["kind"] == "refine")
        targets = r["expected_schemes"] if r["test_type"] == "positive" else \
            r.get("excluded_schemes", []) if r["test_type"] == "exclusion" else []
        retrieved = {m["slug"]: m["rank"] for m in matched}
        for slug in targets:
            direct = match(profile, facts, [slug], u["confidence"], u["evidence"])[0]
            entry["targets"].append({"slug": slug, "retrieved_rank": retrieved.get(slug), **{
                k: direct[k] for k in ("status", "reason", "conditions", "preferences", "residual_touches",
                                       "unverified_conditions", "to_confirm", "blocking_fields", "note")}})

        # -- attribution: replay the row through cumulative steps ----------
        if old:
            old_row = next(e for e in old["rows"] if e["id"] == r["id"])
            old_status = {c["slug"]: c["status"] for c in old_row["candidates"]}
            old_status.update({t["slug"]: t["status"] for t in old_row["targets"]})
            tracked = list(dict.fromkeys([c["slug"] for c in entry["candidates"]] + list(old_status) + targets))
            chains = {slug: [] for slug in tracked}
            for label, v, rules in attribution_steps(old["understand_prompt_version"], version):
                uu = u if v == version else understood(r, v)
                for res in match(uu["profile"], uu["profile"].get("other_facts", []), tracked, uu["confidence"],
                                 uu["evidence"], rules=rules):
                    chains[res["slug"]].append((label, res["status"]))
            entry["verdict_chains"] = {
                slug: {"old": old_status.get(slug), "replayed_v1": chain[0][1], "new": chain[-1][1],
                       "in_old_top": slug in {c["slug"] for c in old_row["candidates"]},
                       "in_new_top": slug in retrieved, "steps": chain}
                for slug, chain in chains.items()}
        results.append(entry)

    metrics = compute_metrics(results)
    gold_by_id = {r["id"]: r for r in rows}
    extraction = score_profiles([(e["profile"], gold_by_id[e["id"]]["expected_profile"]) for e in results])
    fill = {k: {"filled": sum(e["profile"].get(k) not in (None, []) for e in results), "rows": len(results),
                "values": {e["id"]: e["profile"].get(k) for e in results if e["profile"].get(k) not in (None, [])}}
            for k in NEW_FIELDS}

    # -- report -------------------------------------------------------------
    def metric_lines(m):
        p, x = m["positive"], m["exclusion"]
        return [f"positive ({p['n']}): eligible {p['status']['eligible']}, needs_checking {p['status']['needs_checking']}, "
                f"not_eligible {p['status']['not_eligible']} | false-exclusion {p['false_exclusion_rate']} | "
                f"retrieved {p['retrieved_top']} | end-to-end {p['end_to_end_rate']}",
                f"exclusion ({x['n']}): correct {x['correct_not_eligible']}, wrongly included "
                f"{x['wrongly_included_eligible']}, not ruled out {x['not_ruled_out_needs_checking']} | "
                f"retrieved {x['retrieved_top']}",
                f"top-{args.top} candidates: {m['candidate_status']}"]
    print()
    if old:
        print(f"{'':12s}v1 ({old['understand_prompt_version']}, matcher v1)")
        old_metrics = {**old["metrics"], "candidate_status": {
            s: old["metrics"]["candidate_status"].get(s, 0) for s in STATUSES}}
        for line in metric_lines(old_metrics):
            print(f"{'':12s}{line}")
        print(f"{'':12s}v2 ({version}, matcher v2)")
    for line in metric_lines(metrics):
        print(f"{'':12s}{line}")

    print("\nprofile extraction (gold's seven scored fields; occupation approximate)"
          + (f" — v1 then v2" if old else ""))
    for key in SCORED_KEYS:
        f = extraction["fields"][key]
        was = old["profile_extraction"]["fields"][key] if old else None
        prefix = f"{was['accuracy']} / {was['false_fill_rate']}  ->  " if was else ""
        print(f"  {key:18s} {prefix}accuracy {f['accuracy']} (n={f['gold_non_null']})  false-fill "
              f"{f['false_fill_rate']} (n={f['gold_null']})")
    o, wo = extraction["overall"], (old["profile_extraction"]["overall"] if old else None)
    print(f"  {'all':18s} {str(wo['accuracy']) + ' / ' + str(wo['false_fill_rate']) + '  ->  ' if wo else ''}"
          f"accuracy {o['accuracy']} (n={o['gold_non_null']})  false-fill {o['false_fill_rate']} (n={o['gold_null']})")
    print("fill rate of the added profile fields:")
    for k, f in fill.items():
        print(f"  {k:32s} {f['filled']:2d}/{f['rows']}  {f['values']}")

    print("\nclarify rows" + (" (v1 selection -> v2 selection)" if old else ""))
    for e in results:
        if e["test_type"] != "clarify":
            continue
        s = e["clarifying_selection"] or {}
        was = next((x["clarifying_selection"] for x in old["rows"] if x["id"] == e["id"]), None) if old else None
        chosen = f"{s.get('field')} ({s.get('kind')}, {s.get('score')})" if s.get("field") else \
            f"nothing (best {s.get('below_min_score')} below {s.get('min_score')})" if s else "nothing (no blocking field)"
        print(f"  {e['id']} [{e['language']}] {(str(was['field']) if was else 'None') + ' -> ' if old else ''}{chosen}")
        print(f"      asks: {e.get('clarifying_question')}   | understand fallback: {e['fallback_question']}")
        print(f"      gold notes: {e['notes']}")

    if old:
        print("\nchanged verdicts (old run -> new run; credited to the first step that changed it)")
        mismatches = []
        for e in results:
            for slug, ch in e["verdict_chains"].items():
                if ch["old"] is not None and ch["old"] != ch["replayed_v1"]:
                    mismatches.append((e["id"], slug, ch["old"], ch["replayed_v1"]))
                base = ch["old"] if ch["old"] is not None else ch["replayed_v1"]
                if base == ch["new"]:
                    continue
                statuses = [s for _, s in ch["steps"]]
                first = next((label for (label, s), prev in zip(ch["steps"][1:], statuses) if s != prev),
                             "none: the replay already differs from the old run")
                path = " -> ".join(f"{s}" + (f" [{label}]" if i and s != statuses[i - 1] else "")
                                   for i, (label, s) in enumerate(ch["steps"]) if i == 0 or s != statuses[i - 1])
                where = ("target" if slug in {t["slug"] for t in e["targets"]} else "top-10")
                print(f"  {e['id']} {slug:22s} ({where}) {base} -> {ch['new']}   first changed by {first}   [{path}]")
        print(f"  replay check: v1 statuses reproduced for every scheme in the old run"
              if not mismatches else f"  REPLAY MISMATCHES (old run vs replayed v1): {mismatches}")

    print("\ntraces (target scheme per row)")
    for e in results:
        head = f"{e['id']} [{e['language']}] {e['test_type']}"
        if not e["targets"]:
            s = e["clarifying_selection"] or {}
            print(f"\n{head}: no target; selector would ask about {s.get('field')}")
            continue
        for t in e["targets"]:
            print(f"\n{head} -> {t['slug']}: {t['status'].upper()}  (retrieved rank {t['retrieved_rank']})  "
                  f"reason: {t['reason']}")
            for c in t["conditions"]:
                print(f"    {c['result']:7s}{'' if c['decisive'] else ' (non-decisive)'} {c['field']}: "
                      f"rule={json.dumps(c['constraint'], ensure_ascii=False)[:70]} profile={c['profile_value']!r}"
                      f"{' [' + c['confidence'] + ']' if c['confidence'] else ''}"
                      + (f"  ({c['note']})" if c["note"] else ""))
                if c["source_span"]:
                    print(f"             span: {c['source_span'][:110]}")
            for p in t["preferences"]:
                print(f"    preference (not checked) {p['field']}={p['value']}: {p['source_span'][:90]}")
            for tt in t["residual_touches"]:
                print(f"    residual touch {tt['similarity']}: {tt['other_fact'][:50]!r} ~ {tt['residual'][:70]!r}")
            if t["to_confirm"]:
                print(f"    to confirm: {t['to_confirm']}")
            print(f"    unverified conditions: {len(t['unverified_conditions'])}")

    out = {
        "split": args.split, "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "understand_prompt_version": version, "matcher": {
            "rules": list(matcher.RULES), "occupation_threshold": matcher.OCCUPATION_THRESHOLD,
            "residual_touch_threshold": matcher.RESIDUAL_TOUCH_THRESHOLD,
            "min_clarify_score": matcher.MIN_CLARIFY_SCORE},
        "top": args.top, "retrieval_mode": "hybrid", "gold_sha256": file_sha256(GOLD_PATH),
        "index": {k: retriever.manifest[k] for k in ("chunks_sha256", "chunk_count", "build_date")},
        "compared_with": str(args.compare) if args.compare else None,
        "metrics": metrics, "profile_extraction": extraction, "new_field_fill": fill, "rows": results,
    }
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    path = EVAL_DIR / f"e2e_{args.split}_{datetime.now():%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\nsaved {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare human labels (to_verify.csv) against model predictions (predictions.json).

Run this only after data/interim/labels/to_verify.csv has been filled in by
hand -- human_category must be non-blank for every row. Never reads model
output while a human is still labeling; that's enforced by label_categories.py,
not here.

With --weights <sample_slugs.json> it also reports the post-stratified estimate
for the pool the sample was drawn from. A verification sample is stratified on
the model's predicted category with a floor for rare classes, so the raw rate
is the agreement on that sample, not on the corpus; --exclude-stratum sizes a
single suspected error source against the rest.
"""
import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LABELS_DIR = ROOT / "data/interim/labels"
TO_VERIFY_PATH = LABELS_DIR / "to_verify.csv"
PREDICTIONS_PATH = LABELS_DIR / "predictions.json"


def load_to_verify(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


Z95 = 1.959963984540054


def wilson(p, n):
    """95% interval that still behaves at small n and at p near 0 or 1."""
    zz = Z95 * Z95
    centre = (p + zz / (2 * n)) / (1 + zz / n)
    half = Z95 * math.sqrt(p * (1 - p) / n + zz / (4 * n * n)) / (1 + zz / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def post_stratified(counts, weights, pool_size, draws=40000, seed=20260920):
    """Corpus-wide agreement from a sample stratified on the model's label.

    `counts` maps stratum -> [n_h, agree_h]; `weights` holds the sampling
    weights from sample_slugs.json, w_h = (N_h/N) / (n_h/n), so the stratum's
    share of the pool is W_h = n_h * w_h / sum. Sampling is deliberately not
    proportional (rare classes have a floor), so the raw sample rate answers a
    different question than the pool-wide one.

    Three intervals, because each fails somewhere the others do not:
      - stratified normal + FPC: the textbook one, but a stratum where every
        sampled scheme agreed contributes zero variance, so with floors of 4-8
        it reads far too narrow;
      - Kish: Wilson at the effective sample size after weighting, which is
        what the unequal weights cost;
      - Jeffreys + FPC: a Beta(1/2,1/2) posterior per stratum, then the
        unsampled remainder of that stratum drawn from it -- it does not take
        "5 of 5 agreed" as certainty, and a fully sampled stratum is exact.
    """
    total_w = sum(n_h * weights[h] for h, (n_h, _) in counts.items())
    W = {h: n_h * weights[h] / total_w for h, (n_h, _) in counts.items()}
    p = {h: a_h / n_h for h, (n_h, a_h) in counts.items()}
    est = sum(W[h] * p[h] for h in counts)
    Nh = {h: max(n_h, round(W[h] * pool_size)) for h, (n_h, _) in counts.items()} if pool_size else {}

    var = 0.0
    for h, (n_h, _) in counts.items():
        if n_h < 2:
            continue
        s2 = p[h] * (1 - p[h]) * n_h / (n_h - 1)
        fpc = (1 - n_h / Nh[h]) if Nh else 1.0
        var += W[h] ** 2 * fpc * s2 / n_h
    se = math.sqrt(var)

    unit_w = [W[h] / n_h for h, (n_h, _) in counts.items() for _ in range(n_h)]
    n_eff = sum(unit_w) ** 2 / sum(x * x for x in unit_w)

    rng = random.Random(seed)
    sims = []
    for _ in range(draws):
        acc = 0.0
        for h, (n_h, a_h) in counts.items():
            ph = rng.betavariate(a_h + 0.5, n_h - a_h + 0.5)
            rest = Nh.get(h, 0) - n_h if Nh else 0
            if rest <= 0:
                acc += W[h] * (ph if not Nh else a_h / n_h)
                continue
            unseen = (sum(rng.random() < ph for _ in range(rest)) if rest <= 60 else
                      min(rest, max(0, round(rng.gauss(rest * ph, math.sqrt(rest * ph * (1 - ph)))))))
            acc += W[h] * (a_h + unseen) / Nh[h]
        sims.append(acc)
    sims.sort()

    return {
        "estimate": est,
        "stratified_normal": (est - Z95 * se, est + Z95 * se),
        "se": se,
        "n_eff": n_eff,
        "kish_wilson": wilson(est, n_eff),
        "jeffreys_fpc": (sims[int(0.025 * draws)], sims[int(0.975 * draws) - 1]),
        "W": W,
        "p": p,
        "Nh": Nh,
        "pool_share": sum(W.values()),
        "perfect_strata": sum(1 for h in counts if p[h] == 1.0),
    }


def pct(x):
    return f"{x * 100:.1f}%"


def interval(t):
    return f"[{pct(t[0])}, {pct(t[1])}]"


def print_post_stratified(compared, predictions, design, path, exclude):
    weights = design.get("category_weights", design)
    pool_size = design.get("pool_size")
    counts = {}
    for row in compared:
        pred = predictions[row["slug"]]
        c = counts.setdefault(pred["category"], [0, 0])
        c[0] += 1
        c[1] += row["human_category"].strip() == pred["category"]

    missing = sorted(set(counts) - set(weights))
    if missing:
        sys.stdout.flush()  # keep the message after the report when stdout is a file
        print(f"error: no category_weights for predicted strata: {missing}", file=sys.stderr)
        sys.exit(1)
    n = sum(n_h for n_h, _ in counts.values())
    expected = sum(n_h * weights[h] for h, (n_h, _) in counts.items())
    if abs(expected - n) / n > 0.01:
        print(f"  note: sum(n_h * w_h) = {expected:.1f}, not {n} -- the weights were computed for a "
              f"different allocation than the rows compared here; shares renormalised.")

    res = post_stratified(counts, weights, pool_size)
    agree = sum(a for _, a in counts.values())
    print()
    print("=" * 70)
    print(f"POST-STRATIFIED AGREEMENT  (weights: {path})")
    print("=" * 70)
    if pool_size:
        print(f"pool the estimate covers : {pool_size} schemes in {len(counts)} strata "
              f"(strata = the model's predicted category)")
    print(f"raw, stratified sample   : {agree}/{n} = {pct(agree / n)}   Wilson {interval(wilson(agree / n, n))}")
    print(f"corpus-wide estimate     : {pct(res['estimate'])}")
    print(f"   Kish n_eff={res['n_eff']:.1f}, Wilson  {interval(res['kish_wilson'])}   <- headline")
    print(f"   stratified normal + FPC   {interval(res['stratified_normal'])}   "
          f"(SE {res['se'] * 100:.2f}pp; narrow -- {res['perfect_strata']} strata had no disagreements)")
    print(f"   Jeffreys + FPC            {interval(res['jeffreys_fpc'])}")
    print()
    print(f"  {'stratum':40s} {'agree':>7s} {'W_h':>7s} {'of disagreement':>16s}")
    for h in sorted(counts, key=lambda h: -res["W"][h] * (1 - res["p"][h])):
        n_h, a_h = counts[h]
        share = res["W"][h] * (1 - res["p"][h]) / (1 - res["estimate"]) if res["estimate"] < 1 else 0.0
        print(f"  {h:40s} {a_h:>3d}/{n_h:<3d} {pct(res['W'][h]):>7s} {pct(share):>16s}")

    for name in exclude:
        if name not in counts:
            sys.stdout.flush()
            print(f"\nerror: --exclude-stratum {name!r} is not a predicted stratum here", file=sys.stderr)
            sys.exit(1)
        kept = {h: c for h, c in counts.items() if h != name}
        if not kept:
            continue
        sub = post_stratified(kept, weights, pool_size)
        k_agree = sum(a for _, a in kept.values())
        k_n = sum(c for c, _ in kept.values())
        print()
        print(f"excluding schemes the model labelled {name!r}:")
        print(f"  raw            : {k_agree}/{k_n} = {pct(k_agree / k_n)}   Wilson {interval(wilson(k_agree / k_n, k_n))}")
        print(f"  corpus-wide    : {pct(sub['estimate'])}   Kish {interval(sub['kish_wilson'])}   "
              f"Jeffreys {interval(sub['jeffreys_fpc'])}")
        print(f"  covers {pct(sum(res['W'][h] for h in kept))} of the pool -- agreement on what the model did NOT")
        print(f"  call {name!r}, not what a corrected rule would score.")
    print("=" * 70)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--to-verify", type=Path, default=TO_VERIFY_PATH)
    ap.add_argument("--predictions", type=Path, default=PREDICTIONS_PATH)
    ap.add_argument("--weights", type=Path, default=None,
                    help="sample_slugs.json holding category_weights; adds the corpus-wide "
                         "post-stratified estimate, which the raw sample rate is not")
    ap.add_argument("--exclude-stratum", action="append", default=[], metavar="CATEGORY",
                    help="also report agreement with this predicted category dropped, to size "
                         "one suspected error source (repeatable)")
    args = ap.parse_args()
    if args.exclude_stratum and not args.weights:
        ap.error("--exclude-stratum needs --weights")

    if not args.to_verify.exists():
        print(f"error: {args.to_verify} not found. Run label_categories.py first.", file=sys.stderr)
        sys.exit(1)
    if not args.predictions.exists():
        print(f"error: {args.predictions} not found. Run label_categories.py first.", file=sys.stderr)
        sys.exit(1)

    rows = load_to_verify(args.to_verify)
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))

    blanks = [r["slug"] for r in rows if not r.get("human_category", "").strip()]
    if blanks:
        print("error: human_category is blank for some rows -- verification is incomplete.",
              file=sys.stderr)
        print(f"  {len(blanks)} of {len(rows)} rows still need a human_category: {blanks}",
              file=sys.stderr)
        print("Fill these in before running compare_labels.py.", file=sys.stderr)
        sys.exit(1)

    no_prediction = [r["slug"] for r in rows if r["slug"] not in predictions]
    compared = [r for r in rows if r["slug"] in predictions]

    if not compared:
        print("error: none of the human-labeled slugs have a matching prediction "
              "in predictions.json.", file=sys.stderr)
        sys.exit(1)

    n = len(compared)
    agree = 0
    near_miss = 0
    disagreements = []
    by_confidence = {}  # confidence -> [n, agree]
    by_human_category = {}  # human_category -> {"count": n, "predicted_as": Counter}

    for row in compared:
        slug = row["slug"]
        pred = predictions[slug]
        human = row["human_category"].strip()
        model_cat = pred["category"]
        confidence = pred.get("confidence", "unknown")

        by_confidence.setdefault(confidence, [0, 0])
        by_confidence[confidence][0] += 1

        is_agree = human == model_cat
        if is_agree:
            agree += 1
            by_confidence[confidence][1] += 1
        else:
            bucket = by_human_category.setdefault(human, {"count": 0, "predicted_as": {}})
            bucket["count"] += 1
            bucket["predicted_as"][model_cat] = bucket["predicted_as"].get(model_cat, 0) + 1
            disagreements.append({
                "slug": slug,
                "scheme_name": row["scheme_name"],
                "human": human,
                "model": model_cat,
                "reason": pred.get("reason", ""),
            })
            if pred.get("runner_up") == human:
                near_miss += 1

    print("=" * 70)
    print("LABEL COMPARISON REPORT")
    print("=" * 70)
    print(f"rows in to_verify.csv     : {len(rows)}")
    print(f"compared (have prediction): {n}")
    if no_prediction:
        print(f"skipped, no prediction   : {len(no_prediction)}  {no_prediction}")
    print()
    print(f"agreement rate  : {agree}/{n}  ({agree / n * 100:.1f}%)")
    print(f"near-miss rate  : {near_miss}/{n}  ({near_miss / n * 100:.1f}%)  "
          f"(human label == model's runner_up, among disagreements)")

    print()
    print("agreement by model confidence:")
    for conf in sorted(by_confidence, key=lambda c: {"high": 0, "medium": 1, "low": 2}.get(c, 3)):
        total, right = by_confidence[conf]
        print(f"  {conf:8s} {right}/{total}  ({right / total * 100:.1f}%)")

    print()
    print("disagreements by human (true) category:")
    if not by_human_category:
        print("  none -- perfect agreement")
    else:
        for cat, info in sorted(by_human_category.items(), key=lambda kv: -kv[1]["count"]):
            predicted_as = ", ".join(f"{c} x{n}" for c, n in
                                      sorted(info["predicted_as"].items(), key=lambda kv: -kv[1]))
            print(f"  {cat:40s} {info['count']:3d}  -> predicted as: {predicted_as}")

    print()
    print(f"all disagreements ({len(disagreements)}):")
    if not disagreements:
        print("  none")
    else:
        for d in disagreements:
            print(f"  [{d['slug']}] {d['scheme_name']}")
            print(f"      human: {d['human']}")
            print(f"      model: {d['model']}")
            print(f"      model reason: {d['reason']}")
    print("=" * 70)

    if args.weights:
        if not args.weights.exists():
            print(f"error: {args.weights} not found.", file=sys.stderr)
            sys.exit(1)
        print_post_stratified(compared, predictions,
                              json.loads(args.weights.read_text(encoding="utf-8")),
                              args.weights, args.exclude_stratum)


if __name__ == "__main__":
    main()

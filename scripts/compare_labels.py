#!/usr/bin/env python3
"""Compare human labels (to_verify.csv) against model predictions (predictions.json).

Run this only after data/interim/labels/to_verify.csv has been filled in by
hand -- human_category must be non-blank for every row. Never reads model
output while a human is still labeling; that's enforced by label_categories.py,
not here.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LABELS_DIR = ROOT / "data/interim/labels"
TO_VERIFY_PATH = LABELS_DIR / "to_verify.csv"
PREDICTIONS_PATH = LABELS_DIR / "predictions.json"


def load_to_verify(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--to-verify", type=Path, default=TO_VERIFY_PATH)
    ap.add_argument("--predictions", type=Path, default=PREDICTIONS_PATH)
    args = ap.parse_args()

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


if __name__ == "__main__":
    main()

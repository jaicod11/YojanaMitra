"""Step 6: how good can agreement with silver labels be, and what does it mean?

    python scripts/classifier/label_ceiling.py      (after train_eval.py)

Uses the best model's out-of-fold predictions (each scheme predicted by a model
that never saw it or its label) and compares them with:
- the 100 hand labels of the round-2 blind verification, which sampled only
  schemes labelled by gemini-3.1-flash-lite, stratified by the label at the
  time with a floor per category and Social Welfare oversampled. Raw agreement
  on that sample over-represents hard cases, so a reweighted estimate is also
  given, using the sample's own category_weights by sampling stratum.
- the 30 hand labels of round 1, reported separately (an earlier labelling run).
- the silver labels, split by the model that produced them.

Writes data/eval/classifier/label_ceiling.json.
"""
import csv
import json
from collections import Counter

from common import (OUT_DIR, ROUND1_CSV, ROUND2_CSV, ROUND2_SAMPLE, ROOT, load_dataset, load_hand_labels,
                    write_json)

ROUND2_PREDICTIONS = ROOT / "data" / "interim" / "labels" / "verification_round2" / "predictions.json"


def agreement(pairs, weights=None):
    """pairs: [(a, b)]; weights: parallel list or None."""
    if not pairs:
        return None
    weights = weights or [1.0] * len(pairs)
    hits = sum(w for (a, b), w in zip(pairs, weights) if a == b)
    return round(hits / sum(weights), 4)


def main():
    cv = json.loads((OUT_DIR / "cv_results.json").read_text(encoding="utf-8"))
    best = cv["best_model"]
    oof = {r["slug"]: r for r in csv.DictReader(open(OUT_DIR / "oof_predictions.csv", encoding="utf-8"))}
    rows = {r["slug"]: r for r in load_dataset()}
    pred = {s: r[best] for s, r in oof.items()}

    # -- round 2 -----------------------------------------------------------
    hand2 = load_hand_labels(ROUND2_CSV)
    sample = json.loads(ROUND2_SAMPLE.read_text(encoding="utf-8"))
    stratum = {s: v["category"] for s, v in json.loads(ROUND2_PREDICTIONS.read_text(encoding="utf-8")).items()}
    weights = [sample["category_weights"][stratum[s]] for s in hand2]
    silver = {s: rows[s]["category"] for s in rows}
    r2 = {
        "schemes": len(hand2),
        "sampling": {k: sample[k] for k in ("purpose", "pool_filter", "pool_size", "stratified_by", "seed")},
        "current_silver_label_models": dict(Counter(rows[s]["label_model"] for s in hand2)),
        "raw": {
            "model_vs_hand": agreement([(pred[s], hand2[s]) for s in hand2]),
            "silver_vs_hand": agreement([(silver[s], hand2[s]) for s in hand2]),
            "model_vs_silver": agreement([(pred[s], silver[s]) for s in hand2]),
        },
        "reweighted_by_sampling_stratum": {
            "model_vs_hand": agreement([(pred[s], hand2[s]) for s in hand2], weights),
            "silver_vs_hand": agreement([(silver[s], hand2[s]) for s in hand2], weights),
            "model_vs_silver": agreement([(pred[s], silver[s]) for s in hand2], weights),
        },
        "breakdown": dict(Counter(
            ("model right" if pred[s] == hand2[s] else "model wrong") + ", " +
            ("silver right" if silver[s] == hand2[s] else "silver wrong") for s in hand2)),
        "model_wrong_where_silver_wrong_same_way": sum(pred[s] == silver[s] != hand2[s] for s in hand2),
        "disagreements": [{"slug": s, "hand": hand2[s], "silver": silver[s], "model": pred[s]}
                          for s in hand2 if len({hand2[s], silver[s], pred[s]}) > 1],
    }

    # -- round 1 (supplementary) -------------------------------------------------
    hand1 = load_hand_labels(ROUND1_CSV)
    r1 = {"schemes": len(hand1),
          "current_silver_label_models": dict(Counter(rows[s]["label_model"] for s in hand1)),
          "model_vs_hand": agreement([(pred[s], hand1[s]) for s in hand1]),
          "silver_vs_hand": agreement([(silver[s], hand1[s]) for s in hand1]),
          "note": "Round 1 checked an earlier labelling run; its schemes were excluded from round 2."}

    # -- agreement with silver, by labelling model --------------------------------
    by_model = {}
    for model in sorted({r["label_model"] for r in rows.values()}):
        slugs = [s for s, r in rows.items() if r["label_model"] == model]
        classes = Counter(silver[s] for s in slugs)
        by_model[model] = {
            "schemes": len(slugs), "model_vs_silver_accuracy": agreement([(pred[s], silver[s]) for s in slugs]),
            "largest_class_share": round(classes.most_common(1)[0][1] / len(slugs), 4),
            "largest_class": classes.most_common(1)[0][0],
            "hand_verified_in_round2": sum(s in hand2 for s in slugs)}

    out = {"best_model": best, "round2_blind_verification": r2, "round1_verification": r1,
           "agreement_with_silver_by_label_model": by_model,
           "overall_model_vs_silver_accuracy": agreement([(pred[s], silver[s]) for s in rows])}
    write_json(OUT_DIR / "label_ceiling.json", out)

    print(f"best model: {best}")
    print(f"round 2 ({r2['schemes']} hand labels; silver by {r2['current_silver_label_models']}):")
    print(f"  raw          model vs hand {r2['raw']['model_vs_hand']} | silver vs hand {r2['raw']['silver_vs_hand']} | "
          f"model vs silver {r2['raw']['model_vs_silver']}")
    rw = r2["reweighted_by_sampling_stratum"]
    print(f"  reweighted   model vs hand {rw['model_vs_hand']} | silver vs hand {rw['silver_vs_hand']} | "
          f"model vs silver {rw['model_vs_silver']}")
    print(f"  breakdown: {r2['breakdown']} | model wrong in the same way as silver: "
          f"{r2['model_wrong_where_silver_wrong_same_way']}")
    print(f"round 1 ({r1['schemes']}): model vs hand {r1['model_vs_hand']} | silver vs hand {r1['silver_vs_hand']} | "
          f"silver by {r1['current_silver_label_models']}")
    for m, v in by_model.items():
        print(f"  {m:24s} n={v['schemes']:4d} model vs silver {v['model_vs_silver_accuracy']} | largest class "
              f"{v['largest_class']} ({v['largest_class_share']:.0%}) | in round 2: {v['hand_verified_in_round2']}")
    print(f"overall model vs silver: {out['overall_model_vs_silver_accuracy']}")


if __name__ == "__main__":
    main()

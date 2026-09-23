"""How much do near-duplicate schemes inflate the best model's CV score?

    python scripts/classifier/group_cv.py

The EDA found 76 near-duplicate pairs (stored bge-m3 overview vectors, cosine
>= 0.95), mostly one scheme offered in several states or under several
sub-names. Under plain stratified CV a scheme's near-twin can sit in the
training folds while it is tested, so the model partly recalls rather than
generalises.

1. Groups: connected components of the near-duplicate pairs, so schemes linked
   transitively land in one group; every other scheme is its own group.
2. The best model (bge-m3 + LogisticRegression, balanced) with its selected
   hyperparameters (C=10, chosen by the nested CV in all five original folds),
   scored three ways:
   - the original StratifiedKFold(5, shuffle, seed 42), re-run as a check that
     fixed C=10 reproduces the reported numbers;
   - StratifiedGroupKFold(5, shuffle, seed 42) with groups = component id, so
     near-duplicates never straddle a fold boundary (GroupKFold if it cannot
     stratify: every class in every test fold, each class within one scheme of
     an exact fifth of its size per fold);
   - near-duplicates dropped, keeping the first slug of each group, then the
     original StratifiedKFold on what is left.

Writes data/eval/classifier/near_duplicate_cv.json.
"""
import json
from collections import Counter

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from common import N_FOLDS, OUT_DIR, SEED, load_dataset, load_overview_embeddings, outer_cv, set_seeds, write_json
from eda import NEAR_DUPLICATE_COSINE
from train_eval import evaluate

BEST = "bge-m3 + LogisticRegression (balanced)"
BEST_C = 10.0


def best_model_spec():
    return ("embedding", lambda: Pipeline([("clf", LogisticRegression(C=BEST_C, max_iter=5000,
                                                                      class_weight="balanced"))]), False)


def near_duplicate_groups(emb):
    """Component id per scheme, from the pairs with cosine >= the EDA threshold
    (the stored vectors are unit-norm, so the dot product is the cosine)."""
    n = len(emb)
    sims = emb @ emb.T
    i, j = np.triu_indices(n, k=1)
    near = sims[i, j] >= NEAR_DUPLICATE_COSINE
    graph = coo_matrix((np.ones(near.sum()), (i[near], j[near])), shape=(n, n))
    _, component = connected_components(graph, directed=False)
    return int(near.sum()), component


def fold_support(folds, y, classes):
    return {c: [int(np.sum(y[test] == c)) for _, test in folds] for c in classes}


def stratifies_adequately(support, y):
    counts = Counter(y)
    return all(min(v) > 0 and max(abs(k - counts[c] / N_FOLDS) for k in v) <= 1 for c, v in support.items())


def row(label, design, res, n):
    s = res["summary"]
    return {"treatment": label, "folds": design, "schemes": n, "summary": s, "per_fold": res["folds"]}


def main():
    set_seeds()
    rows = load_dataset()
    slugs = np.array([r["slug"] for r in rows])
    y = np.array([r["category"] for r in rows])
    emb = load_overview_embeddings(list(slugs))
    classes = [c for c, _ in Counter(y).most_common()]
    n_pairs, component = near_duplicate_groups(emb)
    sizes = Counter(component)
    grouped = np.array([sizes[c] > 1 for c in component])
    groups = [np.flatnonzero(component == c) for c in sorted({c for c in component if sizes[c] > 1},
                                                             key=lambda c: (-sizes[c], slugs[component == c][0]))]
    print(f"{n_pairs} near-duplicate pairs -> {len(groups)} groups, {grouped.sum()} schemes; sizes "
          f"{dict(sorted(Counter(len(g) for g in groups).items()))}")

    stratified = list(outer_cv().split(np.zeros(len(y)), y))
    sgkf = list(StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(
        np.zeros(len(y)), y, component))
    support_sgkf = fold_support(sgkf, y, classes)
    adequate = stratifies_adequately(support_sgkf, y)
    if adequate:
        group_folds, group_design = sgkf, f"StratifiedGroupKFold(n_splits=5, shuffle=True, random_state={SEED})"
    else:
        group_folds = list(GroupKFold(n_splits=N_FOLDS).split(np.zeros(len(y)), y, component))
        group_design = "GroupKFold(n_splits=5), fallback: StratifiedGroupKFold could not stratify"
    print(f"StratifiedGroupKFold stratifies adequately: {adequate}; using {group_design}")

    spec = best_model_spec()
    res_strat, oof_strat = evaluate("original stratified CV, C=10", spec, None, emb, y, stratified)
    res_group, oof_group = evaluate("group-aware CV, C=10", spec, None, emb, y, group_folds)

    keep = np.ones(len(y), dtype=bool)
    for g in groups:
        keep[g[1:]] = False  # rows are sorted by slug, so g[0] is the first slug
    kept_folds = list(outer_cv().split(np.zeros(keep.sum()), y[keep]))
    res_drop, _ = evaluate("near-duplicates dropped, C=10", spec, None, emb[keep], y[keep], kept_folds)

    reported = json.loads((OUT_DIR / "cv_results.json").read_text(encoding="utf-8"))
    reported = next(m for m in reported["models"] if m["name"] == BEST)["summary"]
    reproduces = reported == res_strat["summary"]
    print(f"fixed C=10 on the original folds reproduces cv_results.json: {reproduces}")

    acc_on = lambda oof, mask: round(float(np.mean(oof[mask] == y[mask])), 4)
    removed = Counter(y[~keep])
    out = {
        "seed": SEED, "model": f"{BEST}, C={BEST_C} (the value the nested CV chose in all five original folds), "
                               "max_iter=5000, on the stored bge-m3 overview vectors",
        "near_duplicate_groups": {
            "definition": f"connected components of the scheme pairs with bge-m3 cosine >= {NEAR_DUPLICATE_COSINE}",
            "pairs": n_pairs, "groups": len(groups), "schemes_in_a_group": int(grouped.sum()),
            "singletons": int((~grouped).sum()), "group_ids_total": len(sizes),
            "group_size_distribution": {str(k): v for k, v in sorted(Counter(len(g) for g in groups).items())},
            "groups_with_mixed_labels": sum(len(set(y[g])) > 1 for g in groups),
            "groups": [{"slugs": list(slugs[g]), "labels": dict(Counter(y[g]))} for g in groups]},
        "stratification_check": {
            "rule": "every class in every test fold, and each class within one scheme of an exact fifth of its "
                    "size in every fold",
            "stratified_group_kfold_adequate": adequate, "design_used": group_design,
            "per_fold_class_support": {"StratifiedKFold (original)": fold_support(stratified, y, classes),
                                       "StratifiedGroupKFold": support_sgkf},
            "fold_sizes": {"StratifiedKFold (original)": [len(t) for _, t in stratified],
                           "StratifiedGroupKFold": [len(t) for _, t in sgkf]}},
        "results": [
            row("Original stratified CV (reported)", "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
                res_strat, len(y)),
            row("Group-aware CV", group_design, res_group, len(y)),
            row("Near-duplicates dropped (first slug of each group kept)",
                "StratifiedKFold(n_splits=5, shuffle=True, random_state=42) on the remaining schemes",
                res_drop, int(keep.sum()))],
        "reproduces_reported_numbers": reproduces,
        "accuracy_on_the_grouped_schemes": {
            "schemes": int(grouped.sum()), "original_stratified_cv": acc_on(oof_strat, grouped),
            "group_aware_cv": acc_on(oof_group, grouped)},
        "accuracy_on_the_other_schemes": {
            "schemes": int((~grouped).sum()), "original_stratified_cv": acc_on(oof_strat, ~grouped),
            "group_aware_cv": acc_on(oof_group, ~grouped)},
        "dropped": {"schemes": int((~keep).sum()), "by_class": dict(removed.most_common()),
                    "slugs": list(slugs[~keep])},
    }
    write_json(OUT_DIR / "near_duplicate_cv.json", out)
    for key in ("accuracy_on_the_grouped_schemes", "accuracy_on_the_other_schemes"):
        print(key, out[key])
    print("dropped by class:", dict(removed.most_common()))


if __name__ == "__main__":
    main()

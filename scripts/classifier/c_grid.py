"""Re-tune C for the best model with a grid that is not cut off at the top.

    python scripts/classifier/c_grid.py

train_eval.py's nested CV chose C=10, the largest value in its grid
{0.1, 1, 10}, in all five folds for bge-m3 + LogisticRegression (balanced), so
the optimum may lie above the grid. This re-runs the same nested CV for that
model only, same everything (outer StratifiedKFold(5, seed 42), inner
StratifiedKFold(3, seed 42), macro-F1 scoring, max_iter=5000), with larger C
values added. If a fold still picks the largest value, the grid is extended
once more. No other model is re-tuned.

Writes data/eval/classifier/c_grid_best_model.json.
"""
import time
import warnings
from collections import Counter

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from common import OUT_DIR, SEED, inner_cv, load_dataset, load_overview_embeddings, outer_cv, set_seeds, write_json
from train_eval import C_GRID

BEST = "bge-m3 + LogisticRegression (balanced)"
EXTENDED = C_GRID + [30.0, 100.0, 300.0, 1000.0]
EXTENDED_AGAIN = EXTENDED + [3000.0, 10000.0]


def nested_cv(emb, y, folds, grid):
    """Nested CV of the best model over grid. n_jobs=1 so that convergence
    warnings from the inner fits are caught here."""
    fold_rows, inner_curves = [], []
    n_warnings = 0
    t0 = time.time()
    for k, (train, test) in enumerate(folds):
        search = GridSearchCV(Pipeline([("clf", LogisticRegression(max_iter=5000, class_weight="balanced"))]),
                              {"clf__C": grid}, cv=inner_cv(), scoring="f1_macro", n_jobs=1, refit=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            search.fit(emb[train], y[train])
        n_warnings += sum(issubclass(w.category, ConvergenceWarning) for w in caught)
        pred = search.predict(emb[test])
        fold_rows.append({"fold": k, "C": search.best_params_["clf__C"],
                          "macro_f1": f1_score(y[test], pred, average="macro", zero_division=0),
                          "weighted_f1": f1_score(y[test], pred, average="weighted", zero_division=0),
                          "accuracy": accuracy_score(y[test], pred)})
        inner_curves.append([round(float(s), 4) for s in search.cv_results_["mean_test_score"]])
    summary = {m: {"mean": round(float(np.mean([f[m] for f in fold_rows])), 4),
                   "std": round(float(np.std([f[m] for f in fold_rows])), 4)} for m in ("macro_f1", "weighted_f1", "accuracy")}
    chosen = [f["C"] for f in fold_rows]
    print(f"  grid {grid}: C per fold {chosen} | macro-F1 {summary['macro_f1']['mean']:.4f} ± "
          f"{summary['macro_f1']['std']:.4f} | weighted-F1 {summary['weighted_f1']['mean']:.4f} | "
          f"acc {summary['accuracy']['mean']:.4f} | convergence warnings {n_warnings} | {time.time() - t0:.0f}s")
    return {"grid": grid, "C_per_fold": chosen, "at_top_of_grid": [c == max(grid) for c in chosen],
            "summary": summary, "per_fold": fold_rows, "convergence_warnings": n_warnings,
            "inner_cv_macro_f1_by_C": {"per_outer_fold": inner_curves,
                                       "mean_over_outer_folds": dict(zip(map(str, grid),
                                                                         np.round(np.mean(inner_curves, axis=0), 4).tolist()))}}


def main():
    set_seeds()
    rows = load_dataset()
    y = np.array([r["category"] for r in rows])
    emb = load_overview_embeddings([r["slug"] for r in rows])
    folds = list(outer_cv().split(np.zeros(len(y)), y))

    print(f"{BEST}, nested CV with a wider C grid")
    original = nested_cv(emb, y, folds, C_GRID)  # check: must reproduce the reported run
    runs = [original, nested_cv(emb, y, folds, EXTENDED)]
    if any(runs[-1]["at_top_of_grid"]):
        print("  still at the top of the grid in some fold; extending once more")
        runs.append(nested_cv(emb, y, folds, EXTENDED_AGAIN))
    final = runs[-1]
    settled = not any(final["at_top_of_grid"])
    print(f"  final grid {'settles in the interior' if settled else 'still pinned at its top'}; "
          f"C chosen {dict(Counter(final['C_per_fold']))}")
    write_json(OUT_DIR / "c_grid_best_model.json", {
        "seed": SEED, "model": BEST,
        "design": "nested CV as in train_eval.py: outer StratifiedKFold(5, shuffle, seed 42), inner "
                  "StratifiedKFold(3, shuffle, seed 42), macro-F1 scoring, LogisticRegression(max_iter=5000, "
                  "class_weight='balanced') on the stored bge-m3 overview vectors",
        "runs": runs, "final_grid": final["grid"], "settles_in_interior": settled,
        "note": "The first run is the original grid, re-run to check it reproduces the reported numbers. "
                "Only this model is re-tuned; every other row of comparison.csv keeps the original grid."})


if __name__ == "__main__":
    main()

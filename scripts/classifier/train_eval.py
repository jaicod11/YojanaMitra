"""Steps 4-5: train the baselines under one stratified 5-fold CV, compare
them, and analyse the best model.

    python scripts/classifier/train_eval.py [--only NAME ...]

Every model sees exactly the same folds (StratifiedKFold, 5 splits, shuffled,
seed 42). Every vectorizer lives inside a Pipeline fitted on the training part
of each fold only. The only tuning is C for the logistic regression and
LinearSVC models, chosen by an inner 3-fold CV on the training part of each
outer fold (nested CV); the outer test fold never influences it. RandomForest
is not tuned.

The bge-m3 vectors are the retrieval index's stored embeddings of each
scheme's overview chunk (name + description). Computing them before the split
is not leakage: the encoder is pretrained and frozen, each vector depends only
on its own scheme's text, and no statistic is fitted across schemes. The
classifier on top is fitted per fold like every other model.

Writes to data/eval/classifier/: cv_results.json, comparison.csv,
per_class_best.csv, oof_predictions.csv, errors_sample.json,
figures/confusion_*.png and models/best_model.joblib.
"""
import argparse
import csv
import json
import time
from collections import Counter

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

import plots
from common import (FIG_DIR, MODEL_DIR, OUT_DIR, SEED, load_dataset, load_overview_embeddings, outer_cv, inner_cv,
                    set_seeds, write_json)

C_GRID = [0.1, 1.0, 10.0]


def tfidf():
    # Fixed in advance, not tuned: word 1-2-grams, sublinear tf, rare and
    # near-ubiquitous terms dropped.
    return TfidfVectorizer(lowercase=True, strip_accents="unicode", ngram_range=(1, 2), min_df=2, max_df=0.95,
                           sublinear_tf=True)


def tuned(pipeline):
    return GridSearchCV(pipeline, {"clf__C": C_GRID}, cv=inner_cv(), scoring="f1_macro", n_jobs=-1, refit=True)


def model_specs():
    """name -> (input kind, estimator factory, tuned?)"""
    specs = {"Majority class": ("text", lambda: DummyClassifier(strategy="most_frequent"), False)}
    for balanced in (False, True):
        cw = "balanced" if balanced else None
        tag = " (balanced)" if balanced else ""
        specs[f"TF-IDF + LogisticRegression{tag}"] = ("text", lambda cw=cw: tuned(Pipeline([
            ("tfidf", tfidf()), ("clf", LogisticRegression(max_iter=5000, class_weight=cw))])), True)
        specs[f"TF-IDF + LinearSVC{tag}"] = ("text", lambda cw=cw: tuned(Pipeline([
            ("tfidf", tfidf()), ("clf", LinearSVC(class_weight=cw, random_state=SEED, max_iter=20000))])), True)
        specs[f"TF-IDF + RandomForest{tag}"] = ("text", lambda cw=cw: Pipeline([
            ("tfidf", tfidf()), ("clf", RandomForestClassifier(n_estimators=500, class_weight=cw, random_state=SEED,
                                                               n_jobs=-1))]), False)
        specs[f"bge-m3 + LogisticRegression{tag}"] = ("embedding", lambda cw=cw: tuned(Pipeline([
            ("clf", LogisticRegression(max_iter=5000, class_weight=cw))])), True)
    order = ["Majority class", "TF-IDF + LogisticRegression", "TF-IDF + LogisticRegression (balanced)",
             "TF-IDF + LinearSVC", "TF-IDF + LinearSVC (balanced)", "TF-IDF + RandomForest",
             "TF-IDF + RandomForest (balanced)", "bge-m3 + LogisticRegression", "bge-m3 + LogisticRegression (balanced)"]
    return {k: specs[k] for k in order}


def evaluate(name, spec, texts, emb, y, folds):
    kind, factory, is_tuned = spec
    X = texts if kind == "text" else emb
    oof = np.empty(len(y), dtype=object)
    fold_rows, chosen_c = [], []
    t0 = time.time()
    for k, (train, test) in enumerate(folds):
        model = factory()
        Xtr = [X[i] for i in train] if kind == "text" else X[train]
        Xte = [X[i] for i in test] if kind == "text" else X[test]
        model.fit(Xtr, y[train])
        pred = model.predict(Xte)
        oof[test] = pred
        if is_tuned:
            chosen_c.append(model.best_params_["clf__C"])
        fold_rows.append({"fold": k, "macro_f1": f1_score(y[test], pred, average="macro", zero_division=0),
                          "weighted_f1": f1_score(y[test], pred, average="weighted", zero_division=0),
                          "accuracy": accuracy_score(y[test], pred)})
    summary = {m: {"mean": round(float(np.mean([f[m] for f in fold_rows])), 4),
                   "std": round(float(np.std([f[m] for f in fold_rows])), 4)} for m in ("macro_f1", "weighted_f1", "accuracy")}
    print(f"  {name:40s} macro-F1 {summary['macro_f1']['mean']:.3f} ± {summary['macro_f1']['std']:.3f} | "
          f"weighted-F1 {summary['weighted_f1']['mean']:.3f} | acc {summary['accuracy']['mean']:.3f} | "
          f"{time.time() - t0:.0f}s" + (f" | C per fold {chosen_c}" if chosen_c else ""))
    return {"name": name, "input": kind, "tuned_C_per_fold": chosen_c, "folds": fold_rows, "summary": summary,
            "seconds": round(time.time() - t0, 1)}, oof


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", help="run only these model names (for timing)")
    args = ap.parse_args()
    set_seeds()
    rows = load_dataset()
    texts = [r["text"] for r in rows]
    y = np.array([r["category"] for r in rows])
    emb = load_overview_embeddings([r["slug"] for r in rows])
    folds = list(outer_cv().split(np.zeros(len(y)), y))
    classes = [c for c, _ in Counter(y).most_common()]
    print(f"{len(y)} schemes, {len(classes)} classes, {len(folds)} stratified folds (seed {SEED}); "
          f"smallest class per test fold: {min(Counter(y[t]).get(classes[-1], 0) for _, t in folds)}")

    specs = model_specs()
    if args.only:
        specs = {k: v for k, v in specs.items() if k in args.only}
    results, oofs = [], {}
    for name, spec in specs.items():
        res, oof = evaluate(name, spec, texts, emb, y, folds)
        results.append(res)
        oofs[name] = oof
    if args.only:
        return

    candidates = [r for r in results if r["name"] != "Majority class"]
    best = max(candidates, key=lambda r: r["summary"]["macro_f1"]["mean"])
    pred = oofs[best["name"]]
    report = classification_report(y, pred, labels=classes, output_dict=True, zero_division=0)
    per_class = [{"category": c, "precision": round(report[c]["precision"], 4), "recall": round(report[c]["recall"], 4),
                  "f1": round(report[c]["f1-score"], 4), "support": int(report[c]["support"])} for c in classes]
    cm = confusion_matrix(y, pred, labels=classes)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    confusions = sorted(((classes[i], classes[j], int(cm[i, j]), round(float(cm_norm[i, j]), 4))
                         for i in range(len(classes)) for j in range(len(classes)) if i != j and cm[i, j]),
                        key=lambda t: -t[2])
    edu, swe = "Education & Learning", "Social Welfare & Empowerment"
    edu_swe = {"education_predicted_as_social_welfare": int(cm[classes.index(edu), classes.index(swe)]),
               "social_welfare_predicted_as_education": int(cm[classes.index(swe), classes.index(edu)])}

    rng = np.random.default_rng(SEED)
    wrong = np.flatnonzero(pred != y)
    sample = sorted(rng.choice(wrong, size=min(10, len(wrong)), replace=False))
    errors = [{"slug": rows[i]["slug"], "true_silver": y[i], "predicted": pred[i], "label_model": rows[i]["label_model"],
               "scheme_name": rows[i]["scheme_name"], "snippet": rows[i]["text"][:320]} for i in sample]

    out = {"seed": SEED, "folds": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
           "tuning": f"C in {C_GRID} for LogisticRegression / LinearSVC models, by inner StratifiedKFold(3) "
                     "on each outer training fold (scoring: macro-F1); nothing else tuned",
           "tfidf": "TfidfVectorizer(lowercase, strip_accents='unicode', ngram_range=(1, 2), min_df=2, max_df=0.95, "
                    "sublinear_tf=True), fitted inside each fold",
           "models": results, "best_model": best["name"],
           "best_model_selection_note": "Chosen by mean macro-F1 on the same folds that are reported, among 8 "
                                        "candidates; the gap to the runner-up should be read with that in mind.",
           "per_class_best": per_class, "largest_confusions": confusions[:15],
           "education_vs_social_welfare": edu_swe, "errors_sample": errors, "classes": classes}
    write_json(OUT_DIR / "cv_results.json", out)
    write_json(OUT_DIR / "errors_sample.json", errors)
    with open(OUT_DIR / "comparison.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "macro_f1_mean", "macro_f1_std", "weighted_f1_mean", "weighted_f1_std", "accuracy_mean",
                    "accuracy_std", "C_per_fold"])
        for r in results:
            s = r["summary"]
            w.writerow([r["name"], s["macro_f1"]["mean"], s["macro_f1"]["std"], s["weighted_f1"]["mean"],
                        s["weighted_f1"]["std"], s["accuracy"]["mean"], s["accuracy"]["std"],
                        " ".join(map(str, r["tuned_C_per_fold"]))])
    with open(OUT_DIR / "per_class_best.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["category", "precision", "recall", "f1", "support"])
        w.writeheader()
        w.writerows(per_class)
    with open(OUT_DIR / "oof_predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["slug", "fold", "silver_label", "label_model"] + [r["name"] for r in results])
        fold_of = {int(i): k for k, (_, test) in enumerate(folds) for i in test}
        for i, r in enumerate(rows):
            w.writerow([r["slug"], fold_of[i], y[i], r["label_model"]] + [oofs[m["name"]][i] for m in results])
    plots.confusion(cm, classes, FIG_DIR / "confusion_best_counts.png", f"Confusion matrix, {best['name']} (counts, "
                    "out-of-fold over 5 folds)", normalised=False)
    plots.confusion(cm_norm, classes, FIG_DIR / "confusion_best_row_normalised.png",
                    f"Confusion matrix, {best['name']} (row-normalised)", normalised=True)

    # Final model: the best configuration refitted on all 2,066 schemes (C
    # re-chosen by the same inner CV), for later use; not used in any number above.
    kind, factory, _ = model_specs()[best["name"]]
    final = factory()
    final.fit(texts if kind == "text" else emb, y)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final, "input": kind, "classes": classes, "seed": SEED,
                 "input_text": "cleaned scheme_name + description + benefits_text (scripts/classifier/common.py)"},
                MODEL_DIR / "best_model.joblib")

    print(f"\nbest: {best['name']}")
    for p in per_class:
        print(f"  {p['category']:42s} P {p['precision']:.3f} R {p['recall']:.3f} F1 {p['f1']:.3f} n={p['support']}")
    print("largest confusions (true -> predicted, count, share of true class):")
    for t in confusions[:10]:
        print("  ", t)
    print("education vs social welfare:", edu_swe)


if __name__ == "__main__":
    main()

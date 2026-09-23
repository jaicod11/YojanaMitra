"""Steps 1-3: load the dataset, explore it, check for confounds, draw the EDA
figures and record the cleaning.

    python scripts/classifier/eda.py

Writes data/eval/classifier/eda.json and data/eval/classifier/figures/.
Anything fitted here (the term statistics, the PCA, the confound models' CV)
is for description only; no model evaluated in train_eval.py uses it.
"""
import re
import statistics
from collections import Counter, defaultdict

import numpy as np
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

import plots
from common import (FIG_DIR, LABELS_PATH, OUT_DIR, SCHEMES_DIR, SEED, clean_field, load_dataset,
                    load_overview_embeddings, outer_cv, set_seeds, write_json)

FIELDS = ("scheme_name", "description", "benefits_text", "eligibility_text", "tags")
NEAR_DUPLICATE_COSINE = 0.95


def describe(values):
    q = statistics.quantiles(values, n=4)
    return {"median": statistics.median(values), "q1": q[0], "q3": q[2], "mean": round(statistics.mean(values), 1),
            "min": min(values), "max": max(values)}


def field_stats(rows):
    out = {}
    for f in FIELDS:
        texts = [" ".join(r[f]) if f == "tags" else r[f] for r in rows]
        present = [t for t in texts if t.strip()]
        out[f] = {"empty": len(texts) - len(present),
                  "chars": describe([len(t) for t in present]),
                  "words": describe([len(t.split()) for t in present])}
    return out


def distinctive_terms(rows, classes, top=10):
    """Weighted log-odds with an informative Dirichlet prior (Monroe, Colaresi
    & Quinn 2008): per class against all other classes, on unigram counts."""
    vec = CountVectorizer(lowercase=True, stop_words="english", token_pattern=r"(?u)\b[a-z][a-z]{2,}\b", min_df=3)
    counts = vec.fit_transform([r["text"] for r in rows])
    vocab = np.array(vec.get_feature_names_out())
    labels = np.array([r["category"] for r in rows])
    prior = np.asarray(counts.sum(axis=0)).ravel() + 0.01
    a0 = prior.sum()
    out = {}
    for cls in classes:
        yi = np.asarray(counts[labels == cls].sum(axis=0)).ravel()
        yj = np.asarray(counts[labels != cls].sum(axis=0)).ravel()
        ni, nj = yi.sum(), yj.sum()
        delta = (np.log((yi + prior) / (ni + a0 - yi - prior)) - np.log((yj + prior) / (nj + a0 - yj - prior)))
        z = delta / np.sqrt(1 / (yi + prior) + 1 / (yj + prior))
        out[cls] = [str(t) for t in vocab[np.argsort(-z)[:top]]]
    return out


def duplicates(rows, emb):
    by_text = defaultdict(list)
    by_body = defaultdict(list)
    for r in rows:
        by_text[r["text"].lower()].append(r)
        by_body[(r["description"].strip().lower(), r["benefits_text"].strip().lower())].append(r)
    exact = [g for g in by_text.values() if len(g) > 1]
    body = [g for g in by_body.values() if len(g) > 1]
    sims = emb @ emb.T
    iu = np.triu_indices(len(rows), k=1)
    near = [(i, j, float(sims[i, j])) for i, j in zip(*iu) if sims[i, j] >= NEAR_DUPLICATE_COSINE]
    same_label = sum(rows[i]["category"] == rows[j]["category"] for i, j, _ in near)
    return {
        "exact_duplicate_input_texts": {"groups": len(exact), "rows": sum(len(g) for g in exact)},
        "same_description_and_benefits_different_name": {
            "groups": len(body), "rows": sum(len(g) for g in body),
            "groups_with_conflicting_labels": sum(len({r["category"] for r in g}) > 1 for g in body),
            "examples": [[r["slug"] for r in g] for g in body]},
        "near_duplicates_by_embedding": {
            "threshold_cosine": NEAR_DUPLICATE_COSINE, "method": "stored bge-m3 overview vectors (name + description)",
            "pairs": len(near), "rows_involved": len({k for i, j, _ in near for k in (i, j)}),
            "pairs_with_same_label": same_label,
            "examples": [[rows[i]["slug"], rows[j]["slug"], round(s, 3), rows[i]["category"] == rows[j]["category"]]
                         for i, j, s in sorted(near, key=lambda t: -t[2])[:8]]},
    }


def stems(category):
    """Words of a category name that could appear literally: 'Women & Child' -> women, child."""
    return [w.lower()[:5] for w in re.findall(r"[A-Za-z]+", category) if len(w) >= 4]


def literal_overlap(rows, field, classes):
    own = any_cls = 0
    for r in rows:
        text = (" ".join(r["tags"]) if field == "tags" else r[field]).lower()
        hits = {c for c in classes if any(re.search(rf"\b{s}", text) for s in stems(c))}
        own += r["category"] in hits
        any_cls += bool(hits)
    return {"own_category_word_present": round(own / len(rows), 4), "any_category_word_present": round(any_cls / len(rows), 4)}


def cv_scores(model_factory, X, y):
    """Macro-F1 and accuracy over the same stratified 5 folds as the main
    evaluation; every transform is inside the pipeline, fitted per fold."""
    f1s, accs = [], []
    for train, test in outer_cv().split(np.zeros(len(y)), y):
        model = model_factory()
        model.fit([X[i] for i in train] if isinstance(X, list) else X[train], y[train])
        pred = model.predict([X[i] for i in test] if isinstance(X, list) else X[test])
        f1s.append(f1_score(y[test], pred, average="macro", zero_division=0))
        accs.append(accuracy_score(y[test], pred))
    return {"macro_f1_mean": round(float(np.mean(f1s)), 4), "macro_f1_std": round(float(np.std(f1s)), 4),
            "accuracy_mean": round(float(np.mean(accs)), 4), "accuracy_std": round(float(np.std(accs)), 4)}


def confounds(rows, classes):
    y = np.array([r["category"] for r in rows])
    tags = [" ".join(r["tags"]) or "notags" for r in rows]
    names = [r["scheme_name"] for r in rows]
    geo = np.array([[r["level"] or "none", r["state"] or "central"] for r in rows], dtype=object)

    def text_lr():
        return make_pipeline(TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True),
                             LogisticRegression(max_iter=5000))

    return {
        "note": "Each input alone, same folds and pipeline rules as the main evaluation; LogisticRegression C=1, "
                "not tuned. tags are myScheme's own tags; the silver labeller never saw them.",
        "majority_class": cv_scores(lambda: DummyClassifier(strategy="most_frequent"), names, y),
        "tags_only": cv_scores(text_lr, tags, y),
        "scheme_name_only": cv_scores(text_lr, names, y),
        "state_and_level_only": cv_scores(
            lambda: make_pipeline(OneHotEncoder(handle_unknown="ignore"), LogisticRegression(max_iter=5000)), geo, y),
        "literal_category_words": {"tags": literal_overlap(rows, "tags", classes),
                                   "scheme_name": literal_overlap(rows, "scheme_name", classes)},
        "tags_equal_to_category_name": sum(r["category"].lower() in [t.lower() for t in r["tags"]] for r in rows),
    }


def main():
    set_seeds()
    clean_log = Counter()
    rows = load_dataset(clean_log)
    n = len(rows)
    classes = [c for c, _ in Counter(r["category"] for r in rows).most_common()]
    counts = Counter(r["category"] for r in rows)
    y_words = {c: [len(r["text"].split()) for r in rows if r["category"] == c] for c in classes}
    emb = load_overview_embeddings([r["slug"] for r in rows])

    eda = {
        "seed": SEED,
        "files_read": {"scheme_records": str(SCHEMES_DIR.relative_to(SCHEMES_DIR.parents[2])) + "/*.json",
                       "labels": str(LABELS_PATH.relative_to(LABELS_PATH.parents[3]))},
        "rows": {"scheme_records": n, "labels": n, "joined": n},
        "label_models": dict(Counter(r["label_model"] for r in rows)),
        "fields": field_stats(rows),
        "class_distribution": [{"category": c, "count": counts[c], "percent": round(100 * counts[c] / n, 2)}
                               for c in classes],
        "classes_under_25": [c for c in classes if counts[c] < 25],
        "input_text_words": {"overall": describe([len(r["text"].split()) for r in rows]),
                             "by_class": {c: describe(v) for c, v in y_words.items()}},
        "distinctive_terms": distinctive_terms(rows, classes),
        "duplicates": duplicates(rows, emb),
        "confounds": confounds(rows, classes),
        "cleaning": {
            "rows_before": n, "rows_after": n,
            "steps": {
                "mojibake repair (ftfy), per field": clean_log["mojibake_fixed"],
                "glued section heading removed ('BenefitsProvides' -> 'Provides'), per field":
                    clean_log["glued_heading_removed"],
                "run-on words split ('CardLandholding' -> 'Card Landholding'), per field": clean_log["run_on_split"],
                "whitespace collapsed, per field": clean_log["whitespace_normalised"],
            },
            "empty_text": "No record has an empty name or description, so none is dropped; the 20 records with no "
                          "benefits_text use name + description.",
            "deduplication": "Kept all rows. No two records have the same input text. Three groups (7 records) share "
                             "description and benefits under different scheme names; they are distinct schemes "
                             "with the same label in every group, so they stay. Near-duplicates are kept and "
                             "reported; they make the CV scores somewhat optimistic.",
            "lowercasing": "The stored text keeps case (the embeddings were computed on cased text); the TF-IDF "
                           "vectorizers lowercase inside the pipeline.",
        },
    }
    write_json(OUT_DIR / "eda.json", eda)

    plots.class_distribution([(c, counts[c]) for c in classes], FIG_DIR / "class_distribution.png", n)
    plots.length_by_class(y_words, FIG_DIR / "text_length_by_class.png")
    pca = PCA(n_components=2, random_state=SEED)
    xy = pca.fit_transform(emb)
    plots.projection_small_multiples(xy, [r["category"] for r in rows], classes, FIG_DIR / "pca_embeddings_by_class.png",
                                     pca.explained_variance_ratio_)

    print(f"rows: {n} | classes: {len(classes)} | under 25: {eda['classes_under_25']}")
    for c in classes:
        print(f"  {counts[c]:4d} {100 * counts[c] / n:5.1f}%  {c}")
    for f, s in eda["fields"].items():
        print(f"  {f:17s} empty {s['empty']:4d} | words median {s['words']['median']:6.0f} q1 {s['words']['q1']:6.0f} "
              f"q3 {s['words']['q3']:6.0f} max {s['words']['max']:6d}")
    print("input words:", eda["input_text_words"]["overall"])
    print("duplicates:", {k: {kk: vv for kk, vv in v.items() if kk != "examples"} for k, v in eda["duplicates"].items()})
    print("confounds:", {k: v for k, v in eda["confounds"].items() if k != "note"})
    print("cleaning:", eda["cleaning"]["steps"])
    print(f"saved {OUT_DIR / 'eda.json'} and 3 figures in {FIG_DIR}")


if __name__ == "__main__":
    main()

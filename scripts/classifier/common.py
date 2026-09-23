"""Shared pieces of the scheme-category classifier (scripts/classifier/).

The task: predict a scheme's myScheme category (15 classes) from its text.
Everything here only reads the data folders; outputs go to data/eval/classifier/.

Inputs read
- data/interim/schemes/<slug>.json            scheme records (2,066)
- data/interim/labels/predictions.json         silver category labels (LLM), one per slug, with
                                               the labelling model recorded per record
- data/interim/labels/verification_round2/to_verify.csv   100 hand labels (blind, round 2)
- data/interim/labels/verification_round1/to_verify.csv   30 hand labels (round 1)
- data/index/chunk_ids.json, data/index/embeddings/*.npy   stored bge-m3 vectors (reused, not rebuilt)
"""
import csv
import glob
import json
import random
import re
from pathlib import Path

import ftfy
import numpy as np
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
SCHEMES_DIR = ROOT / "data" / "interim" / "schemes"
LABELS_PATH = ROOT / "data" / "interim" / "labels" / "predictions.json"
ROUND1_CSV = ROOT / "data" / "interim" / "labels" / "verification_round1" / "to_verify.csv"
ROUND2_CSV = ROOT / "data" / "interim" / "labels" / "verification_round2" / "to_verify.csv"
ROUND2_SAMPLE = ROOT / "data" / "interim" / "labels" / "verification_round2" / "sample_slugs.json"
INDEX_DIR = ROOT / "data" / "index"
OUT_DIR = ROOT / "data" / "eval" / "classifier"
FIG_DIR = OUT_DIR / "figures"
MODEL_DIR = OUT_DIR / "models"

SEED = 42
N_FOLDS = 5
# The classifier reads what the silver labeller read (scripts/label_categories.py
# builds its prompt from exactly these three fields; it never sees tags or
# eligibility_text), but untruncated.
TEXT_FIELDS = ("scheme_name", "description", "benefits_text")


def set_seeds():
    random.seed(SEED)
    np.random.seed(SEED)


def outer_cv():
    """The stratified 5-fold split every reported number uses."""
    return StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)


def inner_cv():
    """Used only inside a training fold, to choose C (nested CV)."""
    return StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

# A section heading glued to the first sentence by the scraper: "BenefitsProvides ..."
_GLUED_HEADER = re.compile(r"^(?:Details|Benefits|Description|Objective|Introduction)(?=[A-Z])")
# Items run together without a space: "Aadhaar CardLandholding papers", "shops.Provides"
_RUN_ON = re.compile(r"(?<=[a-z])(?=[A-Z][a-z])|(?<=[a-z][.;:!?])(?=[A-Z])")
_SPACE = re.compile(r"\s+")


def clean_field(text, log=None):
    """Mojibake repair, glued-heading removal, run-on splitting, whitespace.
    Case is kept; the TF-IDF vectorizer lowercases. log, if given, counts
    which steps changed something."""
    raw = text or ""
    fixed = ftfy.fix_text(raw)
    if log is not None and fixed != raw:
        log["mojibake_fixed"] += 1
    stripped = _GLUED_HEADER.sub("", fixed)
    if log is not None and stripped != fixed:
        log["glued_heading_removed"] += 1
    split = _RUN_ON.sub(" ", stripped)
    if log is not None and split != stripped:
        log["run_on_split"] += 1
    out = _SPACE.sub(" ", split).strip()
    if log is not None and out != split.strip():
        log["whitespace_normalised"] += 1
    return out


def input_text(record, log=None):
    """scheme_name. description. benefits_text (empty parts skipped)."""
    parts = [clean_field(record.get(f), log) for f in TEXT_FIELDS]
    return ". ".join(p.rstrip(". ") for p in parts if p) + "."


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_records():
    return {d["slug"]: d for d in (json.loads(Path(p).read_text(encoding="utf-8"))
                                   for p in sorted(glob.glob(str(SCHEMES_DIR / "*.json"))))}


def load_labels():
    return json.loads(LABELS_PATH.read_text(encoding="utf-8"))


def load_dataset(log=None):
    """One row per scheme: slug, silver label, labelling model, the raw fields
    and the cleaned input text. Sorted by slug, so row order is fixed."""
    records, labels = load_records(), load_labels()
    if set(records) != set(labels):
        raise ValueError(f"records and labels disagree: {len(set(records) ^ set(labels))} slugs differ")
    rows = []
    for slug in sorted(records):
        r, lab = records[slug], labels[slug]
        rows.append({
            "slug": slug, "category": lab["category"], "label_model": lab["model"],
            "label_provider": lab["provider"], "label_mode": lab.get("mode"),
            "scheme_name": r.get("scheme_name") or "", "description": r.get("description") or "",
            "benefits_text": r.get("benefits_text") or "", "eligibility_text": r.get("eligibility_text") or "",
            "tags": r.get("tags") or [], "level": r.get("level"), "state": r.get("state"),
            "text": input_text(r, log),
        })
    return rows


def load_hand_labels(path):
    return {row["slug"]: row["human_category"].strip()
            for row in csv.DictReader(open(path, encoding="utf-8")) if row.get("human_category", "").strip()}


def load_overview_embeddings(slugs):
    """Stored bge-m3 vector of each scheme's overview chunk (scheme name +
    description), in the order of slugs. Read from the retrieval index's
    shards; nothing is re-embedded."""
    ids = json.loads((INDEX_DIR / "chunk_ids.json").read_text(encoding="utf-8"))
    shards = sorted(glob.glob(str(INDEX_DIR / "embeddings" / "shard_*.npy")))
    vectors = np.concatenate([np.load(s) for s in shards]).astype(np.float32)
    if len(vectors) != len(ids):
        raise ValueError(f"{len(vectors)} vectors for {len(ids)} chunk ids")
    row = {cid[: -len("::overview")]: i for i, cid in enumerate(ids) if cid.endswith("::overview")}
    missing = [s for s in slugs if s not in row]
    if missing:
        raise ValueError(f"no overview embedding for {len(missing)} schemes, e.g. {missing[:3]}")
    return vectors[[row[s] for s in slugs]]


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

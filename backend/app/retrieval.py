"""Scheme retrieval over the chunk index built by scripts/build_index.py.

search(query, k, mode) returns schemes, not chunks:

- "dense":  BAAI/bge-m3 embeddings, cosine similarity (FAISS inner product
            over normalized vectors), top CANDIDATES chunks.
- "bm25":   rank_bm25 over the same chunks, top CANDIDATES chunks with a
            positive score. This is the keyword baseline.
- "hybrid": both candidate lists fused with reciprocal rank fusion,
            1 / (RRF_K + rank), summed per chunk.

Chunk scores are aggregated to schemes by taking each slug's best chunk, and
that chunk's raw_text (the clause without its indexing prefix) is returned
as the evidence. Only the candidate chunks are
aggregated, so fewer than k schemes can come back when the candidates
concentrate on a few schemes.
"""
import hashlib
import json
import pickle
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CHUNKS_PATH = ROOT / "data" / "processed" / "chunks.jsonl"
INDEX_DIR = ROOT / "data" / "index"
MODEL_NAME = "BAAI/bge-m3"
CANDIDATES = 50
RRF_K = 60
MODES = ("hybrid", "dense", "bm25")

_STOPWORDS = frozenset("""
a an and are as at be by for from has have in is it its of on or that the this to was were will with
""".split())


def bm25_tokens(text):
    """Lowercased word tokens minus a few English function words. Shared by
    the index builder and the query side so both tokenize identically."""
    return [t for t in re.findall(r"\w+", text.lower()) if t not in _STOPWORDS]


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pick_device():
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_embedding_model(device, max_seq_length, fp16=False):
    """bge-m3 as used by both the index builder and queries.

    fp16 halves memory and runs about 1.7x faster on MPS (8 GB M1); vectors
    stay within 0.9998 cosine of fp32. It is ignored on CPU, where fp32
    queries against an fp16-built index are equivalent in practice.

    use_safetensors=False: the repo's main branch ships only pytorch_model.bin,
    and without this flag transformers also tries to fetch a bot-converted
    model.safetensors (a second 2.3 GB copy of the same weights) from a
    pull-request branch; that download stalled at 0 bytes on 2026-09-22."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME, device=device, model_kwargs={"use_safetensors": False})
    model.max_seq_length = max_seq_length
    if fp16 and device != "cpu":
        model.half()
    return model


class Retriever:
    def __init__(self, index_dir=INDEX_DIR, chunks_path=CHUNKS_PATH, device=None):
        self.index_dir = Path(index_dir)
        manifest_path = self.index_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"{manifest_path} not found; run scripts/build_index.py first")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if file_sha256(chunks_path) != self.manifest["chunks_sha256"]:
            raise RuntimeError(f"{chunks_path} changed since the index was built; rerun scripts/build_index.py")
        self.chunk_ids = json.loads((self.index_dir / "chunk_ids.json").read_text(encoding="utf-8"))
        by_id = {}
        with open(chunks_path, encoding="utf-8") as f:
            for line in f:
                c = json.loads(line)
                by_id[c["chunk_id"]] = c
        self.chunks = [by_id[i] for i in self.chunk_ids]
        with open(self.index_dir / "bm25.pkl", "rb") as f:
            self.bm25 = pickle.load(f)
        self.device = device
        self._dense_index = None
        self._model = None

    # -- candidate lists: [(chunk_position, score)], best first ------------

    def _bm25_candidates(self, query):
        scores = self.bm25.get_scores(bm25_tokens(query))
        top = np.argsort(-scores)[:CANDIDATES]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0]

    def _dense_candidates(self, query):
        if self._dense_index is None:
            import faiss
            dense = self.manifest.get("dense")
            if not dense:
                raise RuntimeError("the dense index has not been built; run scripts/build_index.py")
            if dense["model"] != MODEL_NAME:
                raise RuntimeError(f"index was built with {dense['model']}, but queries use {MODEL_NAME}")
            self._dense_index = faiss.read_index(str(self.index_dir / "dense.faiss"))
            self.device = self.device or pick_device()
            self._model = load_embedding_model(self.device, dense["max_seq_length"],
                                               fp16=dense.get("dtype") == "float16")
        q = self._model.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
        scores, idx = self._dense_index.search(q, CANDIDATES)
        return [(int(i), float(s)) for i, s in zip(idx[0], scores[0]) if i >= 0]

    # -- search ------------------------------------------------------------

    def search(self, query, k=10, mode="hybrid"):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
        ranked = {}
        if mode in ("dense", "hybrid"):
            ranked["dense"] = self._dense_candidates(query)
        if mode in ("bm25", "hybrid"):
            ranked["bm25"] = self._bm25_candidates(query)

        if mode == "hybrid":
            fused = {}
            for hits in ranked.values():
                for rank, (pos, _) in enumerate(hits, start=1):
                    fused[pos] = fused.get(pos, 0.0) + 1.0 / (RRF_K + rank)
            chunk_scores = sorted(fused.items(), key=lambda kv: -kv[1])
        else:
            chunk_scores = ranked[mode]
        ranks = {name: {pos: r for r, (pos, _) in enumerate(hits, start=1)} for name, hits in ranked.items()}

        best = {}
        for pos, score in chunk_scores:
            slug = self.chunks[pos]["slug"]
            if slug not in best or score > best[slug][1]:
                best[slug] = (pos, score)
        results = []
        for slug, (pos, score) in sorted(best.items(), key=lambda kv: -kv[1][1])[:k]:
            c = self.chunks[pos]
            results.append({
                "slug": slug,
                "scheme_name": c["scheme_name"],
                "score": score,
                "state": c["state"],
                "level": c["level"],
                "category": c["category"],
                # raw_text: the bare clause for display and citation; the
                # indexed text carries a "<scheme> — <section>:" prefix.
                "evidence": {"chunk_id": c["chunk_id"], "section": c["section"], "text": c["raw_text"],
                             **{f"{name}_rank": r.get(pos) for name, r in ranks.items()}},
            })
        return results


@lru_cache(maxsize=1)
def default_retriever():
    return Retriever()


def search(query, k=10, mode="hybrid"):
    """Top-k schemes for query, each with its best-matching chunk as evidence."""
    return default_retriever().search(query, k=k, mode=mode)

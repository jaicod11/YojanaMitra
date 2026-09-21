"""Build the retrieval index over data/processed/chunks.jsonl into data/index/.

Lexical: rank_bm25 (BM25Okapi) over backend/app/retrieval.bm25_tokens -> bm25.pkl
Dense:   BAAI/bge-m3 via sentence-transformers, normalized embeddings,
         FAISS IndexFlatIP -> dense.faiss
Also:    chunk_ids.json (the row order both indexes share) and manifest.json.

The embedding run is long and resumable. Vectors are saved in shards of
--shard-size chunks under data/index/embeddings/, each written atomically, so
a killed run loses at most the shard in progress and a rerun skips finished
shards. The shards are tied to a fingerprint of chunks.jsonl, the model and
max_seq_length; if any of those change, the script refuses to mix old and new
vectors until you pass --fresh.

    python scripts/build_index.py                  # BM25, then dense (resumes)
    python scripts/build_index.py --skip-dense     # BM25 only, seconds
    python scripts/build_index.py --benchmark 256  # time a sample, write nothing
    python scripts/build_index.py --fp16           # half precision on MPS
"""
import argparse
import json
import os
import pickle
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
from app.retrieval import (CHUNKS_PATH, INDEX_DIR, MODEL_NAME, bm25_tokens, file_sha256,  # noqa: E402
                           load_embedding_model, pick_device)

SHARD_DIR = INDEX_DIR / "embeddings"


def rel(path):
    try:
        return path.relative_to(ROOT)
    except ValueError:
        return path


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def load_chunks():
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_bm25(chunks):
    from rank_bm25 import BM25Okapi
    t0 = time.time()
    bm25 = BM25Okapi([bm25_tokens(c["text"]) for c in chunks])
    tmp = INDEX_DIR / "bm25.pkl.tmp"
    with open(tmp, "wb") as f:
        pickle.dump(bm25, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, INDEX_DIR / "bm25.pkl")
    log(f"BM25: {len(chunks)} chunks, vocabulary {len(bm25.idf)}, {time.time() - t0:.1f}s -> bm25.pkl")


def load_model(device, args):
    log(f"loading {MODEL_NAME} on {device} in {'fp16' if args.fp16 else 'fp32'} (first run downloads about 2.3 GB)")
    return load_embedding_model(device, args.max_seq_length, fp16=args.fp16)


def embed(model, texts, batch_size):
    return model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=False).astype("float32")


def benchmark(chunks, device, args):
    model = load_model(device, args)
    sample = random.Random(0).sample(chunks, min(args.benchmark, len(chunks)))
    embed(model, [c["text"] for c in sample[:args.batch_size]], args.batch_size)   # warm-up
    t0 = time.time()
    embed(model, [c["text"] for c in sample], args.batch_size)
    dt = time.time() - t0
    rate = len(sample) / dt
    total_min = len(chunks) / rate / 60
    log(f"benchmark: {len(sample)} random chunks in {dt:.1f}s = {rate:.1f} chunks/s on {device}")
    log(f"estimate for all {len(chunks)} chunks: {total_min:.0f} min")


def build_dense(chunks, device, args, fingerprint):
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    fp_path = SHARD_DIR / "fingerprint.json"
    if fp_path.exists():
        old = json.loads(fp_path.read_text(encoding="utf-8"))
        if old != fingerprint:
            sys.exit(f"error: shards in {SHARD_DIR} were made from different chunks/model/settings:\n"
                     f"  have {old}\n  want {fingerprint}\nrerun with --fresh to discard them")
    else:
        fp_path.write_text(json.dumps(fingerprint, indent=2), encoding="utf-8")

    n_shards = (len(chunks) + args.shard_size - 1) // args.shard_size
    todo = [s for s in range(n_shards) if not (SHARD_DIR / f"shard_{s:05d}.npy").exists()]
    log(f"dense: {len(chunks)} chunks in {n_shards} shards of {args.shard_size}; "
        f"{n_shards - len(todo)} already done, {len(todo)} to embed")

    if todo:
        model = load_model(device, args)
        done_chunks, t_start = 0, time.time()
        for s in todo:
            part = chunks[s * args.shard_size:(s + 1) * args.shard_size]
            t0 = time.time()
            vecs = embed(model, [c["text"] for c in part], args.batch_size)
            tmp = SHARD_DIR / f"shard_{s:05d}.tmp.npy"
            np.save(tmp, vecs)
            os.replace(tmp, SHARD_DIR / f"shard_{s:05d}.npy")
            done_chunks += len(part)
            rate = done_chunks / (time.time() - t_start)
            left = sum(min(args.shard_size, len(chunks) - t * args.shard_size) for t in todo) - done_chunks
            log(f"shard {s + 1}/{n_shards}: {len(part)} chunks in {time.time() - t0:.0f}s "
                f"| {rate:.1f} chunks/s | about {left / rate / 60:.0f} min left")

    import faiss
    vecs = np.concatenate([np.load(SHARD_DIR / f"shard_{s:05d}.npy") for s in range(n_shards)])
    assert vecs.shape[0] == len(chunks), (vecs.shape, len(chunks))
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3), f"embeddings not normalized: {norms.min()}..{norms.max()}"
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    tmp = INDEX_DIR / "dense.faiss.tmp"
    faiss.write_index(index, str(tmp))
    os.replace(tmp, INDEX_DIR / "dense.faiss")
    log(f"FAISS IndexFlatIP: {index.ntotal} vectors x {vecs.shape[1]} dims -> dense.faiss")
    return {"model": MODEL_NAME, "dims": int(vecs.shape[1]), "max_seq_length": args.max_seq_length,
            "dtype": "float16" if args.fp16 else "float32", "normalized": True,
            "faiss_index": "IndexFlatIP", "device": device}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-dense", action="store_true", help="build BM25 only")
    ap.add_argument("--benchmark", type=int, metavar="N", help="embed N random chunks, report speed, write nothing")
    ap.add_argument("--fresh", action="store_true", help="discard existing embedding shards first")
    ap.add_argument("--device", choices=["mps", "cpu"], help="default: mps if available, else cpu")
    ap.add_argument("--fp16", action="store_true",
                    help="half precision on MPS: about 1.7x faster on an 8 GB M1, vectors within 0.9998 cosine of fp32")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--shard-size", type=int, default=512)
    ap.add_argument("--max-seq-length", type=int, default=1024,
                    help="tokens per chunk for the embedding model; longer overview chunks are truncated")
    args = ap.parse_args()

    chunks = load_chunks()
    device = args.device or pick_device()
    if args.fp16 and device == "cpu":
        ap.error("--fp16 needs mps; fp16 on CPU is slower than fp32")
    log(f"{len(chunks)} chunks from {rel(CHUNKS_PATH)}; device: {device}")
    if args.benchmark:
        benchmark(chunks, device, args)
        return

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    chunks_sha = file_sha256(CHUNKS_PATH)
    (INDEX_DIR / "chunk_ids.json").write_text(json.dumps([c["chunk_id"] for c in chunks]), encoding="utf-8")
    build_bm25(chunks)

    manifest_path = INDEX_DIR / "manifest.json"
    old = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest = {
        "chunk_count": len(chunks),
        "chunks_sha256": chunks_sha,
        "bm25": {"library": "rank_bm25.BM25Okapi", "tokenizer": "backend/app/retrieval.py:bm25_tokens"},
        # Keep a previous dense build only if it was made from these same chunks.
        "dense": old.get("dense") if old.get("chunks_sha256") == chunks_sha else None,
        "build_date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    if not args.skip_dense:
        if args.fresh and SHARD_DIR.exists():
            shutil.rmtree(SHARD_DIR)
            log(f"removed old shards in {SHARD_DIR}")
        fingerprint = {"chunks_sha256": chunks_sha, "model": MODEL_NAME, "max_seq_length": args.max_seq_length,
                       "dtype": "float16" if args.fp16 else "float32"}
        manifest["dense"] = build_dense(chunks, device, args, fingerprint)
        manifest["build_date"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log(f"manifest -> {rel(manifest_path)} (dense index: {'yes' if manifest['dense'] else 'not built'})")


if __name__ == "__main__":
    main()

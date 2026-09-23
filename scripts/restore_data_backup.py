"""Restore the private Hugging Face backup of data/interim, data/index and
data/cache into a YojanaMitra checkout.

    python scripts/restore_data_backup.py            # restore into data/
    python scripts/restore_data_backup.py --check    # compare the backup with data/; downloads nothing
    python scripts/restore_data_backup.py --force    # restore even where data/<folder> already has files

Needs huggingface_hub and a Hugging Face login with read access to the
private repo (`hf auth login`, or the HF_TOKEN environment variable).

The backup's interim/, index/ and cache/ folders are downloaded to a temporary
directory and copied to data/interim, data/index and data/cache. A folder that
already has files is left alone unless --force is given; then it is renamed to
data/<folder>.before-restore-<timestamp>, never deleted.

data/processed/chunks.jsonl, the chunked corpus the index was built from, is
not in the backup. It is rebuilt from data/interim by scripts/build_chunks.py
and has to match chunks_sha256 in data/index/manifest.json, or the API refuses
to start (backend/app/retrieval.py). The rebuild was checked byte-identical
when the backup was made.
"""
import argparse
import contextlib
import hashlib
import io
import json
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

REPO_ID = "devil2411/yojanamitra-data-backup"
REPO_TYPE = "dataset"
# Folder in the backup repo -> data/<same name> in the checkout.
FOLDERS = ("interim", "index", "cache")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def target(data_dir, repo_path):
    """Where a file at repo_path in the backup belongs in the checkout."""
    return data_dir / repo_path


def check(api, data_dir, revision):
    """Compare every backed-up file with its target path (existence and size)."""
    remote = {f.path: f.size for f in api.list_repo_tree(REPO_ID, repo_type=REPO_TYPE, revision=revision, recursive=True)
              if hasattr(f, "size") and f.path.split("/", 1)[0] in FOLDERS}
    missing = [p for p in remote if not target(data_dir, p).is_file()]
    wrong_size = [p for p in remote if p not in missing and target(data_dir, p).stat().st_size != remote[p]]
    local = {p.relative_to(data_dir).as_posix() for folder in FOLDERS if (data_dir / folder).is_dir()
             for p in (data_dir / folder).rglob("*") if p.is_file() and p.name != ".DS_Store"}
    extra = sorted(local - set(remote))
    print(f"backup: {len(remote)} files, {sum(remote.values()):,} bytes, in {sorted({p.split('/', 1)[0] for p in remote})}")
    print(f"target {data_dir}: {len(remote) - len(missing) - len(wrong_size)} present with matching size, "
          f"{len(missing)} missing, {len(wrong_size)} with a different size, {len(extra)} local files not in the backup")
    for label, paths in (("missing", missing), ("different size", wrong_size), ("not in backup", extra)):
        for p in paths[:10]:
            print(f"  {label}: {p}")
    return 0 if not (missing or wrong_size or extra) else 1


def rebuild_chunks(data_dir):
    """data/processed/chunks.jsonl from data/interim, with build_chunks.py's own code."""
    import build_chunks
    build_chunks.ROOT = data_dir.parent
    build_chunks.SCHEMES_DIR = data_dir / "interim" / "schemes"
    build_chunks.PREDICTIONS_PATH = data_dir / "interim" / "labels" / "predictions.json"
    build_chunks.OUT_PATH = data_dir / "processed" / "chunks.jsonl"
    with contextlib.redirect_stdout(io.StringIO()) as log:
        build_chunks.main()
    print(log.getvalue().splitlines()[0])
    expected = json.loads((data_dir / "index" / "manifest.json").read_text(encoding="utf-8"))["chunks_sha256"]
    if sha256(build_chunks.OUT_PATH) != expected:
        sys.exit(f"error: the rebuilt {build_chunks.OUT_PATH} does not match chunks_sha256 in the index manifest")
    print("chunks.jsonl matches the index manifest")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="compare the backup with the data folder; download nothing")
    ap.add_argument("--force", action="store_true", help="restore even where data/<folder> already has files")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data", help="where to restore (default: data/ of this checkout)")
    ap.add_argument("--revision", default=None, help="a commit of the backup repo (default: the latest)")
    args = ap.parse_args()
    data_dir = args.data_dir.resolve()

    from huggingface_hub import HfApi, snapshot_download
    api = HfApi()
    if args.check:
        sys.exit(check(api, data_dir, args.revision))

    occupied = [f for f in FOLDERS if (data_dir / f).is_dir() and any((data_dir / f).iterdir())]
    if occupied and not args.force:
        sys.exit(f"error: {', '.join(str(data_dir / f) for f in occupied)} already has files; "
                 f"use --check to compare, or --force to restore (existing folders are renamed, not deleted)")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    with tempfile.TemporaryDirectory() as tmp:
        print(f"downloading {REPO_ID} ...")
        snapshot_download(REPO_ID, repo_type=REPO_TYPE, revision=args.revision, local_dir=tmp,
                          allow_patterns=[f"{f}/**" for f in FOLDERS])
        for folder in FOLDERS:
            src, dst = Path(tmp) / folder, data_dir / folder
            if not src.is_dir():
                sys.exit(f"error: the backup has no {folder}/ folder")
            if folder in occupied:
                kept = dst.with_name(f"{folder}.before-restore-{stamp}")
                dst.rename(kept)
                print(f"kept the existing {dst} as {kept}")
            shutil.copytree(src, dst)
            print(f"restored {dst} ({sum(1 for p in dst.rglob('*') if p.is_file())} files)")
    rebuild_chunks(data_dir)


if __name__ == "__main__":
    main()

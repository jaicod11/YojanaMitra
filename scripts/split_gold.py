"""Assign a dev/test split to data/gold/gold_set.jsonl.

Scoreable rows (no skip_scoring) get "dev" or "test": about DEV_SIZE rows go
to dev, allocated across test_types in proportion (largest remainder) and
drawn with a fixed seed within each type. skip_scoring rows get "none".
Deterministic: rerunning reproduces the same split. Changes no other field.
"""
import json
import random
from collections import Counter
from pathlib import Path

GOLD = Path(__file__).resolve().parent.parent / "data" / "gold" / "gold_set.jsonl"
DEV_SIZE = 20
SEED = 20260922


def dump(row):
    return json.dumps(row, ensure_ascii=False, separators=(",", ":"))


def main():
    lines = GOLD.read_text(encoding="utf-8").split("\n")[:-1]
    rows = [json.loads(l) for l in lines]
    assert [dump(r) for r in rows] == lines, "gold file does not round-trip"

    scoreable = [r for r in rows if not r.get("skip_scoring")]
    by_type = {}
    for r in scoreable:
        by_type.setdefault(r["test_type"], []).append(r["id"])

    # Largest-remainder allocation of DEV_SIZE across test_types.
    exact = {t: DEV_SIZE * len(ids) / len(scoreable) for t, ids in by_type.items()}
    alloc = {t: int(x) for t, x in exact.items()}
    for t in sorted(exact, key=lambda t: (-(exact[t] - alloc[t]), t))[:DEV_SIZE - sum(alloc.values())]:
        alloc[t] += 1

    rng = random.Random(SEED)
    dev = set()
    for t in sorted(by_type):
        dev.update(rng.sample(sorted(by_type[t]), alloc[t]))

    for r in rows:
        r["split"] = "none" if r.get("skip_scoring") else ("dev" if r["id"] in dev else "test")

    GOLD.write_text("\n".join(dump(r) for r in rows) + "\n", encoding="utf-8")
    table = Counter((r["test_type"], r["split"]) for r in rows)
    for t in sorted(by_type):
        print(f"{t:10s} dev {table[(t, 'dev')]:2d}  test {table[(t, 'test')]:2d}  none {table[(t, 'none')]}")
    print("totals:", dict(Counter(r["split"] for r in rows)))
    print("dev ids:", sorted(dev))


if __name__ == "__main__":
    main()

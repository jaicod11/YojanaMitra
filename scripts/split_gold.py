"""Assign a dev/test split to data/gold/gold_set.jsonl.

Scoreable rows (no skip_scoring) get "dev" or "test": about DEV_SIZE rows go
to dev, allocated across test_types in proportion (largest remainder) and
drawn with a fixed seed within each type. Then, for each language with no dev
row, one seeded swap exchanges a test row in that language for an English dev
row of the same test_type, so dev covers every language and the per-type
counts stay the same. skip_scoring rows get "none".
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

    # Language coverage: swap one row into dev for each language dev lacks.
    lang = {r["id"]: r["language"] for r in scoreable}
    ttype = {r["id"]: r["test_type"] for r in scoreable}
    swaps = []
    for language in sorted(set(lang.values()) - {lang[i] for i in dev}):
        candidates = sorted(i for i in lang if lang[i] == language and i not in dev
                            and any(lang[d] == "en" and ttype[d] == ttype[i] for d in dev))
        incoming = rng.choice(candidates)
        outgoing = rng.choice(sorted(d for d in dev if lang[d] == "en" and ttype[d] == ttype[incoming]))
        dev = (dev - {outgoing}) | {incoming}
        swaps.append((language, incoming, outgoing, ttype[incoming]))

    for r in rows:
        r["split"] = "none" if r.get("skip_scoring") else ("dev" if r["id"] in dev else "test")

    GOLD.write_text("\n".join(dump(r) for r in rows) + "\n", encoding="utf-8")
    table = Counter((r["test_type"], r["split"]) for r in rows)
    for t in sorted(by_type):
        print(f"{t:10s} dev {table[(t, 'dev')]:2d}  test {table[(t, 'test')]:2d}  none {table[(t, 'none')]}")
    print("totals:", dict(Counter(r["split"] for r in rows)))
    for language, incoming, outgoing, t in swaps:
        print(f"language swap ({language}, {t}): {incoming} -> dev, {outgoing} -> test")
    print("dev languages:", dict(Counter(lang[i] for i in dev)))
    print("dev ids:", sorted(dev))


if __name__ == "__main__":
    main()

"""Smoke test for backend/app/retrieval.py: print the top 5 schemes per mode
for a few hand-written queries. Not an evaluation. The gold set gets its own
harness, and these queries are deliberately not taken from it."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from app.retrieval import MODES, default_retriever  # noqa: E402

QUERIES = [
    ("en", "I lost my job and want to start a small poultry farm. Is there a loan or subsidy?"),
    ("en", "My mother is 70, bedridden, and lives alone in Assam. Is there a pension for her?"),
    ("en", "I am a fisherman in Goa and my boat was damaged in a storm."),
    ("te", "నేను ఆంధ్రప్రదేశ్‌లో చేనేత కార్మికుడిని. నాకు ఆర్థిక సహాయం కావాలి."),     # AP handloom weaver, needs financial help
    ("hi", "मैं उत्तर प्रदेश में निर्माण मज़दूर हूँ और अगले महीने मेरी बेटी की शादी है।"),  # UP construction worker, daughter's wedding next month
]


def main():
    retriever = default_retriever()
    for lang, query in QUERIES:
        print(f"\n{'=' * 100}\n[{lang}] {query}")
        for mode in MODES:
            t0 = time.time()
            results = retriever.search(query, k=5, mode=mode)
            print(f"  -- {mode} ({time.time() - t0:.2f}s)")
            if not results:
                print("     (no results: no chunk shares a keyword with the query)")
            for i, r in enumerate(results, 1):
                ev = r["evidence"]
                where = r["state"] or "central"
                print(f"     {i}. {r['slug']:24s} {r['scheme_name'][:48]:48s} [{where[:14]}] {r['score']:.4f}")
                print(f"        {ev['section']}: {ev['text'][:110]}")


if __name__ == "__main__":
    main()

"""Run the whole scheme-category classifier pipeline, in order:

    python scripts/classifier/run_all.py

1. eda.py           load, EDA, confound checks, cleaning report, EDA figures
2. train_eval.py    baselines under stratified 5-fold CV, comparison, best
                    model's per-class results, confusion matrices, errors,
                    final model (about 4 minutes on an 8-core laptop)
3. label_ceiling.py agreement with the hand-labelled verification samples
4. group_cv.py      the best model under group-aware CV (near-duplicates kept
                    in one fold) and with near-duplicates dropped
5. c_grid.py        the best model's nested CV with a wider C grid

Outputs go to data/eval/classifier/. Seed 42 throughout.
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

for step in ("eda.py", "train_eval.py", "label_ceiling.py", "group_cv.py", "c_grid.py"):
    print(f"\n=== {step} ===", flush=True)
    subprocess.run([sys.executable, str(HERE / step)], check=True)

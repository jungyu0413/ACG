"""Training-split priors.   python scripts/prepare_data.py all | <fold ...> | <dataset ...>"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from acg.config import PROFILES, dataset_of  # noqa: E402
from acg.data import folds, prepare_fold  # noqa: E402

if __name__ == "__main__":
    args = sys.argv[1:] or ["all"]
    unknown = [a for a in args if a != "all" and a not in folds() and a not in PROFILES]
    if unknown:
        sys.exit(f"unknown fold/dataset: {unknown}")
    for fold in [f for f in folds() if "all" in args or f in args or dataset_of(f) in args]:
        prepare_fold(fold)
        print(f"[{fold}] priors written", flush=True)

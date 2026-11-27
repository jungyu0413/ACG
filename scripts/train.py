"""python scripts/train.py <fold> [--seed S] [--out path] [--row paper]"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from acg.config import WEIGHTS_DIR, profile_of  # noqa: E402
from acg.train import Trainer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fold")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="default weights/acg_<fold>_seed<seed>.pth")
    ap.add_argument("--row", default=None, choices=[None, "paper"], help="published-row settings of the fold")
    a = ap.parse_args()
    out = a.out or os.path.join(WEIGHTS_DIR, f"acg_{a.fold}_seed{a.seed}.pth")
    best = Trainer(a.fold, profile_of(a.fold, a.row), a.seed).train(out, log=lambda s: print(s, flush=True))
    print(f"[{a.fold}] best rollout error {best:.3f} -> {out}")


if __name__ == "__main__":
    main()

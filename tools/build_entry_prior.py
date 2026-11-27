"""Entry/exit pairs of the same-camera training scenes -> data/priors/entry/<fold>.json.
    python tools/build_entry_prior.py zara1 zara2
First and last position of every training track; used by GenerateConfig.entry_exit_prior.
"""
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")
SAME_CAMERA = {"zara1": ["crowds_zara02", "crowds_zara03"], "zara2": ["crowds_zara01", "crowds_zara03"]}


def main():
    for fold in sys.argv[1:]:
        origins, goals = [], []
        for stem in SAME_CAMERA[fold]:
            df = pd.read_csv(os.path.join(DATA, "scenes", stem, "trajectory_dense.csv.gz")).sort_values(["agent_id", "frame"])
            for _, a in df.groupby("agent_id"):
                origins.append(a[["x", "y"]].iloc[0].tolist())
                goals.append(a[["x", "y"]].iloc[-1].tolist())
        out = os.path.join(DATA, "priors", "entry", f"{fold}.json")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        json.dump({"origins": origins, "goals": goals, "scenes": SAME_CAMERA[fold]}, open(out, "w"))
        print(f"{fold}: {len(origins)} pairs from {SAME_CAMERA[fold]} -> {out}")


if __name__ == "__main__":
    main()

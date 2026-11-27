"""python scripts/attribute_stats.py <fold|dataset ...> [--tag acg]

Share of agents and mean speed (m/s) per attribute value, generated vs ground truth, pooled over folds and trials.
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from acg.attributes import GROUPS, onehot  # noqa: E402
from acg.config import OUTPUT_DIR, dataset_of, profile_of  # noqa: E402
from acg.data import folds, load_scenes  # noqa: E402
from acg.geometry import image2world  # noqa: E402

SLOT_NAMES = ["child"] + [f"{g}={v}" for g, vals in list(GROUPS.items())[1:] for v in vals]


def track_stats(df, H, fps):
    codes, speeds = [], []
    for _, a in df.sort_values(["agent_id", "frame"]).groupby("agent_id"):
        if len(a) < 2:
            continue
        xy = image2world(a[["x", "y"]].to_numpy(), H)
        speeds.append(float(np.mean(np.hypot(*(np.diff(xy, axis=0).T)) * fps)))
        codes.append(int(a["agent_type"].iloc[0]))
    return np.stack([onehot(c) for c in codes]) if codes else np.zeros((0, 11)), np.array(speeds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folds", nargs="+")
    ap.add_argument("--tag", default="acg")
    a = ap.parse_args()
    names = [f for f in folds() if f in a.folds or dataset_of(f) in a.folds]
    gt_oh, gt_sp, gen_oh, gen_sp = [], [], [], []
    for fold in names:
        fps = profile_of(fold).dataset.fps
        for scene in load_scenes(fold, "test"):
            oh, sp = track_stats(scene.trajectory, scene.H, fps); gt_oh.append(oh); gt_sp.append(sp)
            name = scene.stem.replace("/", "_")
            for p in sorted(glob.glob(os.path.join(OUTPUT_DIR, "generated", fold, f"{name}-{a.tag}-trial*.pkl.gz"))):
                oh, sp = track_stats(pd.read_pickle(p), scene.H, fps); gen_oh.append(oh); gen_sp.append(sp)
    gt_oh, gt_sp, gen_oh, gen_sp = (np.concatenate(x) for x in (gt_oh, gt_sp, gen_oh, gen_sp))
    print(f"{len(gt_oh)} GT agents, {len(gen_oh)} generated agents ({a.tag}), folds {names}")
    print("| attribute value | share GT | share gen | speed GT | speed gen |\n|---|---:|---:|---:|---:|")
    for k, name in enumerate(SLOT_NAMES):
        g, h = gt_oh[:, k] > 0, gen_oh[:, k] > 0
        if g.sum() or h.sum():
            print(f"| {name} | {g.mean():.3f} | {h.mean():.3f} | {gt_sp[g].mean() if g.any() else float('nan'):.2f} | "
                  f"{gen_sp[h].mean() if h.any() else float('nan'):.2f} |")


if __name__ == "__main__":
    main()

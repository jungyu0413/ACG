"""Build data/scenes from the CrowdES preprocessed release.   python tools/import_crowdes.py <repo> [ethucy sdd gcs]

Per scene: seg.png, walkable.png (navmesh area; on SDD the classes tree/sidewalk/road), reference.png, H.txt and
trajectory_dense.csv.gz (agent_id, agent_type = attribute code, frame, x, y). SDD classes pedestrian / rider /
vehicle -> codes 0 / 2 / 4. The GCS test half is re-indexed from frame 0 (tracks_test.csv.gz) so that generated
scenarios span the test duration. Fold entries are written to data/folds.json.
"""
import glob
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
import yaml
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")
ETHUCY = ["eth", "hotel", "univ", "zara1", "zara2"]
SDD_CODE = {0: 0, 1: 2, 2: 4}
SDD_WALKABLE = [5, 6, 7]
GCS_SPLIT_FRAME = 96001


def split_dir(root, fold, stem):
    for split in ("test", "train"):
        if os.path.exists(os.path.join(root, fold, split, "trajectory_dense", f"{stem}_trajectory_dense.csv")):
            return os.path.join(root, fold, split)
    raise FileNotFoundError(f"{fold}/{stem}")


def copy_maps(src, stem, out, walkable_from_seg=None):
    os.makedirs(out, exist_ok=True)
    seg = np.array(Image.open(os.path.join(src, "segmentation", f"{stem}_seg.png")).convert("L"))
    Image.fromarray(seg).save(os.path.join(out, "seg.png"))
    if walkable_from_seg is None:
        walk = np.array(Image.open(os.path.join(src, "navmesh", f"{stem}_walkable_area.png")).convert("L")) > 0
    else:
        walk = np.isin(seg, walkable_from_seg)
    Image.fromarray(np.where(walk, 255, 0).astype(np.uint8)).save(os.path.join(out, "walkable.png"))
    shutil.copy(os.path.join(src, "image", f"{stem}_image.png"), os.path.join(out, "reference.png"))
    shutil.copy(os.path.join(src, "homography", f"{stem}_H.txt"), os.path.join(out, "H.txt"))
    return seg.shape


def read_tracks(src, stem, code_map=None, frame_offset=0):
    df = pd.read_csv(os.path.join(src, "trajectory_dense", f"{stem}_trajectory_dense.csv"))
    df = df[["agent_id", "agent_type", "frame", "x", "y"]].astype({"agent_id": int, "agent_type": int, "frame": int})
    if code_map:
        df["agent_type"] = df["agent_type"].map(code_map).astype(int)
    df["frame"] = df["frame"] - frame_offset
    assert df["frame"].min() >= 0
    return df


def fold_lists(repo, fold):
    cfg = yaml.safe_load(open(os.path.join(repo, "configs", "dataset", f"{fold}.yaml")))
    key = lambda s: s + "_"      # noqa: E731  (file-name order of the release)
    return sorted(cfg["dataset_test"], key=key), sorted(cfg["dataset_train"], key=key)


def main():
    repo = sys.argv[1]
    root = os.path.join(repo, "datasets", "preprocessed")
    which = sys.argv[2:] or ["ethucy", "sdd", "gcs"]
    folds_path = os.path.join(DATA, "folds.json")
    folds = json.load(open(folds_path)) if os.path.exists(folds_path) else {}
    if "ethucy" in which:
        lists = {f: fold_lists(repo, f) for f in ETHUCY}
        for stem in sorted({s for t, r in lists.values() for s in t + r}):
            fold = next(f for f, (t, r) in lists.items() if stem in t + r)
            src = split_dir(root, fold, stem)
            shape = copy_maps(src, stem, os.path.join(DATA, "scenes", stem))
            df = read_tracks(src, stem)
            df.to_csv(os.path.join(DATA, "scenes", stem, "trajectory_dense.csv.gz"), index=False, compression="gzip")
            print(f"{stem}: {shape[1]}x{shape[0]} px, {df.agent_id.nunique()} tracks, frames 0..{df.frame.max()}", flush=True)
        for f, (test, train) in lists.items():
            folds[f] = {"dataset": "ethucy", "test": test, "train": train}
    if "sdd" in which:
        test, train = fold_lists(repo, "sdd")
        for stem in test + train:
            src = split_dir(root, "sdd", stem)
            shape = copy_maps(src, stem, os.path.join(DATA, "scenes", stem), SDD_WALKABLE)
            df = read_tracks(src, stem, SDD_CODE)
            df.to_csv(os.path.join(DATA, "scenes", stem, "trajectory_dense.csv.gz"), index=False, compression="gzip")
            print(f"{stem}: {shape[1]}x{shape[0]} px, {df.agent_id.nunique()} tracks, frames 0..{df.frame.max()}", flush=True)
        folds["sdd"] = {"dataset": "sdd", "test": test, "train": train}
    if "gcs" in which:
        out = os.path.join(DATA, "scenes", "terminal")
        shape = copy_maps(os.path.join(root, "gcs", "test"), "terminal", out)
        tr = read_tracks(os.path.join(root, "gcs", "train"), "terminal")
        te = read_tracks(os.path.join(root, "gcs", "test"), "terminal", frame_offset=GCS_SPLIT_FRAME)
        tr.to_csv(os.path.join(out, "tracks_train.csv.gz"), index=False, compression="gzip")
        te.to_csv(os.path.join(out, "tracks_test.csv.gz"), index=False, compression="gzip")
        print(f"terminal: {shape[1]}x{shape[0]} px, train {tr.agent_id.nunique()} tracks (frames 0..{tr.frame.max()}), "
              f"test {te.agent_id.nunique()} tracks (frames 0..{te.frame.max()}, re-indexed)", flush=True)
        folds["gcs"] = {"dataset": "gcs", "test": ["terminal/test"], "train": ["terminal/train"]}
    json.dump(folds, open(folds_path, "w"), indent=1)
    print("->", folds_path)


if __name__ == "__main__":
    main()

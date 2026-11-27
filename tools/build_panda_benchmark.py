"""PANDA scenes from the raw re-annotation.   python tools/build_panda_benchmark.py <PANDA root> [--out data]

Input: datasets/<scene>/<scene>.json (pedestrians, labels, per-frame positions in metres) and
maps/<scene>/bev_semantic_grid.npz (0.1 m semantic grid). Tracks are split at gaps > 4 frames and interpolated over
shorter ones; agents with unknown life stage or mobility are dropped. The grid is reduced to 0.2 m/px by majority
vote and padded to a square; walkable = unannotated + grass + sidewalk + road + crosswalk, dilated 6 px. Each scene
is split into two halves of frames (tracks_0, tracks_1), each re-indexed from 0 and trimmed to the image.
"""
import argparse
import json
import os

import cv2
import numpy as np
import pandas as pd
from PIL import Image

RES = 0.2
MAX_GAP = 4
SCENES = {"p01": "01_University_Canteen", "p02": "02_OCT_Habour", "p03": "03_Xili_Crossroad",
          "p04": "04_Primary_School", "p07": "07_University_Campus", "p10": "10_Huaqiangbei"}
WALKABLE_RAW = [0, 4, 6, 7, 8]
IGNORE = 255
PALETTE = {0: (89, 101, 121), 1: (171, 135, 182), 2: (122, 156, 57), 3: (186, 219, 134), 4: (54, 118, 72),
           5: (167, 212, 229), 6: (195, 199, 205), 7: (242, 170, 97)}
PALETTE_IGNORE = (241, 241, 241)
LIFE = {"adult_appearing": 0, "older_adult_appearing": 0, "child_appearing": 1}
MOBILITY = {"on_foot": 0, "personal_mobility": 1, "dependent_mobility": 2}
CONSTRAINT = {"no_care": 0, "dependent_care": 1, "unknown_or_occluded": 0}
ATTENTION = {"no_device": 0, "device_distraction": 1, "unknown_or_occluded": 0}


def attr_code(lab):
    if lab["apparent_life_stage"] == "unknown" or lab["mobility_mode"] == "unknown":
        return None
    return (LIFE[lab["apparent_life_stage"]] + 2 * MOBILITY[lab["mobility_mode"]]
            + 6 * CONSTRAINT[lab["constraint_state"]] + 18 * ATTENTION[lab["attention_state"]])


def load_tracks(panda, stem):
    name = SCENES[stem]
    d = json.load(open(f"{panda}/datasets/{name}/{name}.json"))
    tracks = []
    for pid, p in d["pedestrians"].items():
        code = attr_code(p["labels"])
        if code is None:
            continue
        tr = sorted(p["trajectory"], key=lambda t: t["frame_id"])
        fr = np.array([t["frame_id"] - 1 for t in tr])
        xy = np.array([[t["x"], t["y"]] for t in tr], float)
        cuts = np.where(np.diff(fr) > MAX_GAP)[0] + 1
        for f_seg, xy_seg in zip(np.split(fr, cuts), np.split(xy, cuts)):
            if len(f_seg) < 2:
                continue
            full = np.arange(f_seg[0], f_seg[-1] + 1)
            x = np.interp(full, f_seg, xy_seg[:, 0]); y = np.interp(full, f_seg, xy_seg[:, 1])
            tracks.append(dict(code=int(code), frames=full, xy=np.stack([x, y], 1)))
    return tracks, int(d["n_frames"])


def raster(panda, stem):
    g = np.load(f"{panda}/maps/{SCENES[stem]}/bev_semantic_grid.npz")
    sem = g["semantic_labels"].astype(np.int64)
    x_min, y_min, x_max, y_max = [float(v) for v in g["extent_xyxy_m"]]
    H0, W0 = sem.shape[0] // 2 * 2, sem.shape[1] // 2 * 2
    sem = sem[:H0, :W0]
    pooled = np.eye(9, dtype=np.int32)[sem].reshape(H0 // 2, 2, W0 // 2, 2, 9).sum((1, 3))
    seg = pooled.argmax(-1).astype(np.uint8)
    h, w = seg.shape
    S = max(h, w)
    seg_raw = np.zeros((S, S), np.uint8); seg_raw[:h, :w] = seg
    pad = np.ones((S, S), bool); pad[:h, :w] = False
    H = np.array([[RES, 0, x_min + RES / 2], [0, -RES, y_max - RES / 2], [0, 0, 1.0]])
    return seg_raw, pad, H, dict(side=S, x_min=x_min, y_max=y_max)


def walkable(seg_raw, pad):
    walk = (np.isin(seg_raw, WALKABLE_RAW) & ~pad).astype(np.uint8) * 255
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
    return ((cv2.dilate(walk, kernel, iterations=6) > 0) & ~pad).astype(np.uint8) * 255


def class_map(seg_raw, pad):
    out = np.full(seg_raw.shape, IGNORE, np.uint8)
    v = (seg_raw > 0) & ~pad
    out[v] = seg_raw[v] - 1
    return out


def to_pixels(xy, meta):
    u = (xy[:, 0] - meta["x_min"]) / RES - 0.5
    v = (meta["y_max"] - xy[:, 1]) / RES - 0.5
    return np.stack([np.clip(u, 0, meta["side"] - 1), np.clip(v, 0, meta["side"] - 1)], 1)


def half_tracks(tracks, meta, lo, hi):
    rows = []
    for aid, t in enumerate(t for t in tracks if ((t["frames"] >= lo) & (t["frames"] < hi)).sum() >= 2):
        m = (t["frames"] >= lo) & (t["frames"] < hi)
        px = to_pixels(t["xy"][m], meta)
        rows += [(aid, t["code"], int(f) - lo, float("%.4f" % u), float("%.4f" % v)) for f, (u, v) in zip(t["frames"][m], px)]
    return pd.DataFrame(rows, columns=["agent_id", "agent_type", "frame", "x", "y"])


def clean_tracks(df, width, height):
    kept = []
    for _, traj in df.sort_values(["agent_id", "frame"]).groupby("agent_id"):
        mask = (traj["x"] >= 0) & (traj["x"] <= width - 1) & (traj["y"] >= 0) & (traj["y"] <= height - 1)
        start, end = np.argmax(mask), len(mask) - np.argmax(mask[::-1]) - 1
        if start < end:
            t = traj.iloc[start:end + 1].copy()
            t["x"] = t["x"].clip(0, width - 1); t["y"] = t["y"].clip(0, height - 1)
            t["agent_id"] = len(kept)
            kept.append(t)
    return pd.concat(kept, axis=0, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("panda", help="root of the raw PANDA re-annotation (contains datasets/ and maps/)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
    a = ap.parse_args()
    folds_path = os.path.join(a.out, "folds.json")
    folds = json.load(open(folds_path)) if os.path.exists(folds_path) else {}
    for stem in SCENES:
        tracks, n_frames = load_tracks(a.panda, stem)
        seg_raw, pad, H, meta = raster(a.panda, stem)
        seg = class_map(seg_raw, pad)
        img = np.zeros((*seg.shape, 3), np.uint8); img[:] = PALETTE_IGNORE
        for k, col in PALETTE.items():
            img[seg == k] = col
        d = os.path.join(a.out, "scenes", stem)
        os.makedirs(d, exist_ok=True)
        Image.fromarray(seg).save(f"{d}/seg.png")
        Image.fromarray(walkable(seg_raw, pad)).save(f"{d}/walkable.png")
        Image.fromarray(img).save(f"{d}/reference.png")
        np.savetxt(f"{d}/H.txt", H)
        half = n_frames // 2
        for k, (lo, hi) in enumerate(((0, half), (half, n_frames))):
            df = clean_tracks(half_tracks(tracks, meta, lo, hi), meta["side"], meta["side"])
            df.to_csv(f"{d}/tracks_{k}.csv.gz", index=False, compression="gzip")
        folds[f"{stem}A"] = {"dataset": "panda", "test": [f"{stem}/1"], "train": [f"{stem}/0"]}
        folds[f"{stem}B"] = {"dataset": "panda", "test": [f"{stem}/0"], "train": [f"{stem}/1"]}
        print(f"{stem}: {meta['side']} px ({meta['side'] * RES:.0f} m), {n_frames} frames, {len(tracks)} tracks")
    json.dump(folds, open(folds_path, "w"), indent=1)
    print("->", folds_path)


if __name__ == "__main__":
    main()

"""Scenes, folds and training-split priors.

data/scenes/<scene>/{seg.png, walkable.png, reference.png, H.txt, trajectory_dense.csv.gz | tracks_<part>.csv.gz}
data/folds.json: fold -> {"dataset", "test": [parts], "train": [parts]}, a part being "<scene>" or "<scene>/<part>"
data/priors/<fold>/*.json: written by prepare_fold; priors/population/<scene>.json and priors/entry/<fold>.json shipped
"""
import functools
import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import DATA_DIR, profile_of
from .geometry import image2world, load_homography, load_walkable

COLUMNS = ["scene", "agent_id", "agent_type", "frame", "x", "y"]
PRIOR_FILES = ("life", "life0", "goaldist", "attr_dist", "pop", "group")


@functools.lru_cache(maxsize=None)
def folds():
    return json.load(open(os.path.join(DATA_DIR, "folds.json")))


def stems_of(fold, split):
    return list(folds()[fold][split])


def prior_path(fold, kind):
    return os.path.join(DATA_DIR, "priors", fold, f"{kind}.json")


def tracks_path(stem):
    scene, _, part = stem.partition("/")
    return os.path.join(DATA_DIR, "scenes", scene, f"tracks_{part}.csv.gz" if part else "trajectory_dense.csv.gz")


@functools.lru_cache(maxsize=None)
def load_dense(stem):
    df = pd.read_csv(tracks_path(stem))
    df.insert(0, "scene", stem)
    return df[COLUMNS].astype({"agent_id": int, "agent_type": int, "frame": int})


def scene_info(stem, ds):
    df = load_dense(stem)
    H, W = load_walkable(stem).shape
    return {"width": W, "height": H, "length": int(df["frame"].max()) + 1, "num_agents": int(df["agent_id"].nunique()),
            "fps": ds.fps, "sim_fps": ds.sim_fps}


@dataclass
class Scene:
    stem: str
    H: np.ndarray             # image -> metres
    info: dict
    trajectory: pd.DataFrame  # dense ground truth, image px


def load_scenes(fold, split):
    ds = profile_of(fold).dataset
    return [Scene(stem, load_homography(stem), scene_info(stem, ds), load_dense(stem)) for stem in stems_of(fold, split)]


def _group_sizes(tracks, step):
    """Co-moving groups: tracks starting within 3 steps / 2.5 m and ending within 3 steps / 3.0 m of each other."""
    a = tracks.agg(f0=("frame", "first"), x0=("x", "first"), y0=("y", "first"),
                   f1=("frame", "last"), x1=("x", "last"), y1=("y", "last")).reset_index()
    X = a[["f0", "x0", "y0", "f1", "x1", "y1"]].to_numpy()
    n = len(a)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        dd = X[i + 1:] - X[i]
        ok = ((np.abs(dd[:, 0]) <= 3 * step) & (np.hypot(dd[:, 1], dd[:, 2]) <= 2.5)
              & (np.abs(dd[:, 3]) <= 3 * step) & (np.hypot(dd[:, 4], dd[:, 5]) <= 3.0))
        for j in np.nonzero(ok)[0] + i + 1:
            parent[find(j)] = find(i)
    return pd.Series([find(i) for i in range(n)]).value_counts().tolist()


def prepare_priors(fold):
    """Training-split statistics: track lengths in simulator steps (all tracks / tracks present at the scene start),
    origin-goal distance quantiles (metres), attribute-code distribution, crowd-size histogram and group sizes."""
    prof = profile_of(fold)
    step = prof.dataset.step
    life, life0, dist_m, codes, pop, sizes = [], [], [], [], [], []
    for stem in stems_of(fold, "train"):
        g = load_dense(stem).copy()
        g[["x", "y"]] = image2world(g[["x", "y"]].to_numpy(), load_homography(stem))
        f0 = g["frame"].min()
        pop += g[g["frame"] % step == 0].groupby("frame").size().tolist()
        tracks = g.sort_values(["agent_id", "frame"]).groupby("agent_id")
        for _, a in tracks:
            xy = a[["x", "y"]].to_numpy()
            n = int(round(len(a) / step))
            life.append(n)
            if a["frame"].iloc[0] <= f0:
                life0.append(n)
            if len(xy[::step]) >= 2:
                codes.append(int(a["agent_type"].iloc[0]))
                dist_m.append(float(np.hypot(*(xy[-1] - xy[0]))))
        if prof.generate.group_emission:
            sizes += _group_sizes(tracks, step)
    q = np.percentile(dist_m, [25, 50, 75])
    code_p = pd.Series(codes).value_counts(normalize=True)
    priors = {
        "life": [int(x) for x in life],
        "life0": [int(x) for x in (life0 or life)],
        "goaldist": {"p25_m": float(q[0]), "p50_m": float(q[1]), "p75_m": float(q[2])},
        "attr_dist": {"codes": {str(int(k)): float(v) for k, v in code_p.items()}},
        "pop": {str(int(k)): float(v) for k, v in pd.Series(pop).value_counts(normalize=True).sort_index().items()},
    }
    if prof.generate.group_emission:
        group_p = pd.Series(sizes).clip(upper=6).value_counts(normalize=True).sort_index()
        priors["group"] = {"size_p": {int(k): float(v) for k, v in group_p.items()}, "same_code_p": 1.0}
    os.makedirs(os.path.dirname(prior_path(fold, "life")), exist_ok=True)
    for kind, value in priors.items():
        json.dump(value, open(prior_path(fold, kind), "w"))


def load_priors(fold):
    return {kind: json.load(open(prior_path(fold, kind))) for kind in PRIOR_FILES if os.path.exists(prior_path(fold, kind))}


def load_population_prior(stem):
    """CrowdES crowd-size distribution of a test scene -> (counts, probabilities, CrowdES multiplier)."""
    d = json.load(open(os.path.join(DATA_DIR, "priors", "population", f"{stem.split('/')[0]}.json")))
    p = np.asarray(d["p"], float)
    return np.asarray(d["counts"], int), p / p.sum(), float(d["crowdes_multiplier"])


def load_entry_prior(fold):
    d = json.load(open(os.path.join(DATA_DIR, "priors", "entry", f"{fold}.json")))
    return np.asarray(d["origins"], np.float32), np.asarray(d["goals"], np.float32)


def prepare_fold(fold):
    prepare_priors(fold)

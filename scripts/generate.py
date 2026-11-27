"""python scripts/generate.py <fold> [--seed S | --weights path] [--trials N] [--row paper] [--scenes ...] [--tag T]

Writes outputs/generated/<fold>/<scene>-<tag>-trial<t>.pkl.gz and outputs/metrics/<fold>_<tag>_summary.json
(`__mean__` averages all scene x trial pairs).
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from acg.config import DEVICE, OUTPUT_DIR, WEIGHTS_DIR, profile_of  # noqa: E402
from acg.data import load_scenes  # noqa: E402
from acg.generate import Generator  # noqa: E402
from acg.metrics import compute_metrics  # noqa: E402
from acg.model import load_model  # noqa: E402


def average(dicts):
    acc = {}
    for d in dicts:
        for group, vals in d.items():
            for k, v in vals.items():
                acc.setdefault(group, {}).setdefault(k, []).append(float(v))
    return {g: {k: float(np.mean(v)) for k, v in kv.items()} for g, kv in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fold")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--trials", type=int, default=None)
    ap.add_argument("--row", default=None, choices=[None, "paper"])
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--no-metrics", action="store_true")
    a = ap.parse_args()

    prof = profile_of(a.fold, a.row)
    trials = a.trials or prof.generate.trials
    tag = a.tag or ("acg" + (f"_seed{a.seed}" if a.seed else "") + ("_paper" if a.row == "paper" else ""))
    weights = a.weights or os.path.join(WEIGHTS_DIR, f"acg_{a.fold}_seed{a.seed}.pth")
    model = load_model(weights, prof.model, DEVICE)
    gen_dir = os.path.join(OUTPUT_DIR, "generated", a.fold)
    met_dir = os.path.join(OUTPUT_DIR, "metrics")
    os.makedirs(gen_dir, exist_ok=True); os.makedirs(met_dir, exist_ok=True)

    results = {}
    for scene in load_scenes(a.fold, "test"):
        if a.scenes and scene.stem not in a.scenes:
            continue
        gen = Generator(model, a.fold, prof)
        name = scene.stem.replace("/", "_")
        print(f"[{a.fold}/{tag}] {scene.stem}: {scene.info['length']} frames, {scene.info['num_agents']} GT agents", flush=True)
        scores = []
        with torch.no_grad():
            for t in range(trials):
                gen.set_scene(scene)
                df = gen.generate(scene.info["length"], seed=t)
                df["scene"] = scene.stem
                df.to_pickle(os.path.join(gen_dir, f"{name}-{tag}-trial{t}.pkl.gz"))
                if not a.no_metrics:
                    scores.append(compute_metrics(df, scene.trajectory, scene.info, scene.H))
                print(f"  trial {t}: {df['agent_id'].nunique()} agents", flush=True)
        results[scene.stem] = scores
    if not a.no_metrics:
        summary = {s: {tag: average(r)} for s, r in results.items()}
        summary["__mean__"] = {tag: average([m for r in results.values() for m in r])}
        path = os.path.join(met_dir, f"{a.fold}_{tag}_summary.json")
        json.dump(summary, open(path, "w"), indent=2)
        print(json.dumps(summary["__mean__"][tag], indent=1))
        print("->", path)


if __name__ == "__main__":
    main()

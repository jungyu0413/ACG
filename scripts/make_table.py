"""python scripts/make_table.py [--tags acg ...] -> outputs/results_table.md

Rows are listed next to the reference numbers: CrowdES as published (ETH/UCY, SDD), the released CrowdES checkpoint
on the corrected GCS split (population multiplier 0.10), CrowdES retrained with attribute inputs (PANDA, 12-fold
mean). Col in %. Lower is better except Div.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from acg.config import OUTPUT_DIR, dataset_of  # noqa: E402
from acg.data import folds  # noqa: E402

KEYS = [("Scene-Level Realism Metrics", k) for k in ("Density", "Frequency", "Coverage", "Population")] + \
       [("Agent-Level Accuracy Metrics", k) for k in ("Kineamtics", "DTW", "Diversity", "Collision")]
NAMES = ["Dens", "Freq", "Cov", "Pop", "Kin", "DTW", "Div↑", "Col%"]
REFERENCE = {
    "eth": ("CrowdES (published)", [0.020, 0.020, 0.020, 0.208, 0.377, 1.649, 0.203, 0.697]),
    "hotel": ("CrowdES (published)", [0.013, 0.009, 0.009, 0.117, 0.336, 0.643, 0.242, 1.197]),
    "univ": ("CrowdES (published)", [0.347, 0.204, 0.204, 0.734, 0.420, 1.128, 0.340, 0.645]),
    "zara1": ("CrowdES (published)", [0.018, 0.017, 0.017, 0.254, 0.327, 0.675, 0.304, 1.018]),
    "zara2": ("CrowdES (published)", [0.009, 0.013, 0.013, 0.100, 0.227, 0.579, 0.355, 1.021]),
    "sdd": ("CrowdES (published)", [0.038, 0.033, 0.030, 0.463, 0.650, 6.352, 0.354, 0.411]),
    "gcs": ("CrowdES (corrected split, x0.10)", [0.332, 0.121, 0.121, 0.768, 1.192, 5.064, 0.343, 1.22]),
    "panda": ("CrowdES + attributes (12-fold mean)", [0.422, 0.090, 0.040, 0.154, 0.607, 37.48, 0.251, 2.5]),
}


def load(metrics_dir, fold, tag):
    p = os.path.join(metrics_dir, f"{fold}_{tag}_summary.json")
    if not os.path.exists(p):
        return None
    m = json.load(open(p))["__mean__"][tag]
    v = np.array([m[g][k] for g, k in KEYS])
    v[7] *= 100.0
    return v


def fmt(v):
    return " | ".join(f"{x:.2f}" if j == 5 and x >= 10 else f"{x:.3f}" for j, x in enumerate(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default=os.path.join(OUTPUT_DIR, "metrics"))
    ap.add_argument("--tags", nargs="+", default=["acg", "acg_seed1", "acg_seed2"])
    ap.add_argument("--out", default=os.path.join(OUTPUT_DIR, "results_table.md"))
    a = ap.parse_args()
    lines = ["| fold | row | " + " | ".join(NAMES) + " |", "|---|---|" + "---:|" * 8]
    groups = {}
    for fold in folds():
        ds = dataset_of(fold)
        rows = {tag: load(a.metrics, fold, tag) for tag in a.tags}
        if all(v is None for v in rows.values()):
            continue
        if ds == "panda":
            for tag, v in rows.items():
                if v is not None:
                    groups.setdefault(("panda", tag), []).append(v)
            continue
        label, ref = REFERENCE[fold]
        lines.append(f"| {fold} | {label} | {fmt(ref)} |")
        for tag, v in rows.items():
            if v is not None:
                lines.append(f"| {fold} | {tag} | {fmt(v)} |")
                if ds == "ethucy":
                    groups.setdefault(("ethucy", tag), []).append(v)
    eth_folds = [f for f in folds() if dataset_of(f) == "ethucy"]
    if any(k[0] == "ethucy" for k in groups):
        lines.append(f"| ETH/UCY AVG | CrowdES (published) | {fmt(np.mean([REFERENCE[f][1] for f in eth_folds], 0))} |")
        for (ds, tag), vs in groups.items():
            if ds == "ethucy" and len(vs) == len(eth_folds):
                lines.append(f"| ETH/UCY AVG | {tag} | {fmt(np.mean(vs, 0))} |")
    n_panda = len([f for f in folds() if dataset_of(f) == "panda"])
    if any(k[0] == "panda" for k in groups):
        label, ref = REFERENCE["panda"]
        lines.append(f"| PANDA (12 folds) | {label} | {fmt(ref)} |")
        for (ds, tag), vs in groups.items():
            if ds == "panda":
                lines.append(f"| PANDA ({len(vs)}/{n_panda} folds) | {tag} | {fmt(np.mean(vs, 0))} |")
    text = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out, "w").write(text)
    print(text)
    print("->", a.out)


if __name__ == "__main__":
    main()

"""CrowdES crowd-size prior of the ETH/UCY and SDD test scenes -> data/priors/population/<scene>.json.
    CROWDES_REPO=/path/to/Crowd-Behavior-Generation python tools/export_population_prior.py [folds]
p(count) = normalised, thresholded sigmoid of the released emitter_pre checkpoint on the scene segmentation; the
fold's population_multiplier is stored with it. Needs the CrowdES code and checkpoints.
"""
import json
import os
import sys

import numpy as np
import torch
import yaml
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")
REPO = os.environ.get("CROWDES_REPO", "/workspace/Crowd-Behavior-Generation")
sys.path.insert(0, REPO)
from CrowdES.emitter.emitter_pre_model import CrowdESEmitterPreModel  # noqa: E402


def main():
    folds = json.load(open(os.path.join(DATA, "folds.json")))
    names = sys.argv[1:] or [f for f in folds if folds[f]["dataset"] in ("ethucy", "sdd")]
    classes = json.load(open(os.path.join(REPO, "datasets", "segmentation_classes.json")))
    out_dir = os.path.join(DATA, "priors", "population")
    os.makedirs(out_dir, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for fold in names:
        cfg = yaml.safe_load(open(os.path.join(REPO, "configs", "model", f"CrowdES_{fold}.yaml")))["crowd_emitter"]["emitter_pre"]
        model = CrowdESEmitterPreModel.from_pretrained(os.path.join(REPO, "checkpoints", fold, "emitter_pre")).to(dev).eval()
        for stem in folds[fold]["test"]:
            seg = np.array(Image.open(os.path.join(DATA, "scenes", stem, "seg.png")).convert("L"))
            x = torch.FloatTensor(np.eye(len(classes))[seg]).permute(2, 0, 1)[None]        # [1, 7, H, W] one-hot
            x = torch.nn.functional.interpolate(x, size=tuple(cfg["image_size"]), mode="bilinear", align_corners=False)
            with torch.no_grad():
                p = model(x.to(dev)).logits_unit.squeeze(0).sigmoid().cpu().numpy()
            p[p < cfg["population_probability_threshold"]] = 0
            p = p / p.sum()
            json.dump({"fold": fold, "scene": stem, "counts": list(range(len(p))), "p": [float(v) for v in p],
                       "crowdes_multiplier": float(cfg["population_multiplier"])},
                      open(os.path.join(out_dir, f"{stem}.json"), "w"))
            print(f"{fold}/{stem}: mean count {float((np.arange(len(p)) * p).sum()):.1f} x {cfg['population_multiplier']}", flush=True)
        del model


if __name__ == "__main__":
    main()

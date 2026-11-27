"""Attribute codes: code = life_stage + 2 * mobility + 6 * constraint + 18 * attention.

One-hot (11): slot 0 child flag, 1-3 mobility, 4-6 constraint, 7-9 attention, 10 unused.
"""
import json

import numpy as np

ATTR_DIM = 11
GROUPS = {"life_stage": ["adult", "child"],
          "mobility": ["on_foot", "personal_mobility", "dependent_mobility"],
          "constraint": ["none", "dependent_care", "object_constrained"],
          "attention": ["none", "device", "non_device"]}
LOGIT_OFFSETS = np.array([0, 2, 5, 8])


def decode(code):
    code = int(code)
    return code % 2, (code // 2) % 3, (code // 6) % 3, (code // 18) % 3


def onehot(code):
    ls, mob, con, att = decode(code)
    v = np.zeros(ATTR_DIM, np.float32)
    v[0] = ls
    v[1 + mob] = 1
    v[4 + con] = 1
    v[7 + att] = 1
    return v


def onehot_batch(codes):
    return np.stack([onehot(c) for c in codes], 0) if len(codes) else np.zeros((0, ATTR_DIM), np.float32)


def group_indices(onehots):
    oh = np.asarray(onehots)
    return np.stack([oh[:, 0].astype(int), oh[:, 1:4].argmax(1), oh[:, 4:7].argmax(1), oh[:, 7:10].argmax(1)], 1)


def load_code_distribution(path):
    d = json.load(open(path))
    codes = np.array([int(k) for k in d["codes"]])
    p = np.array([d["codes"][k] for k in d["codes"]], float)
    return codes, p / p.sum()

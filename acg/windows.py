"""Training data: per-scene agent tracks on the grid, emitter targets and random multi-agent windows."""
import numpy as np
import torch

from .attributes import onehot_batch
from .config import DEVICE, GRID
from .data import load_scenes
from .geometry import heatmaps, image2grid, metres_per_grid_px, scene_raster

RARE_SLOTS = (0, 2, 3, 5, 6, 8, 9)   # anything but adult / on foot / no constraint / no device


def build_training_scenes(fold, prof):
    """stem -> {agents: {id: (frames, grid_xy, attr_onehot)}, raster, frame range, spawn target [13, G, G],
    attribute-head samples (origin, one-hot)}."""
    cfg, mcfg = prof.train, prof.model
    scenes = {}
    for sc in load_scenes(fold, "train"):
        stem = sc.stem
        raster = scene_raster(stem)[0]
        stationary = cfg.stationary_step_m / (cfg.scene_scale or metres_per_grid_px(stem))
        agents = {}
        for aid, g in sc.trajectory.groupby("agent_id"):
            g = g.sort_values("frame")
            frames = g["frame"].to_numpy().astype(np.int64)
            grid = image2grid(g[["x", "y"]].to_numpy().astype(np.float64), stem).astype(np.float32)
            agents[int(aid)] = (frames, grid, onehot_batch([int(g["agent_type"].iloc[0])])[0])
        fmin, fmax = int(sc.trajectory["frame"].min()), int(sc.trajectory["frame"].max())
        d = dict(agents=agents, raster=raster.to(DEVICE), fmin=fmin, fmax=fmax)
        d["attr_origins"] = torch.tensor(np.stack([g_[0] for _, g_, _ in agents.values()]), dtype=torch.float32, device=DEVICE)
        d["attr_onehot"] = torch.tensor(np.stack([a_[:mcfg.attr_dim] for _, _, a_ in agents.values()]), dtype=torch.float32, device=DEVICE)
        # spawn target: ch v = origins of agents with attribute slot v, ch 11 = all origins (agents present at the
        # first frame did not appear), ch 12 = standing positions
        tgt = torch.zeros(mcfg.spawn_channels, GRID, GRID, device=DEVICE)
        standing = []
        for fr_, g_, a_ in agents.values():
            if len(g_) > 2:
                standing.append(g_[1:][np.hypot(*(np.diff(g_, axis=0).T)) < stationary][::2])
            if fr_[0] <= fmin:
                continue
            hm = heatmaps(torch.tensor(g_[0], device=DEVICE))
            tgt[11] = torch.maximum(tgt[11], hm)
            for v in np.nonzero(a_[:mcfg.attr_dim] > 0)[0]:
                tgt[int(v)] = torch.maximum(tgt[int(v)], hm)
        if standing:
            P = torch.tensor(np.concatenate(standing, 0), device=DEVICE, dtype=torch.float32)
            for i in range(0, len(P), 256):
                tgt[12] = torch.maximum(tgt[12], heatmaps(P[i:i + 256]).max(0).values)
        d["spawn_target"] = tgt
        scenes[stem] = d
        print(f"  {stem}: {len(agents)} agents, frames [{fmin}, {fmax}]", flush=True)
    return scenes


def _position_at(frames, grid, frame):
    i = np.searchsorted(frames, frame)
    return grid[i] if i < len(frames) and frames[i] == frame else None


def sample_window(scene, rng, cfg, step):
    """Agents present at a random anchor frame, with `history` past and `horizon` future positions at the simulator
    rate. History before an agent's first frame repeats its first position; future after its last frame repeats
    the last position and is masked out. The goal is the agent's final position."""
    agents, fmin, fmax = scene["agents"], scene["fmin"], scene["fmax"]
    lo, hi = fmin + (cfg.history - 1) * step, fmax - 1
    if hi <= lo:
        return None
    f0 = int(rng.integers(lo, hi + 1))
    if cfg.attr_balance > 0 and rng.random() < cfg.attr_balance:
        if "rare" not in scene:
            scene["rare"] = [aid for aid, (_, _, a_) in agents.items() if any(a_[k] > 0 for k in RARE_SLOTS)]
        if scene["rare"]:
            fr_ = agents[scene["rare"][int(rng.integers(0, len(scene["rare"])))]][0]
            lo_a, hi_a = max(lo, int(fr_[0])), min(hi, int(fr_[-1]) - 1)
            if hi_a > lo_a:
                f0 = int(rng.integers(lo_a, hi_a + 1))
    hist_frames = [f0 - (cfg.history - 1 - k) * step for k in range(cfg.history)]
    fut_frames = [f0 + (k + 1) * step for k in range(cfg.horizon)]
    H_l, F_l, M_l, G_l, A_l = [], [], [], [], []
    for _, (fr, grid, attr) in agents.items():
        p0 = _position_at(fr, grid, f0)
        if p0 is None:
            continue
        hist = np.zeros((cfg.history, 2), np.float32)
        for k, hf in enumerate(hist_frames):
            p = _position_at(fr, grid, hf)
            if p is None:
                p = _position_at(fr, grid, max(hf, fr[0])) if hf < fr[0] else p0
                if p is None:
                    p = grid[0]
            hist[k] = p
        fut = np.zeros((cfg.horizon, 2), np.float32)
        mask = np.zeros(cfg.horizon, np.float32)
        last = p0.copy()
        for k, ff in enumerate(fut_frames):
            p = _position_at(fr, grid, ff)
            if p is not None:
                fut[k] = p; mask[k] = 1.0; last = p
            else:
                fut[k] = last
        H_l.append(hist); F_l.append(fut); M_l.append(mask); G_l.append(grid[-1].astype(np.float32)); A_l.append(attr)
    if cfg.max_agents and len(H_l) > cfg.max_agents:
        keep = sorted(rng.choice(len(H_l), cfg.max_agents, replace=False).tolist())
        H_l, F_l, M_l, G_l, A_l = [[L[i] for i in keep] for L in (H_l, F_l, M_l, G_l, A_l)]
    if len(H_l) < cfg.min_agents:
        return None
    return tuple(torch.from_numpy(np.stack(L)).to(DEVICE) for L in (H_l, F_l, M_l, G_l, A_l))


def sample_windows(scenes, rng, n, cfg, step):
    names = list(scenes.keys())
    out, tries = [], 0
    while len(out) < n and tries < n * 20:
        tries += 1
        s = names[int(rng.integers(0, len(names)))]
        w = sample_window(scenes[s], rng, cfg, step)
        if w is not None:
            out.append((s,) + w)
    return out


def collate(scenes, windows):
    """Concatenate windows -> raster, hist, fut, mask, goal, attr [A, ...] and groups [W, 2] (agent ranges)."""
    rasters, parts, groups, off = [], [[] for _ in range(5)], [], 0
    for s, *tensors in windows:
        n = tensors[0].shape[0]
        rasters.append(scenes[s]["raster"][None].expand(n, -1, -1, -1))
        for p, t in zip(parts, tensors):
            p.append(t)
        groups.append((off, off + n)); off += n
    hist, fut, mask, goal, attr = (torch.cat(p, 0) for p in parts)
    return torch.cat(rasters, 0), hist, fut, mask, goal, attr, torch.tensor(groups, device=DEVICE)

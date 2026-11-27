"""Coordinate frames (image px, world metres, network grid), scene rasters and heatmaps."""
import functools
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.spatial import cKDTree

from .config import DATA_DIR, DEVICE, GRID, OBS, SIGMA

NUM_SEG_CLASSES = 8


def image2world(coord, H):
    shape = coord.shape
    coord = coord.reshape(-1, 2)
    x, y = coord[..., 0], coord[..., 1]
    world = (H @ np.stack([x, y, np.ones_like(x)], axis=-1).T).T
    world = world / world[..., [2]]
    return world[..., :2].reshape(shape)


def world2image(coord, H):
    shape = coord.shape
    coord = coord.reshape(-1, 2)
    x, y = coord[..., 0], coord[..., 1]
    image = (np.linalg.inv(H) @ np.stack([x, y, np.ones_like(x)], axis=-1).T).T
    image = image / image[..., [2]]
    return image[..., :2].reshape(shape)


def scene_dir(stem):
    return os.path.join(DATA_DIR, "scenes", stem.split("/")[0])


@functools.lru_cache(maxsize=None)
def load_homography(stem):
    return np.loadtxt(os.path.join(scene_dir(stem), "H.txt"))


@functools.lru_cache(maxsize=None)
def load_segmentation(stem):
    return np.array(Image.open(os.path.join(scene_dir(stem), "seg.png")).convert("L"))


@functools.lru_cache(maxsize=None)
def load_walkable(stem):
    return np.array(Image.open(os.path.join(scene_dir(stem), "walkable.png")).convert("L")) > 0


def seg_to_onehot(seg):
    """[H, W] class ids -> [8, H, W] one-hot (ids outside 0..7 -> zero vector)."""
    seg = np.asarray(seg)
    eye = np.eye(NUM_SEG_CLASSES + 1, dtype=np.float32)
    return eye[np.where(seg < NUM_SEG_CLASSES, seg, NUM_SEG_CLASSES)][..., :NUM_SEG_CLASSES].transpose(2, 0, 1)


_raster_cache = {}


def scene_raster(stem):
    """[9, GRID, GRID] network input (walkable + class one-hot) and the native map size (H, W)."""
    stem = stem.split("/")[0]
    if stem not in _raster_cache:
        walk = load_walkable(stem).astype(np.float32)
        chan = np.concatenate([walk[None], seg_to_onehot(load_segmentation(stem))], 0)
        H, W = walk.shape
        t = F.interpolate(torch.from_numpy(chan).to(DEVICE)[None], size=(GRID, GRID), mode="bilinear", align_corners=False)[0]
        _raster_cache[stem] = (t, H, W)
    return _raster_cache[stem]


def metres_per_image_px(stem):
    """Homography scale at the image centre (geometric mean of both axes)."""
    _, H, W = scene_raster(stem)
    Hm = load_homography(stem)
    c = np.array([[W / 2.0, H / 2.0]])
    J = np.stack([image2world(c + d, Hm)[0] - image2world(c, Hm)[0] for d in ([1.0, 0.0], [0.0, 1.0])], 1)
    return float(np.sqrt(abs(np.linalg.det(J))))


def metres_per_grid_px(stem):
    _, H, W = scene_raster(stem)
    return metres_per_image_px(stem) * float(np.sqrt(W * H) / GRID)


def image2grid(px, stem):
    _, H, W = scene_raster(stem)
    g = np.asarray(px, dtype=float).copy()
    g[..., 0] = g[..., 0] * (GRID / W)
    g[..., 1] = g[..., 1] * (GRID / H)
    return g


def grid2image(g, stem):
    _, H, W = scene_raster(stem)
    px = np.asarray(g, dtype=float).copy()
    px[..., 0] = px[..., 0] * (W / GRID)
    px[..., 1] = px[..., 1] * (H / GRID)
    return px


def world2grid(coords, stem):
    return image2grid(world2image(np.asarray(coords, dtype=float), load_homography(stem)), stem)


def grid2world(g, stem):
    return image2world(grid2image(g, stem), load_homography(stem))


class WalkableSnapper:
    """Moves image coordinates to the nearest walkable pixel."""

    def __init__(self, walkable):
        ys, xs = np.nonzero(walkable)
        self.tree = cKDTree(np.column_stack((xs, ys)))

    def __call__(self, coords):
        _, idx = self.tree.query(coords)
        return self.tree.data[idx].astype(int)


def heatmaps(grid_xy, sigma=SIGMA, grid=GRID):
    """Grid coordinates [..., 2] -> Gaussian heatmaps [..., grid, grid], peak 1."""
    ys = torch.arange(grid, device=grid_xy.device).float()
    yy, xx = torch.meshgrid(ys, ys, indexing="ij")
    gx = grid_xy[..., 0, None, None]
    gy = grid_xy[..., 1, None, None]
    return torch.exp(-((xx - gx) ** 2 + (yy - gy) ** 2) / (2 * sigma ** 2))


def soft_argmax(hm):
    """Heatmap logits [..., G, G] -> expected grid coordinates [..., 2]."""
    G = hm.shape[-1]
    p = F.softmax(hm.reshape(*hm.shape[:-2], G * G), -1).reshape(*hm.shape)
    ax = torch.arange(G, device=hm.device).float()
    return torch.stack([(p.sum(-2) * ax).sum(-1), (p.sum(-1) * ax).sum(-1)], -1)


def render_context(pos_hist, groups):
    """Encoder inputs at the last step of pos_hist [A, T, 2] (grid): ego heatmap of the last OBS positions,
    neighbour heatmap (current positions of the other agents of the same group), position and velocity history
    (both / GRID). groups [W, 2] = (start, end) agent ranges that share a scene."""
    t = pos_hist.shape[1] - 1
    win = pos_hist[:, max(0, t - OBS + 1):t + 1]
    ego = heatmaps(win).sum(1).clamp(0, 1)
    cur = pos_hist[:, t]
    cur_hm = heatmaps(cur)
    nbr = torch.zeros_like(ego)
    for s, e in groups.tolist():
        seg = cur_hm[s:e]
        nbr[s:e] = (seg.sum(0, keepdim=True) - seg).clamp(0, 1)
    if win.shape[1] >= 2:
        v = win[:, 1:] - win[:, :-1]
        v = torch.cat([torch.zeros_like(v[:, :1]), v], 1) / GRID
    else:
        v = torch.zeros(pos_hist.shape[0], 1, 2, device=pos_hist.device)
    return ego, nbr, cur / GRID, v

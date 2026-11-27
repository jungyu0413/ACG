"""Scenario generation.

Per window: the emitter tops the scene up to a target population (origins from the spawn map, attributes from the
attribute head or the training distribution); the simulator then advances all agents in chunks: agents without a
goal sample one from the goal head (walkable cells beyond a sampled origin-goal distance, biased towards the
standing-occupancy map) and a lifetime from the training track lengths; the velocity head is rolled out with an
optional per-agent latent, AR(1) noise, velocity inertia, a speed multiplier and a collision-resolution step.
Agents leave at the map border or at the end of their lifetime and get a new goal on arrival. Tracks are
Kalman-smoothed and interpolated to the dataset frame rate. The trial index seeds every random draw.
"""
import random

import numpy as np
import pandas as pd
import torch
from pykalman import KalmanFilter

from .attributes import LOGIT_OFFSETS, group_indices, load_code_distribution, onehot_batch
from .config import DEVICE, GRID
from .data import load_entry_prior, load_population_prior, load_priors, prior_path
from .geometry import (WalkableSnapper, grid2image, grid2world, image2world, load_walkable, metres_per_grid_px,
                       metres_per_image_px, render_context, scene_raster, world2grid, world2image)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def kalman_smooth(traj):
    """Constant-acceleration Kalman smoother (one EM iteration per axis), as in CrowdES."""
    if traj.shape[0] == 1:
        return traj
    A = np.array([[1, 1, 1], [0, 1, 1], [0, 0, 1]])
    Q = np.array([[1 / 20, 1 / 8, 1 / 6], [1 / 8., 1 / 3, 1 / 2], [1 / 6., 1 / 2, 1 / 1]]) * 0.5
    out = []
    for d in range(2):
        kf = KalmanFilter(transition_matrices=A, observation_matrices=np.array([[1, 0, 0]]),
                          transition_covariance=Q, observation_covariance=np.array([[1]]))
        means, _ = kf.em(traj[:, d], n_iter=1).smooth(traj[:, d])
        out.append(means)
    return np.stack(out)[:, :, 0].T


class Generator:
    def __init__(self, model, fold, prof):
        self.m, self.fold, self.cfg, self.ds = model, fold, prof.generate, prof.dataset
        cfg = self.cfg
        self.window = cfg.window_seconds * self.ds.sim_fps
        priors = load_priors(fold)
        self.codes, self.code_p = load_code_distribution(prior_path(fold, "attr_dist"))
        self.life, self.life0 = priors["life"], priors["life0"]
        self.life_cap = int(np.quantile(self.life, cfg.lifetime_cap_quantile)) if cfg.lifetime_cap_quantile else None
        gd = priors["goaldist"]
        self.goal_quantiles_m = np.array([gd["p25_m"], gd["p50_m"], gd["p75_m"]])
        if cfg.population_source == "histogram":
            pop = priors["pop"]
            self.hist_values = np.array([int(k) for k in pop])
            self.hist_p = np.array([pop[k] for k in pop]); self.hist_p /= self.hist_p.sum()
        if cfg.group_emission:
            grp = priors["group"]
            ks = np.array([int(k) for k in grp["size_p"]])
            pk = np.array([grp["size_p"][str(k)] for k in ks], float)
            self.group_sizes, self.group_p, self.group_same_code = ks, pk / pk.sum(), float(grp["same_code_p"])
        self.entry = load_entry_prior(fold) if cfg.entry_exit_prior else None
        self.dropped = 0

    def set_scene(self, scene):
        cfg = self.cfg
        self.stem, self.H = scene.stem, scene.H
        walk = load_walkable(scene.stem)
        self.size = walk.shape                          # (height, width)
        self.snap = WalkableSnapper(walk)
        self.mpp = cfg.scene_scale or metres_per_grid_px(scene.stem)
        self.m_per_px = metres_per_image_px(scene.stem)
        self.coll_r = cfg.collision_radius_px if cfg.collision_radius_px > 0 else round(cfg.collision_radius_m / self.mpp, 2)
        self.goal_quantiles = (self.goal_quantiles_m / self.mpp).astype(np.float32)
        if cfg.population_source == "crowdes":
            self.pop_values, self.pop_p, self.pop_mult = load_population_prior(scene.stem)
        else:
            self.pop_values, self.pop_p, self.pop_mult = self.hist_values, self.hist_p, 1.0
        self._spawn = None

    def spawn_maps(self):
        """Walkable-masked spawn probabilities [13, G, G] and the empty-scene encoding."""
        if self._spawn is None:
            scene = scene_raster(self.stem)[0][None].to(DEVICE)
            zero = torch.zeros(1, GRID, GRID, device=DEVICE)
            b, skips = self.m.encode(scene, zero, zero, torch.zeros(1, self.m.social.out[-1].out_features, device=DEVICE))
            prob = torch.sigmoid(self.m.spawn_logits(b, skips)[0]) * scene[0, 0]
            self._spawn = (prob, b, skips)
        return self._spawn

    # ------------------------------------------------------------------ emitter
    def sample_origins_and_codes(self, n):
        """Origins from the all-origin map, attributes from the attribute head at each origin (restricted to the
        training codes); or codes from the training distribution and origins from the code's attribute-value maps."""
        prob, b, skips = self.spawn_maps()
        H, W = self.size
        if self.cfg.attribute_source == "prior":
            codes = np.random.choice(self.codes, size=n, p=self.code_p)
            out = np.zeros((n, 2), np.float32)
            for i, oh in enumerate(onehot_batch(list(codes))):
                vals = np.nonzero(oh > 0)[0]
                d = (prob[11] * prob[vals].mean(0)).clamp(min=1e-8).reshape(-1) ** (1.0 / self.cfg.spawn_temperature)
                idx = int(torch.multinomial(d / d.sum(), 1).item())
                out[i] = [(idx % GRID) * (W / GRID), (idx // GRID) * (H / GRID)]
        else:
            d = prob[11].clamp(min=1e-8).reshape(-1) ** (1.0 / self.cfg.spawn_temperature)
            idx = torch.multinomial(d / d.sum(), n, replacement=True)
            gpos = torch.stack([(idx % GRID).float(), (idx // GRID).float()], -1)
            lg = self.m.attribute_logits(b.expand(n, -1, -1, -1), skips[0].expand(n, -1, -1, -1), gpos)
            lp = self.m.attr_head.group_log_probs(lg).float().cpu().numpy()
            score = lp[:, LOGIT_OFFSETS[None, :] + group_indices(onehot_batch(list(self.codes)))].sum(-1)
            pc = np.exp(score - score.max(1, keepdims=True)); pc /= pc.sum(1, keepdims=True)
            codes = np.array([np.random.choice(self.codes, p=pc[i]) for i in range(n)])
            out = grid2image(gpos.cpu().numpy(), self.stem).astype(np.float32)
        return np.asarray(self.snap(out), np.float32).reshape(n, 2), codes

    def emit(self):
        cfg = self.cfg
        first = self.frame < 0
        target = int(np.random.choice(self.pop_values, p=self.pop_p) * self.pop_mult * cfg.population_multiplier)
        dropped = 0 if first else self.dropped
        cap = cfg.max_emit_per_window if not first else max(cfg.max_emit_per_window, target)
        n = min(target - len(self.active) + dropped, cap)
        if n <= 0:
            self.added = 0
            return
        origins, codes = self.sample_origins_and_codes(n)
        entry_idx = None
        if self.entry is not None:
            entry_idx = np.random.randint(0, len(self.entry[0]), n)
            origins = self.entry[0][entry_idx] + np.random.normal(0, cfg.entry_jitter_px, (n, 2)).astype(np.float32)
            origins = np.asarray(self.snap(origins), np.float32).reshape(n, 2)
        leader_of = {}
        if cfg.group_emission:   # co-moving groups share origin (+ jitter), entry time and (with prob.) the code
            j = 0
            while j < n:
                k = int(np.random.choice(self.group_sizes, p=self.group_p))
                k = max(1, min(k, n - j))
                for m in range(1, k):
                    leader_of[j + m] = j
                    if np.random.rand() < self.group_same_code:
                        codes[j + m] = codes[j]
                    origins[j + m] = origins[j] + np.random.normal(0, cfg.group_jitter_m / max(self.m_per_px, 1e-6), 2)
                if k > 1:
                    origins[j + 1:j + k] = np.asarray(self.snap(origins[j + 1:j + k]), np.float32).reshape(k - 1, 2)
                j += k
        ids = {}
        for j in range(n):
            self.last_id += 1
            ids[j] = aid = self.last_id
            self.new.append(aid)
            p = {"agent_type": int(codes[j]), "frame_origin": 0 if first else int(np.random.randint(0, self.window)),
                 "origin_xy": origins[j], "goal_xy": origins[j].copy(), "preroll": first}
            if j in leader_of:
                p["frame_origin"], p["leader"] = self.agents[ids[leader_of[j]]]["frame_origin"], ids[leader_of[j]]
            if entry_idx is not None:
                p["exit_xy"] = self.entry[1][entry_idx[j]]
            p["frame_global"] = self.frame + p["frame_origin"]
            self.agents[aid] = p
            self.traj[aid] = np.empty((0, 2))
        self.added = n

    # ------------------------------------------------------------------ goals
    def attr_onehots(self):
        return torch.tensor(onehot_batch([int(self.agents[a]["agent_type"]) for a in self.active]), dtype=torch.float32, device=DEVICE)

    def agent_latents(self):
        """Per-agent latent [B, D] and speed multiplier [B] of the active agents (drawn once per agent)."""
        cfg = self.cfg
        if cfg.latent_sigma <= 0 and cfg.speed_heterogeneity <= 0:
            return None, None
        for aid in self.active:
            p = self.agents[aid]
            if "latent" not in p:
                p["latent"] = np.random.randn(self.m.step_feature_dim).astype(np.float32)
                p["speed_mult"] = float(np.exp(np.random.randn() * cfg.speed_heterogeneity)) if cfg.speed_heterogeneity > 0 else 1.0
        z = torch.tensor(np.stack([self.agents[a]["latent"] for a in self.active]), device=DEVICE) * cfg.latent_sigma if cfg.latent_sigma > 0 else None
        het = torch.tensor([self.agents[a]["speed_mult"] for a in self.active], device=DEVICE) if cfg.speed_heterogeneity > 0 else None
        return z, het

    def assign_goals(self, hist_img):
        """Goal head for every active agent; agents with a goal keep it. A first goal also fixes the lifetime."""
        cfg, m = self.cfg, self.m
        B = hist_img.shape[0]
        ea = m.embed_attributes(self.attr_onehots())
        grid_hist = torch.as_tensor(world2grid(image2world(np.asarray(hist_img, np.float32), self.H), self.stem),
                                    dtype=torch.float32, device=DEVICE)
        scene = scene_raster(self.stem)[0][None].expand(B, -1, -1, -1).to(DEVICE)
        groups = torch.tensor([[0, B]], device=DEVICE)
        ego, nbr, pos, vel = render_context(grid_hist, groups)
        soc = m.social(pos, vel, torch.zeros(B, B, device=DEVICE), ea)
        b, skips = m.encode(scene, ego, nbr, soc, cfg.encode_chunk)
        gl = m.goal_logits(b, skips, ea)
        gl = gl + (scene[:, 0] - 1.0) * cfg.goal_walk_penalty
        yy, xx = torch.meshgrid(torch.arange(GRID, device=DEVICE), torch.arange(GRID, device=DEVICE), indexing="ij")
        dist = (torch.stack([xx, yy], -1).float()[None] - (pos * GRID)[:, None, None, :]).norm(dim=-1)
        qs = np.random.choice(self.goal_quantiles, B)
        min_dist = torch.as_tensor(qs * np.random.uniform(0.8, 1.2, B) * cfg.goal_dist_scale,
                                   dtype=torch.float32, device=DEVICE).clamp(min=cfg.goal_min_dist_px)[:, None, None]
        gl = gl + torch.where(dist < min_dist, torch.full_like(gl, -1e4), torch.zeros_like(gl))
        occ = self.spawn_maps()[0][cfg.occupancy_channel]
        gl = gl + cfg.goal_occupancy_weight * torch.log(occ / occ.max().clamp(min=1e-8) + 1e-3)[None]
        p = torch.softmax((gl / cfg.goal_temperature).reshape(B, -1), -1)
        idx = torch.multinomial(p, 1)[:, 0]
        goal_grid = torch.stack([(idx % GRID).float(), (idx // GRID).float()], -1)
        goal_world = grid2world(goal_grid.cpu().numpy()[:, None, :], self.stem)[:, 0]
        goal_img = np.asarray(world2image(np.asarray(goal_world, np.float32), self.H), np.float32).reshape(B, 2)
        out = np.zeros((B, 2), np.float32)
        for i, aid in enumerate(self.active):
            p = self.agents[aid]
            if not p.get("has_goal", False):
                p["goal_xy"] = goal_img[i].copy()
                p["has_goal"] = True
                if "exit_xy" in p:
                    p["goal_xy"] = p["exit_xy"] + np.random.normal(0, cfg.entry_jitter_px, 2).astype(np.float32)
                if "max_life" not in p:
                    life = int(np.random.choice(self.life))
                    if p.get("preroll"):   # present at the start: remaining time from the first-frame cohort
                        life = int(np.random.choice(self.life0)) + self.window
                    if self.life_cap is not None:
                        life = min(life, self.life_cap)
                    p["max_life"] = life
                leader = self.agents.get(p.get("leader", -1))
                if leader is not None and leader.get("has_goal") and p["leader"] in self.active:
                    jitter = np.random.normal(0, cfg.group_goal_jitter_m / max(self.m_per_px, 1e-6), 2).astype(np.float32)
                    p["goal_xy"] = leader["goal_xy"] + jitter
                    if "max_life" in leader:
                        p["max_life"] = leader["max_life"]
            out[i] = p["goal_xy"]
        return out

    # ------------------------------------------------------------------ simulator
    def resolve_collisions(self, nxt):
        if self.coll_r <= 0 or nxt.shape[0] < 2:
            return nxt
        for _ in range(self.cfg.collision_iters):
            rel = nxt[:, None, :] - nxt[None, :, :]
            d = rel.norm(dim=-1)
            close = (d < self.coll_r) & (d > 1e-6)
            push = (rel / (d[..., None] + 1e-6) * (self.coll_r - d).clamp(min=0)[..., None] * close[..., None]).sum(1)
            nxt = (nxt + self.cfg.collision_push * push).clamp(0, GRID - 1)
        return nxt

    def rollout(self, hist_world, goal_world, steps, latent, speed_mult):
        """Velocity-head rollout of all active agents -> [B, steps, 2] metres."""
        cfg, m = self.cfg, self.m
        hist_world, goal_world = np.asarray(hist_world, np.float32), np.asarray(goal_world, np.float32)
        B = hist_world.shape[0]
        ea = m.embed_attributes(self.attr_onehots())
        hist = torch.as_tensor(world2grid(hist_world, self.stem), dtype=torch.float32, device=DEVICE)
        goal = torch.as_tensor(world2grid(goal_world[:, None, :], self.stem)[:, 0], dtype=torch.float32, device=DEVICE)
        scene = scene_raster(self.stem)[0][None].expand(B, -1, -1, -1).to(DEVICE)
        groups = torch.tensor([[0, B]], device=DEVICE)
        mask = torch.zeros(B, B, device=DEVICE)
        noise, inertia, preds = None, None, []
        for _ in range(steps):
            ego, nbr, pos, vel = render_context(hist, groups)
            soc = m.social(pos, vel, mask, ea)
            cur = hist[:, -1]
            cur_vel = hist[:, -1] - hist[:, -2] if hist.shape[1] > 1 else torch.zeros_like(cur)
            _, skips = m.encode(scene, ego, nbr, soc, cfg.encode_chunk)
            delta = m.step(skips, cur, goal, cur_vel, soc, ea, latent) * cfg.velocity_scale
            if cfg.velocity_noise > 0:
                eps = torch.randn_like(delta)
                rho = cfg.velocity_noise_rho
                noise = eps if noise is None else rho * noise + (1 - rho ** 2) ** 0.5 * eps
                delta = delta + cfg.velocity_noise * noise
            if cfg.velocity_inertia > 0:
                if inertia is None:
                    inertia = torch.where(cur_vel.norm(dim=-1, keepdim=True) > 1e-6, cur_vel, delta)
                inertia = cfg.velocity_inertia * inertia + (1 - cfg.velocity_inertia) * delta
                delta = inertia
            if speed_mult is not None:
                delta = delta * speed_mult[:, None]
            nxt = self.resolve_collisions((cur + delta).clamp(0, GRID - 1))
            preds.append(nxt)
            hist = torch.cat([hist, nxt[:, None]], 1)
        return grid2world(torch.stack(preds, 1).cpu().numpy(), self.stem).astype(np.float32)

    def simulate_window(self):
        cfg = self.cfg
        T_hist, T_fut = cfg.history, cfg.future
        before = len(self.active)
        for chunk in range(self.window // T_fut):
            for aid in self.new:
                if T_fut * chunk <= self.agents[aid]["frame_origin"] < T_fut * (chunk + 1):
                    self.active.append(aid)
            if not self.active:
                continue
            B = len(self.active)
            hist = np.zeros((B, T_hist, 2))
            for i, aid in enumerate(self.active):
                tr = self.traj[aid]
                if len(tr) < T_hist:
                    hist[i] = np.array(self.agents[aid]["origin_xy"])[None, :]
                    if len(tr) > 0:
                        hist[i, -len(tr):] = np.array(tr)
                else:
                    hist[i] = np.array(tr[-T_hist:])
            latent, speed_mult = self.agent_latents()
            goal = self.assign_goals(hist)
            hist_w, goal_w = image2world(hist, self.H), image2world(goal, self.H)
            fut_w = self.rollout(hist_w, goal_w, T_fut, latent, speed_mult)
            fut = self.snap(world2image(fut_w, self.H))
            for i, aid in list(enumerate(self.active)):
                if aid in self.new:
                    start = min(max(0, self.agents[aid]["frame_origin"] - T_fut * chunk), T_fut - 1)
                    tr = fut[i, :T_fut - start]
                    self.new.remove(aid)
                else:
                    tr = np.concatenate([self.traj[aid], fut[i]], axis=0)
                outside = ((fut[i] <= 0) | (fut[i] >= np.array(self.size)[[1, 0]] - 1)).any(axis=1)
                arrived = (np.linalg.norm(fut_w[i] - goal_w[i][None, :], axis=1) < cfg.arrival_m).any()
                if outside.any():
                    tr = tr[:-(T_fut - np.where(outside)[0][0])]
                    self.active.remove(aid)
                elif arrived:
                    self.agents[aid]["has_goal"] = False
                life = self.agents[aid].get("max_life", 0)
                if aid in self.active and life and len(tr) >= life:
                    tr = tr[:life]
                    self.active.remove(aid)
                self.traj[aid] = tr
        self.dropped = self.added + before - len(self.active)

    def generate(self, length, seed):
        """Scenario of `length` dataset frames -> DataFrame (agent_id, agent_type, frame, x, y), image px."""
        set_seed(seed)
        step = self.ds.step
        self.last_id = -1
        self.agents, self.traj, self.active, self.new = {}, {}, [], []
        n_sim = int(np.ceil(length / step))
        for self.frame in range(-self.window, n_sim, self.window):
            self.emit()
            self.simulate_window()
        for aid, tr in self.traj.items():
            if len(tr) >= 2:
                w = image2world(tr, self.H)
                self.traj[aid] = world2image(kalman_smooth(w - w[0]) + w[0], self.H)
        rows = []
        for aid, p in self.agents.items():
            tr = self.traj[aid]
            if len(tr) < 2:
                continue
            f0 = p["frame_global"]
            frames = np.arange(f0, len(tr) + f0) * step
            dense = np.arange(f0 * step, (len(tr) + f0) * step)
            x, y = np.interp(dense, frames, tr[:, 0]), np.interp(dense, frames, tr[:, 1])
            for i in range(len(dense)):
                if 0 <= dense[i] < length:
                    rows.append([aid, p["agent_type"], dense[i], x[i], y[i]])
        df = pd.DataFrame(rows, columns=["agent_id", "agent_type", "frame", "x", "y"])
        df = df.astype({"agent_id": int, "agent_type": int, "frame": int, "x": float, "y": float})
        df["agent_id"] = df["agent_id"].map({a: i for i, a in enumerate(df["agent_id"].unique())})
        return df

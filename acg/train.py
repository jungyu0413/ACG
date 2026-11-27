"""Single-stage training of all heads.

Simulator: free-running rollout of `bptt` steps towards GT waypoints, scored with position error + soft-DTW + LSGAN on
velocity sequences, plus a calibrated speed/acceleration term and/or per-attribute speed and neighbour-distance
quantile terms. Emitter: spawn maps (BCE) and attribute head (cross-entropy) from the empty-scene encoding.
Goal head: destination heatmap (BCE + soft-argmax distance) from the window-start context.
"""
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import DEVICE, GRID
from .geometry import heatmaps, render_context, soft_argmax
from .model import ACG
from .windows import build_training_scenes, collate, sample_windows

QUANTILES = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9])


class Discriminator(nn.Module):
    """LSGAN critic on a velocity sequence [A, T, 2], optionally conditioned on the attribute one-hot."""

    def __init__(self, attr_dim=0, hid=128):
        super().__init__()
        self.attr_dim = attr_dim
        self.gru = nn.GRU(2 + attr_dim, hid, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hid, hid), nn.LeakyReLU(0.2), nn.Linear(hid, 1))

    def forward(self, vel, attr=None):
        x = torch.cat([vel, attr[:, None, :].expand(-1, vel.shape[1], -1)], -1) if self.attr_dim else vel
        _, h = self.gru(x)
        return self.head(h[-1]).squeeze(-1)


def soft_dtw(x, y, gamma=1.0):
    """Soft-DTW between trajectories x, y [A, T, 2] with cost |.|_2 / GRID -> [A]."""
    A, T, _ = x.shape
    D = torch.cdist(x, y) / GRID
    INF = 1e6
    prev = [torch.zeros(A, device=x.device)] + [torch.full((A,), INF, device=x.device) for _ in range(T)]
    for i in range(1, T + 1):
        cur = [torch.full((A,), INF, device=x.device)]
        for j in range(1, T + 1):
            st = torch.stack([prev[j], cur[j - 1], prev[j - 1]], -1)
            cur.append(D[:, i - 1, j - 1] + (-gamma * torch.logsumexp(-st / gamma, -1)))
        prev = cur
    return prev[T]


def kinematics_match(traj, gt):
    """Per agent: mean |speed - speed_gt| + mean |acceleration - acceleration_gt| (grid / GRID)."""
    tv = (traj[:, 1:] - traj[:, :-1]).norm(dim=-1) / GRID
    gv = (gt[:, 1:] - gt[:, :-1]).norm(dim=-1) / GRID
    out = (tv - gv).abs().mean(1)
    if tv.shape[1] > 1:
        out = out + ((tv[:, 1:] - tv[:, :-1]) - (gv[:, 1:] - gv[:, :-1])).abs().mean(1)
    return out


def speed_quantile_loss(traj, gt, alive, attr):
    """Per attribute slot (>= 3 alive agents): L1 between generated and GT step-speed quantiles."""
    vg = (traj[:, 1:] - traj[:, :-1]).norm(dim=-1) / GRID
    vt = (gt[:, 1:] - gt[:, :-1]).norm(dim=-1) / GRID
    qs = QUANTILES.to(traj.device)
    tot, n = 0.0, 0
    for v in range(attr.shape[1]):
        sel = (attr[:, v] > 0) & (alive > 0)
        if sel.sum() < 3:
            continue
        tot = tot + (torch.quantile(vg[sel].reshape(-1), qs) - torch.quantile(vt[sel].reshape(-1), qs)).abs().mean()
        n += 1
    return tot / n if n else traj.sum() * 0.0


def neighbour_quantile_loss(traj, gt, alive, groups):
    """Per window (>= 3 alive agents): L1 between generated and GT nearest-neighbour-distance quantiles."""
    qs = QUANTILES.to(traj.device)
    tot, n = 0.0, 0
    for a, b in groups.tolist():
        sel = torch.nonzero(alive[a:b] > 0).squeeze(-1) + a
        if len(sel) < 3:
            continue

        def nn_dist(x):
            d = torch.cdist(x[sel].transpose(0, 1), x[sel].transpose(0, 1))
            d = d + torch.eye(len(sel), device=x.device)[None] * 1e6
            return d.min(-1).values.reshape(-1) / GRID

        tot = tot + (torch.quantile(nn_dist(traj), qs) - torch.quantile(nn_dist(gt), qs)).abs().mean()
        n += 1
    return tot / n if n else traj.sum() * 0.0


def attribute_head_loss(model, b, s1, windows, scenes):
    """Cross-entropy of the attribute head at GT origins (<= 64 per window), summed over the 4 groups."""
    off = [0, 2, 5, 8]
    tot, n = 0.0, 0
    for i, w in enumerate(windows):
        P, Y = scenes[w[0]]["attr_origins"], scenes[w[0]]["attr_onehot"]
        if P.shape[0] < 2:
            continue
        k = torch.randperm(P.shape[0], device=P.device)[:64]
        P, Y = P[k], Y[k]
        A = P.shape[0]
        lp = model.attr_head.group_log_probs(model.attribute_logits(b[i:i + 1].expand(A, -1, -1, -1), s1[i:i + 1].expand(A, -1, -1, -1), P))
        ti = torch.stack([Y[:, 0].long(), Y[:, 1:4].argmax(-1), Y[:, 4:7].argmax(-1), Y[:, 7:10].argmax(-1)], -1)
        tot = tot + sum(-lp[torch.arange(A, device=P.device), off[g] + ti[:, g]].mean() for g in range(4))
        n += 1
    return tot / n if n else b.sum() * 0.0


def waypoint(k, fut, last_idx, lookahead):
    """GT position `lookahead` steps ahead of rollout step k, clipped to the agent's last alive step."""
    return fut[torch.arange(fut.shape[0], device=fut.device), torch.clamp(last_idx, max=k + lookahead)]


def last_alive(mask):
    idx = (mask.shape[1] - 1) - torch.argmax(torch.flip(mask, [1]), dim=1)
    return torch.where(mask.sum(1) > 0, idx, torch.zeros_like(idx))


def rollout(model, scene, hist, fut, mask, attr_emb, groups, mask_soc, steps, lookahead, recompute=False, latent=None):
    """Free-running rollout -> [A, steps, 2]; gradients flow through positions and the network (the rendered
    heatmaps are detached). recompute: re-run the scene encoder in the backward pass to save memory."""
    h, pos = hist, hist[:, -1]
    vel = hist[:, -1] - hist[:, -2]
    li = last_alive(mask)
    preds = []
    for k in range(steps):
        goal = waypoint(k, fut, li, lookahead)
        ego, nbr, p, v = render_context(h.detach(), groups)
        soc = model.social(p, v, mask_soc, attr_emb)
        if recompute and torch.is_grad_enabled():
            _, skips = checkpoint(model.encode, scene, ego, nbr, soc, use_reentrant=False)
        else:
            _, skips = model.encode(scene, ego, nbr, soc)
        new_pos = pos + model.step(skips, pos, goal, vel, soc, attr_emb, latent)
        preds.append(new_pos)
        vel = new_pos - pos
        pos = new_pos
        h = torch.cat([h, pos[:, None]], 1)
    return torch.stack(preds, 1)


@torch.no_grad()
def rollout_error(model, scenes, cfg, step):
    """Mean free-running position error (grid px) over fixed training windows, `horizon` steps."""
    model.eval()
    windows = sample_windows(scenes, np.random.default_rng(123), cfg.eval_windows, cfg, step)
    err_sum = n_sum = 0.0
    for i in range(0, len(windows), cfg.windows_per_batch):
        scene, hist, fut, mask, goal, attr, groups = collate(scenes, windows[i:i + cfg.windows_per_batch])
        A = hist.shape[0]
        pred = rollout(model, scene, hist, fut, mask, model.embed_attributes(attr), groups,
                       torch.zeros(A, A, device=DEVICE), cfg.horizon, cfg.waypoint_lookahead)
        err_sum += ((pred - fut).norm(dim=-1) * mask).sum().item()
        n_sum += mask.sum().item()
    return err_sum / max(n_sum, 1)


def grad_norm(loss, params):
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return float(sum((g ** 2).sum() for g in grads if g is not None) ** 0.5)


class Trainer:
    def __init__(self, fold, prof, seed=0):
        self.fold, self.prof, self.cfg = fold, prof, prof.train
        self.step = prof.dataset.step
        torch.manual_seed(seed); np.random.seed(seed)
        self.rng = np.random.default_rng(seed)
        self.scenes = build_training_scenes(fold, prof)
        self.model = ACG(prof.model).to(DEVICE)
        self.disc = Discriminator(prof.model.attr_dim if self.cfg.attribute_discriminator else 0).to(DEVICE)
        self.opt = torch.optim.AdamW(self.model.parameters(), self.cfg.lr, weight_decay=1e-5)
        self.opt_disc = torch.optim.Adam(self.disc.parameters(), self.cfg.lr_disc, betas=(0.5, 0.9))
        self.w_kin = None if self.cfg.kinematics_share > 0 else 0.0
        self.calib = [0.0, 0.0, 0]
        print(f"[{fold}] parameters: {sum(p.numel() for p in self.model.parameters()):,}", flush=True)

    def train_batch(self, windows, calibrating):
        cfg, m, D = self.cfg, self.model, self.disc
        scene, hist, fut, mask, goal, attr, groups = collate(self.scenes, windows)
        A = hist.shape[0]
        no_mask = torch.zeros(A, A, device=DEVICE)
        attr_emb = m.embed_attributes(attr)
        latent = cfg.latent_sigma * torch.randn(A, m.step_feature_dim, device=DEVICE) if cfg.latent_sigma > 0 else None
        traj = rollout(m, scene, hist, fut, mask, attr_emb, groups, no_mask, cfg.bptt, cfg.waypoint_lookahead,
                       cfg.recompute_encoder, latent)
        gt = fut[:, :cfg.bptt]
        alive = mask[:, :cfg.bptt].prod(1)
        if alive.sum() < 2:
            return None
        vel_gen = (traj[:, 1:] - traj[:, :-1]) / GRID
        vel_gt = (gt[:, 1:] - gt[:, :-1]) / GRID
        w = alive / alive.sum().clamp(min=1)
        # discriminator step
        self.opt_disc.zero_grad()
        (((D(vel_gt.detach(), attr) - 1) ** 2 * w).sum() + ((D(vel_gen.detach(), attr) ** 2) * w).sum()).backward()
        self.opt_disc.step()
        # generator step
        self.opt.zero_grad()
        den = alive.sum().clamp(min=1)
        l_pos = (((traj - gt).norm(dim=-1).mean(1)) / GRID * alive).sum() / den
        l_dtw = (soft_dtw(traj, gt, cfg.dtw_gamma) * alive).sum() / den
        l_adv = (((D(vel_gen, attr) - 1) ** 2) * w).sum()
        loss = cfg.w_pos * l_pos + cfg.w_dtw * l_dtw + cfg.w_adv * l_adv
        if cfg.w_speed > 0:
            loss = loss + cfg.w_speed * speed_quantile_loss(traj, gt, alive, attr)
        if cfg.w_nbr > 0:
            loss = loss + cfg.w_nbr * neighbour_quantile_loss(traj, gt, alive, groups)
        first = groups[:, 0]
        z = torch.zeros(first.shape[0], GRID, GRID, device=DEVICE)
        bS, skS = m.encode(scene[first], z, z, torch.zeros(first.shape[0], self.prof.model.social_dim, device=DEVICE))
        target = torch.stack([self.scenes[win[0]]["spawn_target"] for win in windows], 0)
        pw = torch.tensor(cfg.heatmap_pos_weight, device=DEVICE)
        loss = loss + cfg.w_spawn * F.binary_cross_entropy_with_logits(m.spawn_logits(bS, skS), target, pos_weight=pw)
        loss = loss + cfg.w_attr * attribute_head_loss(m, bS, skS[0], windows, self.scenes)
        ego, nbr, p, v = render_context(hist, groups)
        b0, sk0 = m.encode(scene, ego, nbr, m.social(p, v, no_mask, attr_emb))
        gl = m.goal_logits(b0, sk0, attr_emb)
        l_goal = (F.binary_cross_entropy_with_logits(gl, heatmaps(goal), pos_weight=pw)
                  + (soft_argmax(gl) - goal).norm(dim=-1).mean() / GRID)
        loss = loss + cfg.w_goal * l_goal
        if cfg.kinematics_share > 0:
            l_kin = (kinematics_match(traj, gt) * alive).sum() / den
            if self.w_kin is not None:
                loss = loss + self.w_kin * l_kin
            elif calibrating:
                params = list(m.vel_head.parameters())
                self.calib[0] += grad_norm(loss, params); self.calib[1] += grad_norm(l_kin, params); self.calib[2] += 1
        if not torch.isfinite(loss):
            self.opt.zero_grad()
            return None
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.vel_head.parameters(), 1.0)
        self.opt.step()
        return l_pos.item(), l_adv.item()

    def train(self, out_path, log=print):
        cfg, best = self.cfg, float("inf")
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        for epoch in range(cfg.epochs):
            t0 = time.time()
            self.model.train(); self.disc.train()
            calibrating = self.w_kin is None and epoch + 1 == cfg.kinematics_calibration_epoch
            windows = sample_windows(self.scenes, self.rng, cfg.windows_per_epoch, cfg, self.step)
            stats = [s for s in (self.train_batch(windows[i:i + cfg.windows_per_batch], calibrating)
                                 for i in range(0, len(windows), cfg.windows_per_batch)) if s is not None]
            if calibrating and self.calib[2] > 0:
                base, kin = self.calib[0] / self.calib[2], self.calib[1] / self.calib[2]
                self.w_kin = cfg.kinematics_share * base / kin if kin > 1e-9 and np.isfinite(base) and np.isfinite(kin) else 0.0
                log(f"[{self.fold}] kinematics term weight {self.w_kin:.4f}")
            err = rollout_error(self.model, self.scenes, cfg, self.step)
            pos, adv = np.mean(stats, 0) if stats else (float("nan"), float("nan"))
            msg = f"[{self.fold}] epoch {epoch + 1}/{cfg.epochs} {time.time() - t0:.0f}s pos={pos:.4f} adv={adv:.3f} rollout_err={err:.3f}"
            if err < best:
                best = err
                torch.save(self.model.state_dict(), out_path)
                msg += "  *"
            log(msg)
        return best

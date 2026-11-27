"""ACG: one shared scene encoder feeding the emitter (spawn decoder + attribute head), the goal decoder and the
velocity head (simulator)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GRID, ModelConfig, width


def cbr(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, 1, 1), nn.GroupNorm(8, o), nn.SiLU())


class SocialEncoder(nn.Module):
    """Attention over co-present agents with a relative-position bias (and attribute-dependent tokens and pair bias
    when attribute_attention), plus a GRU over the agent's own velocities."""

    def __init__(self, ad, dsoc, attr_emb, attribute_attention):
        super().__init__()
        self.tok = nn.Sequential(nn.Linear(2, ad), nn.ReLU())
        self.relbias = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1))
        self.attn = nn.MultiheadAttention(ad, 4, batch_first=True)
        self.velgru = nn.GRU(2, ad, batch_first=True)
        self.out = nn.Sequential(nn.Linear(ad + ad, dsoc), nn.ReLU(), nn.Linear(dsoc, dsoc))
        self.attribute_attention = attribute_attention
        if attribute_attention:
            self.tok_a = nn.Linear(attr_emb, ad)
            self.relbias_a = nn.Sequential(nn.Linear(2 + 2 * attr_emb, 16), nn.ReLU(), nn.Linear(16, 1))
            nn.init.zeros_(self.tok_a.weight); nn.init.zeros_(self.tok_a.bias)
            nn.init.zeros_(self.relbias_a[2].weight); nn.init.zeros_(self.relbias_a[2].bias)

    def forward(self, pos, vel_hist, mask, attr_emb, return_attention=False):
        """pos [A, 2], vel_hist [A, T, 2] (grid / GRID); mask [A, A] additive attention mask."""
        A = pos.shape[0]
        rel = pos[None] - pos[:, None]
        if self.attribute_attention:
            tok = (self.tok(pos) + self.tok_a(attr_emb))[None]
            pair = torch.cat([rel, attr_emb[:, None].expand(-1, A, -1), attr_emb[None].expand(A, -1, -1)], -1)
            bias = self.relbias(rel).squeeze(-1) + self.relbias_a(pair).squeeze(-1) + mask
        else:
            tok = self.tok(pos)[None]
            bias = self.relbias(rel).squeeze(-1) + mask
        soc, weights = self.attn(tok, tok, tok, attn_mask=bias)
        _, hv = self.velgru(vel_hist)
        out = self.out(torch.cat([soc[0], hv[0]], -1))
        return (out, weights[0]) if return_attention else out


class SceneEncoder(nn.Module):
    """U-Net down path over [scene, ego heatmap, neighbour heatmap]; the social embedding is injected by FiLM at
    the bottleneck."""

    def __init__(self, cin, dsoc, w):
        super().__init__()
        self.e1 = cbr(cin, width(32, w)); self.e2 = cbr(width(32, w), width(64, w))
        self.e3 = cbr(width(64, w), width(128, w)); self.e4 = cbr(width(128, w), width(256, w))
        self.pool = nn.MaxPool2d(2)
        self.bott = cbr(width(256, w), width(256, w))
        self.film = nn.Linear(dsoc, width(256, w) * 2)

    def forward(self, x, soc):
        s1 = self.e1(x); s2 = self.e2(self.pool(s1)); s3 = self.e3(self.pool(s2)); s4 = self.e4(self.pool(s3))
        b = self.bott(self.pool(s4))
        g, be = self.film(soc).chunk(2, -1)
        b = b * (1 + g[:, :, None, None]) + be[:, :, None, None]
        return b, (s1, s2, s3, s4)


class MapDecoder(nn.Module):
    """U-Net up path to `cout` full-resolution logit maps."""

    def __init__(self, cout, w):
        super().__init__()
        self.d4 = cbr(width(256, w) + width(256, w), width(128, w)); self.d3 = cbr(width(128, w) + width(128, w), width(64, w))
        self.d2 = cbr(width(64, w) + width(64, w), width(32, w)); self.d1 = cbr(width(32, w) + width(32, w), width(32, w))
        self.head = nn.Conv2d(width(32, w), cout, 1)

    def forward(self, b, skips):
        s1, s2, s3, s4 = skips
        x = self.d4(torch.cat([F.interpolate(b, size=s4.shape[-1], mode="nearest"), s4], 1))
        x = self.d3(torch.cat([F.interpolate(x, size=s3.shape[-1], mode="nearest"), s3], 1))
        x = self.d2(torch.cat([F.interpolate(x, size=s2.shape[-1], mode="nearest"), s2], 1))
        x = self.d1(torch.cat([F.interpolate(x, size=s1.shape[-1], mode="nearest"), s1], 1))
        return self.head(x)


class AttributeHead(nn.Module):
    """p(attributes | origin, scene): the local feature at the origin queries the bottleneck tokens by
    cross-attention; 11 logits = 2/3/3/3-way groups."""

    def __init__(self, c_local, c_bottleneck, n_logits, d=64):
        super().__init__()
        self.q = nn.Linear(c_local + 2, d)
        self.kv = nn.Linear(c_bottleneck + 2, d)
        self.attn = nn.MultiheadAttention(d, 4, batch_first=True)
        self.out = nn.Sequential(nn.Linear(2 * d, d), nn.SiLU(), nn.Linear(d, n_logits))

    def forward(self, b, s1, pos):
        A, _, h, w_ = b.shape
        G = s1.shape[-1]
        g = torch.stack([pos[:, 0] / (G - 1) * 2 - 1, pos[:, 1] / (G - 1) * 2 - 1], -1)
        local = F.grid_sample(s1, g[:, None, None, :], mode="bilinear", align_corners=True).reshape(A, -1)
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=b.device), torch.linspace(-1, 1, w_, device=b.device), indexing="ij")
        coords = torch.stack([xx, yy], -1).reshape(1, h * w_, 2).expand(A, -1, -1)
        tok = torch.cat([b.flatten(2).transpose(1, 2), coords], -1)
        q = self.q(torch.cat([local, g], -1))[:, None]
        kv = self.kv(tok)
        o, _ = self.attn(q, kv, kv)
        return self.out(torch.cat([o[:, 0], q[:, 0]], -1))

    @staticmethod
    def group_log_probs(logits):
        return torch.cat([F.log_softmax(logits[:, 0:2], -1), F.log_softmax(logits[:, 2:5], -1),
                          F.log_softmax(logits[:, 5:8], -1), F.log_softmax(logits[:, 8:11], -1)], -1)


class VelocityHead(nn.Module):
    """MLP from the step context to the next displacement; the attribute embedding (last attr_emb features)
    modulates both hidden layers by FiLM and scales the output."""

    def __init__(self, din, attr_emb):
        super().__init__()
        self.attr_emb = attr_emb
        self.net = nn.Sequential(nn.Linear(din, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 2))
        with torch.no_grad():
            self.net[-1].weight.mul_(0.01); self.net[-1].bias.zero_()
        self.film1 = nn.Linear(attr_emb, 256); self.film2 = nn.Linear(attr_emb, 256); self.scale = nn.Linear(attr_emb, 1)
        for layer in (self.film1, self.film2, self.scale):
            nn.init.zeros_(layer.weight); nn.init.zeros_(layer.bias)

    def forward(self, feat):
        a = feat[:, -self.attr_emb:]
        g1, b1 = self.film1(a).chunk(2, -1)
        g2, b2 = self.film2(a).chunk(2, -1)
        h = self.net[1](self.net[0](feat)); h = h * (1 + g1) + b1
        h = self.net[3](self.net[2](h)); h = h * (1 + g2) + b2
        return self.net[4](h) * torch.exp(self.scale(a))


class ACG(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        w = cfg.width
        self.cfg = cfg
        self.social = SocialEncoder(cfg.social_token_dim, cfg.social_dim, cfg.attr_emb, cfg.attribute_social_attention)
        self.attr_head = AttributeHead(width(32, w), width(256, w), cfg.attr_dim)
        self.attribute_goal_film = cfg.attribute_goal_film
        if cfg.attribute_goal_film:
            self.goal_film = nn.Linear(cfg.attr_emb, 2 * width(256, w))
            nn.init.zeros_(self.goal_film.weight); nn.init.zeros_(self.goal_film.bias)
        self.scene_enc = SceneEncoder(cfg.scene_channels + 2, cfg.social_dim, w)
        self.dec_goal = MapDecoder(1, w)
        self.vel_head = VelocityHead(width(32, w) + cfg.social_dim + 2 + 2 + cfg.attr_emb, cfg.attr_emb)
        self.dec_spawn = MapDecoder(cfg.spawn_channels, w)
        self.attr_proj = nn.Linear(cfg.attr_dim, cfg.attr_emb)

    @property
    def step_feature_dim(self):
        return self.vel_head.net[0].in_features

    def embed_attributes(self, attr):
        return self.attr_proj(attr)

    def encode(self, scene, ego, nbr, soc, chunk=0):
        x = torch.cat([scene, ego[:, None], nbr[:, None]], 1)
        A = x.shape[0]
        if chunk <= 0 or A <= chunk:
            return self.scene_enc(x, soc)
        bs, skips = [], None
        for i in range(0, A, chunk):
            b_i, sk_i = self.scene_enc(x[i:i + chunk], soc[i:i + chunk])
            bs.append(b_i)
            skips = [[] for _ in sk_i] if skips is None else skips
            for j, s in enumerate(sk_i):
                skips[j].append(s)
        return torch.cat(bs, 0), [torch.cat(s, 0) for s in skips]

    def goal_logits(self, b, skips, attr_emb):
        if self.attribute_goal_film:
            g, bb = self.goal_film(attr_emb).chunk(2, -1)
            b = b * (1 + g[:, :, None, None]) + bb[:, :, None, None]
        return self.dec_goal(b, skips)[:, 0]

    def spawn_logits(self, b, skips):
        return self.dec_spawn(b, skips)

    def attribute_logits(self, b, s1, pos):
        return self.attr_head(b, s1, pos)

    def step(self, skips, pos, goal, vel, soc, attr_emb, latent=None):
        """One simulator step: displacement [A, 2] (grid px) from the local scene feature, the social embedding,
        the goal direction, the velocity and the attribute embedding (+ an optional per-agent latent)."""
        s1 = skips[0]
        A, _, G, _ = s1.shape
        grid = torch.stack([pos[:, 0] / (G - 1) * 2 - 1, pos[:, 1] / (G - 1) * 2 - 1], -1)[:, None, None, :]
        local = F.grid_sample(s1, grid, mode="bilinear", align_corners=True).reshape(A, -1)
        gd = goal - pos
        goal_unit = gd / (gd.norm(dim=-1, keepdim=True) + 1e-6)
        feat = torch.cat([local, soc, goal_unit, vel / GRID, attr_emb], -1)
        if latent is not None:
            feat = feat + latent
        return self.vel_head(feat)


def load_model(path, cfg, device):
    model = ACG(cfg).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.eval()

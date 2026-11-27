"""Per-benchmark profiles: model, training recipe and generation setting."""
import dataclasses
import json
import os
from dataclasses import dataclass
from typing import Optional

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("ACG_DATA", os.path.join(ROOT, "data"))
WEIGHTS_DIR = os.environ.get("ACG_WEIGHTS", os.path.join(ROOT, "weights"))
OUTPUT_DIR = os.environ.get("ACG_OUTPUT", os.path.join(ROOT, "outputs"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GRID = 320      # scene raster (GRID x GRID)
SIGMA = 4.0     # heatmap sigma (grid px)
OBS = 8         # history steps rendered into the ego heatmap / velocity GRU


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    fps: int                     # dense track / output frame rate
    sim_fps: int = 5             # simulator step rate

    @property
    def step(self):
        return self.fps // self.sim_fps


@dataclass(frozen=True)
class ModelConfig:
    width: float = 0.5           # U-Net channel multiplier (1.0 = 32/64/128/256)
    scene_channels: int = 9      # walkable + 8 semantic classes
    social_dim: int = 96
    social_token_dim: int = 64
    attr_dim: int = 11           # attribute one-hot
    attr_emb: int = 16
    spawn_channels: int = 13     # 11 attribute-value origin maps + all origins + standing occupancy
    attribute_social_attention: bool = True
    attribute_goal_film: bool = True


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 80
    windows_per_epoch: int = 128
    windows_per_batch: int = 4
    max_agents: int = 48
    min_agents: int = 2
    history: int = OBS
    horizon: int = 20
    bptt: int = 8                # free-running steps back-propagated through
    waypoint_lookahead: int = 4  # training goal = GT position this many steps ahead
    attr_balance: float = 0.0    # probability of anchoring a window on a rare-attribute agent
    lr: float = 3e-4
    lr_disc: float = 2e-4
    w_pos: float = 1.0
    w_dtw: float = 0.5
    w_adv: float = 0.3
    w_goal: float = 1.0
    w_spawn: float = 1.0
    w_attr: float = 0.5
    w_speed: float = 0.0         # per-attribute speed-quantile term
    w_nbr: float = 0.0           # nearest-neighbour-distance quantile term
    attribute_discriminator: bool = False
    kinematics_share: float = 0.3      # speed/acceleration term as a share of the base gradient (0 = off)
    kinematics_calibration_epoch: int = 8
    latent_sigma: float = 0.0    # per-agent latent on the velocity-head input
    dtw_gamma: float = 1.0
    heatmap_pos_weight: float = 50.0
    stationary_step_m: float = 0.15
    scene_scale: Optional[float] = None    # m per grid px (None: from the homography)
    eval_windows: int = 48
    recompute_encoder: bool = True     # gradient checkpointing of the scene encoder


@dataclass(frozen=True)
class GenerateConfig:
    trials: int = 5
    window_seconds: int = 10
    history: int = 10
    future: int = 10
    max_emit_per_window: int = 64
    population_source: str = "crowdes"  # "crowdes": scene prior of the test scene; "histogram": training frames
    population_multiplier: float = 1.0
    spawn_temperature: float = 0.1
    attribute_source: str = "head"      # "head": attribute head; "prior": training distribution
    group_emission: bool = False
    group_jitter_m: float = 0.8
    group_goal_jitter_m: float = 0.5
    entry_exit_prior: bool = False      # origins / goals from recorded entry-exit pairs
    entry_jitter_px: float = 3.0
    occupancy_channel: int = 12
    goal_temperature: float = 0.5
    goal_walk_penalty: float = 12.0
    goal_min_dist_px: float = 8.0
    goal_dist_scale: float = 1.0
    goal_occupancy_weight: float = 1.0
    scene_scale: Optional[float] = None
    lifetime_cap_quantile: Optional[float] = None
    velocity_scale: float = 1.0
    velocity_noise: float = 0.0         # AR(1) step noise (grid px)
    velocity_noise_rho: float = 0.7
    velocity_inertia: float = 0.0       # v_t = a v_{t-1} + (1 - a) delta_t
    latent_sigma: float = 0.0
    speed_heterogeneity: float = 0.0    # per-agent log-normal speed multiplier
    collision_radius_m: float = 0.0
    collision_radius_px: float = 0.0    # takes precedence when > 0
    collision_iters: int = 3
    collision_push: float = 0.5
    arrival_m: float = 0.5
    encode_chunk: int = 64


@dataclass(frozen=True)
class Profile:
    dataset: DatasetConfig
    model: ModelConfig
    train: TrainConfig
    generate: GenerateConfig

    def replace(self, model=None, train=None, generate=None):
        return Profile(self.dataset, dataclasses.replace(self.model, **(model or {})),
                       dataclasses.replace(self.train, **(train or {})), dataclasses.replace(self.generate, **(generate or {})))


PROFILES = {
    "ethucy": Profile(
        DatasetConfig("ethucy", 25),
        ModelConfig(width=0.5, attribute_social_attention=False, attribute_goal_film=False),
        TrainConfig(),
        GenerateConfig(population_multiplier=0.6, velocity_scale=1.2, velocity_noise=0.1, velocity_noise_rho=0.9,
                       velocity_inertia=0.9, collision_radius_m=0.5)),
    "sdd": Profile(
        DatasetConfig("sdd", 30),
        ModelConfig(width=0.25),
        TrainConfig(attr_balance=0.5, latent_sigma=1.0),
        GenerateConfig(population_multiplier=0.7, spawn_temperature=0.5, max_emit_per_window=128, goal_dist_scale=0.8,
                       goal_occupancy_weight=0.25, velocity_scale=0.8, velocity_noise=0.8, latent_sigma=1.0,
                       speed_heterogeneity=0.3, collision_radius_px=20.0)),
    "gcs": Profile(
        DatasetConfig("gcs", 30),
        ModelConfig(width=0.25),
        TrainConfig(latent_sigma=1.0),
        GenerateConfig(population_source="histogram", population_multiplier=0.1, spawn_temperature=0.5,
                       max_emit_per_window=384, attribute_source="prior", goal_temperature=8.0, goal_dist_scale=0.8,
                       goal_occupancy_weight=0.25, velocity_scale=1.3, latent_sigma=1.0, collision_radius_m=0.5)),
    "panda": Profile(
        DatasetConfig("panda", 2, sim_fps=2),
        ModelConfig(width=0.25),
        TrainConfig(epochs=36, max_agents=40, attr_balance=0.5, w_speed=0.5, w_nbr=0.3, attribute_discriminator=True,
                    kinematics_share=0.0, recompute_encoder=False),
        GenerateConfig(population_source="histogram", group_emission=True, history=5, future=5, max_emit_per_window=256,
                       goal_dist_scale=0.6, goal_occupancy_weight=2.0, velocity_noise=0.6, collision_radius_m=1.0)),
}

# Settings of the published rows that differ from the profiles (`--row paper`).
PAPER_ROWS = {
    "eth": {"train": {"epochs": 36, "scene_scale": 0.04605706013694104},
            "generate": {"population_multiplier": 0.8, "velocity_scale": 1.1, "scene_scale": 0.04605706013694104}},
    "hotel": {"model": {"width": 0.25, "attribute_social_attention": True, "attribute_goal_film": True},
              "train": {"attr_balance": 0.5, "latent_sigma": 1.0, "scene_scale": 0.022908760384716896},
              "generate": {"velocity_scale": 1.6, "velocity_noise": 0.6, "velocity_inertia": 0.0, "goal_dist_scale": 1.6,
                           "scene_scale": 0.022908760384716896}},
    "univ": {"generate": {"population_multiplier": 1.0, "velocity_scale": 1.4, "velocity_noise": 0.5, "velocity_inertia": 0.97}},
    "zara1": {"generate": {"population_multiplier": 0.55, "velocity_scale": 1.15, "velocity_inertia": 0.95,
                           "collision_radius_m": 0.3, "lifetime_cap_quantile": 0.8, "entry_exit_prior": True}},
    "zara2": {"generate": {"velocity_noise": 0.0, "entry_exit_prior": True}},
    "sdd": {"train": {"scene_scale": 0.1287}, "generate": {"scene_scale": 0.1287}},
    "gcs": {"train": {"scene_scale": 0.3483472656121415}, "generate": {"scene_scale": 0.3483472656121415}},
}


def dataset_of(fold):
    return json.load(open(os.path.join(DATA_DIR, "folds.json")))[fold]["dataset"]


def profile_of(fold, row=None):
    p = PROFILES[dataset_of(fold)]
    return p.replace(**PAPER_ROWS[fold]) if row == "paper" and fold in PAPER_ROWS else p


def width(c, w):
    """Channel count scaled by w, snapped to a multiple of 8."""
    return max(8, int(round(c * w / 8.0)) * 8)

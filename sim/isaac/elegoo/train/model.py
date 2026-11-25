"""Policy input packing and the action transform, shared by training, evaluation and ONNX export.

Inputs: encoded depth [N,1,120,160], aux [N,8] and a history of the previous HISTORY depth
summaries. aux = (range_cm/150, gyro_rate/250, last_left, last_right, sin err, cos err, level, limit),
with last_* in sent PWM/255 and level 0 / 0.5 / 1 for near / medium / far.
"""

from pathlib import Path

import torch
from tensordict import TensorDict
from torch import nn
from torch.nn import functional as F

ROWS, COLS, HISTORY = 4, 10, 3
FEATURES = 2 * ROWS * COLS
AUX = 8
OBS_DIM = AUX + FEATURES * (1 + HISTORY)
PWM_UNIT = 150 / 255


def summary(depth: torch.Tensor) -> torch.Tensor:
    """Nearest (max of encoded depth) and mean over a 4 x 10 grid of 30 x 16 px cells."""
    cells = (120 // ROWS, 160 // COLS)
    return torch.cat((F.max_pool2d(depth, cells).flatten(1), F.avg_pool2d(depth, cells).flatten(1)), dim=1)


def pack(depth: torch.Tensor, aux: torch.Tensor, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    f = summary(depth)
    vector = torch.cat((aux, f, history.flatten(1)), dim=1)
    return vector, torch.cat((f[:, None], history[:, :-1]), dim=1)


def wheels(raw: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
    """Residual on a heading-following prior; returns signed PWM/255 per side, |PWM| <= 150 * limit."""
    sin, cos, limit = aux[:, 4], aux[:, 5], aux[:, 7]
    v = (0.6 * cos.clamp_min(0).square() + raw[:, 0]).clamp(-0.4, 1.0)
    turn = (torch.where(cos > 0, 0.6 * sin, 0.6 * torch.sign(sin)) + raw[:, 1]).clamp(-0.8, 0.8)
    side = torch.stack((v - turn, v + turn), dim=1)
    peak = side.abs().amax(1, keepdim=True)
    scale = torch.minimum(torch.ones_like(peak), limit[:, None] / peak.clamp_min(1e-6))
    return side * scale * PWM_UNIT


class DriverModel(nn.Module):
    """Deterministic inference graph: (depth, aux, history) -> (PWM/255 per side, next history)."""

    def __init__(self, normalizer: nn.Module, mlp: nn.Module, head: nn.Module):
        super().__init__()
        self.normalizer, self.mlp, self.head = normalizer, mlp, head

    def forward(self, depth: torch.Tensor, aux: torch.Tensor, history: torch.Tensor):
        vector, next_history = pack(depth, aux, history)
        return wheels(self.head(self.mlp(self.normalizer(vector))), aux), next_history


def driver_model(actor) -> DriverModel:
    """Wrap an rsl_rl MLPModel actor (normalizer + MLP + Gaussian mean) for inference and export."""
    return DriverModel(actor.obs_normalizer, actor.mlp, actor.distribution.as_deterministic_output_module())


ACTOR = {"hidden_dims": [256, 128], "activation": "elu", "obs_normalization": True}
PPO_CFG = {
    "num_steps_per_env": 32, "save_interval": 1000,
    "actor": {"class_name": "MLPModel", **ACTOR,
              "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 0.25,
                                   "std_range": (0.03, 0.5), "std_type": "scalar"}},
    "critic": {"class_name": "MLPModel", **ACTOR},
    "obs_groups": {"actor": ["driver"], "critic": ["driver"]},
    "algorithm": {"class_name": "PPO", "learning_rate": 3e-4, "num_learning_epochs": 5,
                  "num_mini_batches": 4, "gamma": 0.99, "lam": 0.95, "entropy_coef": 0.002,
                  "desired_kl": 0.01, "schedule": "adaptive", "max_grad_norm": 1.0},
    "check_for_nan": True,
}


def load_actor(path: Path, device: str):
    """Rebuild the trained rsl_rl actor from a runner checkpoint (no simulator needed)."""
    from rsl_rl.models import MLPModel

    obs = TensorDict({"driver": torch.zeros((1, OBS_DIM), device=device)}, batch_size=[1])
    cfg = {k: v for k, v in PPO_CFG["actor"].items() if k != "class_name"}
    actor = MLPModel(obs, PPO_CFG["obs_groups"], "actor", 2, **cfg).to(device)
    actor.load_state_dict(torch.load(path, map_location=device, weights_only=False)["actor_state_dict"])
    return actor.eval()

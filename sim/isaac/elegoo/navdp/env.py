"""DriverEnv with the camera driver's real-car inputs (sim/cmdtest/DRIVER_API.md) and the privileged
signals used to fine-tune NavDP (expert trajectory labels, clearance-based trajectory scores).

Inputs per car and step, as the Mac side gets them:
- rgb: the car's RTX camera in the textured visual scene, appearance randomisation (appearance.py), a JPEG
  round trip at a per-episode quality of 60-90, delivered 1 or 2 steps late (0.10-0.25 s); 12 % of steps
  drop the frame and its depth (None on the car).
- depth: DriverEnv.policy_obs(): the ray-cast camera with every observation_spec.md corruption (encoded),
  lens drawn per episode within calibration.yaml. Optional Depth Pro domain (p7, set_depth_domain; fitted to
  depth_measure.py on the DA3 renders): the per-episode global scale drawn log-uniform from a per-car range instead
  of the spec's 0.75-1.33; a depth-range compression z * (z / 1.5 m)^-gamma, gamma ~ U(0, gamma_max) per episode
  (Depth Pro over-reads near scenes more); a smooth multiplicative error (8 x 6 control points, bicubic, log sd =
  smooth_sd x U(0.25, 1.75) per episode, drifting AR(1) per step).
- range_cm: cone sonar, missing on 50 % of steps; gyro rate / integrated heading; last sent PWM.
- command: target_xy = the navigator's lookahead waypoint (DriverEnv.command, with its noise and staleness)
  in the gyro frame, speed_limit.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..train.driver_env import DriverEnv, RunSpec

HERE = Path(__file__).resolve().parent
for candidate in (HERE.parents[2] / "models" / "navdp2", HERE.parents[1] / "model"):
    if candidate.is_dir() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

PREDICT = 24
RES, X0, Y0 = 0.02, -11.49, 11.17      # floor map grid, sim/isaac/elegoo/nav/plan.py
HALF_WIDTH = 0.075
RGB_SIZE = (224, 168)
JPEG_QUALITY = (60, 70, 80, 90)
SMOOTH_GRID = (6, 8)      # rows, cols of the low-frequency depth error field
PIVOT_M = 1.5             # depth left unchanged by the compression
SMOOTH_RHO = 0.95         # per-step AR(1) persistence of that field


@dataclass(frozen=True, slots=True)
class NavDPSpec(RunSpec):
    drop_rate: float = 0.12
    sonar_missing: float = 0.5
    label_spacing_m: float = 0.05
    depth_scale: tuple[float, float] | None = None   # Depth Pro domain: global scale range (None = spec 0.75-1.33)
    smooth_sd: float = 0.0                           # Depth Pro domain: mean log sd of the smooth error field
    depth_gamma: float = 0.0                         # Depth Pro domain: max depth-range compression exponent


def clearance_grid(device) -> torch.Tensor:
    """Distance (m) from each floor-map cell to the nearest non-free cell (plan.py Map.clearance)."""
    from PIL import Image
    from scipy import ndimage

    for root in (HERE.parents[3], HERE.parents[1]):
        path = root / "scans" / "merged" / "floor_map.png"
        if path.exists():
            free = np.array(Image.open(path)) == 254
            return torch.tensor(ndimage.distance_transform_edt(free) * RES, dtype=torch.float32, device=device)
    raise FileNotFoundError("scans/merged/floor_map.png")


class NavDPEnv(DriverEnv):
    def __init__(self, sim, spec: NavDPSpec):
        super().__init__(sim, spec)
        n, w, h = spec.count, spec.camera.width, spec.camera.height
        self.clearance_map = clearance_grid(self.device)
        self.frames = torch.zeros((3, n, h, w, 3), dtype=torch.uint8, device=self.device)
        self.quality = torch.zeros(n, dtype=torch.long, device=self.device)
        self.lag = torch.ones(n, dtype=torch.long, device=self.device)
        self.scale_range = torch.zeros((n, 2), device=self.device)   # log lo, log hi; lo == hi == 0 -> spec draw
        self.smooth_mean = torch.full((n,), spec.smooth_sd, device=self.device)
        self.gamma_max = torch.full((n,), spec.depth_gamma, device=self.device)
        self.smooth_sd = torch.zeros(n, device=self.device)
        self.gamma = torch.zeros(n, device=self.device)
        self.smooth = torch.zeros((n, 1, *SMOOTH_GRID), device=self.device)
        if spec.depth_scale is not None:
            self.set_depth_domain(self.ids, spec.depth_scale, spec.smooth_sd, spec.depth_gamma)

    def set_depth_domain(self, ids: torch.Tensor, scale: tuple[float, float] | None, smooth_sd: float,
                         gamma_max: float) -> None:
        """Per car: global depth scale range (None = the spec's own draw), mean smooth error sd and maximum
        compression exponent; applies from the cars' next reset."""
        lo, hi = (0.0, 0.0) if scale is None else (float(np.log(scale[0])), float(np.log(scale[1])))
        self.scale_range[ids] = torch.tensor((lo, hi), device=self.device)
        self.smooth_mean[ids] = smooth_sd
        self.gamma_max[ids] = gamma_max

    def reset(self, ids=None, routes=None) -> None:
        super().reset(ids, routes)
        if not self.ready:
            return
        ids = self.ids if ids is None else ids
        self.quality[ids] = torch.randint(len(JPEG_QUALITY), (len(ids),), device=self.device)
        self.lag[ids] = torch.randint(1, 3, (len(ids),), device=self.device)
        self.frames[:, ids] = 0
        lo, hi = self.scale_range[ids].unbind(1)
        custom = (hi > lo) & ~self.depth_sim.stress[ids]
        drawn = torch.lerp(lo, hi, torch.rand(len(ids), device=self.device)).exp()
        self.depth_sim.scale[ids] = torch.where(custom, drawn, self.depth_sim.scale[ids])
        u = torch.rand((2, len(ids)), device=self.device)
        self.smooth_sd[ids] = self.smooth_mean[ids] * (0.25 + 1.5 * u[0])
        self.gamma[ids] = self.gamma_max[ids] * u[1]
        self.smooth[ids] = torch.randn_like(self.smooth[ids])

    def depth_domain(self, encoded: torch.Tensor) -> torch.Tensor:
        """Each car's compression z * (z / PIVOT_M)^-gamma and smooth field exp(sd * f) (f unit-variance,
        bicubic from SMOOTH_GRID control points) on the decoded depth; no-return pixels (encoded 0) stay
        no-return. Steps the field's AR(1)."""
        from ..train.depth import decode, encode

        self.smooth = SMOOTH_RHO * self.smooth + (1 - SMOOTH_RHO**2) ** 0.5 * torch.randn_like(self.smooth)
        if not bool(((self.smooth_sd > 0) | (self.gamma > 0)).any()):
            return encoded
        field = torch.nn.functional.interpolate(self.smooth, size=encoded.shape[-2:], mode="bicubic",
                                                align_corners=True)[:, 0]
        z = decode(encoded)
        log_gain = self.smooth_sd[:, None, None] * field - self.gamma[:, None, None] * (z / PIVOT_M).log()
        return torch.where(encoded > 1e-4, encode(z * log_gain.exp()), encoded)

    def jpeg(self, rgb: torch.Tensor) -> torch.Tensor:
        from torchvision.io import decode_jpeg, encode_jpeg

        out = torch.empty_like(rgb)
        chw = rgb.permute(0, 3, 1, 2).contiguous()
        for k, q in enumerate(JPEG_QUALITY):
            ids = (self.quality == k).nonzero().squeeze(1)
            if not len(ids):
                continue
            data = [d.cpu() for d in encode_jpeg([chw[i] for i in ids.tolist()], quality=q)]
            out[ids] = torch.stack(decode_jpeg(data, device=self.device)).permute(0, 2, 3, 1)
        return out

    def car_inputs(self) -> dict[str, torch.Tensor]:
        depth, aux = self.policy_obs()
        depth = self.depth_domain(depth[:, 0])[:, None]
        rgb = self.camera.data.output["rgb"].torch[..., :3]
        if self.appearance is not None:
            rgb = self.appearance.augment(rgb)
        self.frames = torch.cat((self.jpeg(rgb)[None], self.frames[:-1]), 0)
        n = len(self.ids)
        late = self.frames[self.lag, self.ids]
        valid = (torch.rand(n, device=self.device) >= self.spec.drop_rate) & (self.age > self.lag)
        err = self.target_yaw - torch.deg2rad(self.sensors.yaw_deg)
        target = self.waypoint_distance[:, None] * torch.stack((err.cos(), err.sin()), 1)
        sonar = aux[:, 0] * 150
        sonar = torch.where(torch.rand(n, device=self.device) < self.spec.sonar_missing, torch.inf, sonar)
        return {
            "rgb": late, "rgb_valid": valid, "depth": depth[:, 0], "depth_valid": valid,
            "target_xy": target, "speed_limit": aux[:, 7], "heading": torch.deg2rad(self.sensors.yaw_deg),
            "range_cm": sonar, "last_pwm": (self.last * 255).round(),
            "rgb_age": self.lag.float() * self.control_dt + 0.05 * torch.rand(n, device=self.device),
            "gyro_rate": torch.deg2rad(self.sensors.rate_deg_s),
        }

    def expert_path(self, spacing: float | None = None) -> torch.Tensor:
        """Privileged label: the planned route from the car's true pose, resampled every `spacing` m along
        its length (24 points, the last repeated at the goal), in the car frame. [N,24,2]."""
        spacing = spacing or self.spec.label_spacing_m
        p = self.privileged()
        xy, yaw = p["pos_w"][:, :2], p["heading_rad"]
        final = self.counts[self.route] - 1
        k = torch.minimum(self.progress_index[:, None] + 1 + torch.arange(48, device=self.device), final[:, None])
        poly = torch.cat((xy[:, None], self.way[self.route[:, None], k]), 1)
        seg = (poly[:, 1:] - poly[:, :-1]).norm(dim=-1)
        along = torch.cat((torch.zeros_like(seg[:, :1]), seg.cumsum(1)), 1)
        s = (torch.arange(1, PREDICT + 1, device=self.device) * spacing)[None].minimum(along[:, -1:])
        hi = torch.searchsorted(along.contiguous(), s.contiguous()).clamp(1, poly.shape[1] - 1)
        lo = hi - 1
        a0, a1 = along.gather(1, lo), along.gather(1, hi)
        f = ((s - a0) / (a1 - a0).clamp_min(1e-6)).clamp(0, 1)[..., None]
        rows = torch.arange(len(xy), device=self.device)[:, None]
        world = torch.lerp(poly[rows, lo], poly[rows, hi], f) - xy[:, None]
        c, s_ = yaw.cos()[:, None], yaw.sin()[:, None]
        return torch.stack((c * world[..., 0] + s_ * world[..., 1], -s_ * world[..., 0] + c * world[..., 1]), -1)

    def path_score(self, ids: torch.Tensor, paths: torch.Tensor, expert: torch.Tensor) -> torch.Tensor:
        """Privileged critic target for car-frame paths [n,M,24,2] of cars `ids`: progress toward the expert
        path's end (normalised, -1..1) - 2 if any point is within 7.5 cm of an obstacle - mean proximity
        (clearance below 15 cm, linear)."""
        p = self.privileged()
        xy, yaw = p["pos_w"][ids, :2], p["heading_rad"][ids]
        c, s = yaw.cos()[:, None, None], yaw.sin()[:, None, None]
        wx = xy[:, None, None, 0] + c * paths[..., 0] - s * paths[..., 1]
        wy = xy[:, None, None, 1] + s * paths[..., 0] + c * paths[..., 1]
        row = ((Y0 - wy) / RES).round().long().clamp(0, self.clearance_map.shape[0] - 1)
        col = ((wx - X0) / RES).round().long().clamp(0, self.clearance_map.shape[1] - 1)
        clear = self.clearance_map[row, col]
        end = expert[ids][:, None, -1]
        reach = end.norm(dim=-1).clamp_min(0.05)
        progress = ((reach - (end - paths[..., -1, :]).norm(dim=-1)) / reach).clamp(-1, 1)
        hit = (clear < HALF_WIDTH).any(-1).float()
        proximity = ((0.15 - clear) / (0.15 - HALF_WIDTH)).clamp(0, 1).mean(-1)
        return progress - 2.0 * hit - proximity

"""Batched simulation of the car's narrow observation interface."""

import torch
from isaaclab.utils.math import quat_apply

from .world import BOXES, PAN_PIVOT


class Sensors:
    def __init__(self, parameters: dict[str, torch.Tensor], *, obstacles: bool) -> None:
        self.p = parameters
        self.yaw_deg = torch.zeros_like(parameters["gyro_bias_deg_s"])
        self.rate_deg_s = torch.zeros_like(self.yaw_deg)
        device = self.yaw_deg.device
        self.bounds = torch.tensor(
            [[box.center, box.size] for box in BOXES] if obstacles else [],
            device=device,
        ).reshape(-1, 2, 3)
        # A dense disk on the cone, including the axis and four concentric rings.
        angle = torch.arange(32, device=device) * (2 * torch.pi / 32)
        ring = torch.arange(1, 5, device=device) / 4
        plane = (ring[:, None, None] * torch.stack((angle.cos(), angle.sin()), -1)).reshape(-1, 2)
        plane = torch.cat((torch.zeros((1, 2), device=device), plane))
        lateral = plane[None] * torch.tan(torch.deg2rad(self.p["ultrasonic_full_cone_deg"] / 2))[:, None, None]
        self.rays = torch.nn.functional.normalize(
            torch.cat((torch.ones_like(lateral[..., :1]), lateral), -1), dim=-1
        )
        self.pivot = torch.tensor(PAN_PIVOT, device=device)
        self.floor_points = torch.tensor(
            [[0.0915, y, 0.00365] for y in (-0.012, 0.0, 0.012)], device=device,
        )

    def integrate(self, quat: torch.Tensor, angular_w: torch.Tensor, dt: float) -> None:
        vertical = torch.zeros_like(angular_w)
        vertical[:, 2] = 1
        yaw_rate = torch.rad2deg((quat_apply(quat, vertical) * angular_w).sum(-1))
        limit = self.p["gyro_rate_limit_deg_s"]
        yaw_rate += self.p["gyro_bias_deg_s"] + self.p["gyro_noise_deg_s"] * torch.randn_like(yaw_rate)
        self.rate_deg_s = yaw_rate.clamp(-limit, limit)
        self.yaw_deg += self.rate_deg_s * dt

    def floor_bits(self, root: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
        n = len(root)
        pos = quat_apply(quat[:, None].expand(-1, 3, -1), self.floor_points.expand(n, -1, -1)) + root[:, None]
        clearance = pos[..., 2] + self.p["floor_noise_m"][:, None] * torch.randn_like(pos[..., 2])
        return clearance < self.p["floor_threshold_m"][:, None]

    def range_cm(self, root: torch.Tensor, quat: torch.Tensor, pan: torch.Tensor) -> torch.Tensor:
        """Nearest floor or box hit over the sampled cone, rounded to integer cm."""
        pan_quat = torch.zeros_like(quat)
        pan_quat[:, 2], pan_quat[:, 3] = (pan / 2).sin(), (pan / 2).cos()
        emitter = torch.zeros_like(root)
        emitter[:, 0], emitter[:, 2] = 0.0247, 0.0225
        origin = root + quat_apply(quat, self.pivot + quat_apply(pan_quat, emitter))
        r = self.rays.shape[1]
        rays = quat_apply(quat[:, None].expand(-1, r, -1), quat_apply(pan_quat[:, None].expand(-1, r, -1), self.rays))
        z = rays[..., 2]
        floor = torch.where(z < -1e-6, -origin[:, 2:3] / z, torch.inf)
        nearest = floor.amin(dim=1)
        if self.bounds.numel():
            center, size = self.bounds.unbind(dim=1)
            # Slab intersections with the same local boxes the scene spawns.
            denom = rays[:, :, None]
            denom = torch.where(denom.abs() < 1e-7, 1e-7, denom)
            low = (center - size / 2 - origin[:, None, :])[:, None] / denom
            high = (center + size / 2 - origin[:, None, :])[:, None] / denom
            enter = torch.minimum(low, high).amax(-1).clamp_min(0)
            leave = torch.maximum(low, high).amin(-1)
            hits = torch.where(leave >= enter, enter, torch.inf)
            nearest = torch.minimum(nearest, hits.amin(dim=(1, 2)))
        cm = nearest * 100 + self.p["ultrasonic_range_noise_cm"] * torch.randn_like(nearest)
        return cm.round().clamp_min(0).minimum(self.p["ultrasonic_cap_cm"]).to(torch.int32)

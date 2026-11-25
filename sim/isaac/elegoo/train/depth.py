"""Vectorized depth observation per sim/realdepth/observation_spec.md v1 (Depth Pro 2-S magnitudes).

Each corruption (spec section 4, numbers 1-11) has a name in NAMES and can be switched off for
ablation. `stress` is per car: a stressed car uses the upper end of every range in the spec and
forces every per-episode/per-frame Bernoulli corruption on (windows, dilation, thin loss,
flattening); see README.md for the exact list.
"""

import math

import torch
from torch.nn import functional as F

NAMES = frozenset(
    "scale jitter shape flatten bleed thin noise windows upper failure latency".split()
)
Z_NEAR, Z_FAR = 0.15, 5.0


def encode(z: torch.Tensor) -> torch.Tensor:
    """Spec section 3, identical to realdepth.encode: inf / NaN / no hit -> far."""
    z = torch.where(torch.isfinite(z), z, Z_FAR).float().clamp(Z_NEAR, Z_FAR)
    return 1 - (z / Z_NEAR).log() / math.log(Z_FAR / Z_NEAR)


def decode(obs: torch.Tensor) -> torch.Tensor:
    """Inverse of encode (metres); 0 maps to 5 m."""
    return Z_NEAR * (Z_FAR / Z_NEAR) ** (1 - obs)


class DepthCorruptor:
    """Per-camera temporal state. Input z is optical-axis depth in metres at 160 x 120."""

    def __init__(self, count: int, device, enabled: frozenset[str], dt: torch.Tensor):
        if unknown := enabled - NAMES:
            raise KeyError(unknown)
        self.enabled, self.dt = enabled, dt
        base = torch.zeros(count, device=device)
        self.stress = torch.zeros(count, dtype=torch.bool, device=device)
        self.scale = torch.ones_like(base)
        self.jitter = torch.zeros_like(base)
        self.upper = torch.ones_like(base)
        self.burst = torch.zeros_like(base, dtype=torch.long)
        self.mode = self.burst.clone()
        self.near = self.scale.clone()
        self.window = torch.zeros((count, 2, 4), device=device)
        self.px = torch.arange(160, device=device)[None, None, None, :]
        self.py = torch.arange(120, device=device)[None, None, :, None]
        self.history = torch.zeros((count, 2, 120, 160), device=device)
        self.age = self.burst.clone()
        self.knoise = self.kernel(base, 1.5, 5)
        # Shape noise is drawn at 32 x 24 (5 px cells), so sigma 4 cells = 20 image pixels.
        self.kshape = self.kernel(base, 20 / 5, 12)
        self.edge_offsets = torch.arange(-3, 4, device=device, dtype=base.dtype)

    @staticmethod
    def kernel(like: torch.Tensor, sigma: float, radius: int) -> torch.Tensor:
        x = torch.arange(-radius, radius + 1, device=like.device, dtype=like.dtype)
        k = (-0.5 * (x / sigma).square()).exp()
        return k / k.sum()

    @staticmethod
    def blur(x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        r = k.numel() // 2
        x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), k[None, None, None, :])
        return F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), k[None, None, :, None])

    @staticmethod
    def rescale(x: torch.Tensor, sd: float) -> torch.Tensor:
        return (x - x.mean((-1, -2), keepdim=True)) * (
            sd / x.std((-1, -2), correction=0, keepdim=True).clamp_min(1e-6)
        )

    def reset(self, ids: torch.Tensor) -> None:
        """Per-episode draws (corruptions 1, 8, 9) and temporal state."""
        u = torch.rand((len(ids), 12), device=ids.device)
        stress = self.stress[ids]
        lo, hi = math.log(0.75), math.log(1.33)
        scale = torch.lerp(torch.full_like(u[:, 0], lo), torch.full_like(u[:, 0], hi), u[:, 0])
        self.scale[ids] = torch.where(stress, torch.where(u[:, 0] < 0.5, lo, hi), scale).exp()
        self.upper[ids] = torch.where(stress, torch.where(u[:, 1] < 0.5, 0.7, 1.4), 0.7 + 0.7 * u[:, 1])
        for i in range(2):
            a, b, c, d = u[:, 4 * i + 4:4 * i + 8].unbind(1)
            width = torch.where(stress, 60.0, 20 + 40 * c)
            height = 10 + 27 * d
            left, top = a * (160 - width), b * (60 - height)
            active = stress | ((u[:, 2] < 0.3) & ((u[:, 3] < 0.5) | (i == 0)))
            self.window[ids, i] = torch.stack((left, top, left + width * active, top + height), 1)
        self.jitter[ids] = 0
        self.burst[ids] = 0
        self.age[ids] = 0

    def __call__(self, z: torch.Tensor, world_height: torch.Tensor) -> torch.Tensor:
        n = len(z)
        enabled = self.enabled
        stress = self.stress
        valid = torch.isfinite(z) & (z > 0)
        z = torch.where(valid, z, 20.0).float().clamp_min(Z_NEAR)
        u = torch.rand((n, 10), device=z.device)
        image = lambda v: v[:, None, None]
        if "scale" in enabled:  # 1
            z = z * image(self.scale)
        if "jitter" in enabled:  # 2
            self.jitter = 0.5 * self.jitter + 0.087 * torch.randn_like(self.jitter)
            z = z * image(self.jitter.exp())
        if enabled & {"shape", "flatten", "failure"}:
            low = torch.randn((n, 1, 24, 32), device=z.device)
            low = F.interpolate(self.blur(low, self.kshape), size=z.shape[-2:],
                                mode="bilinear", align_corners=False)[:, 0]
            shape = self.rescale(low, 0.10).exp()
        if "shape" in enabled:  # 3
            z = z * shape
        if "flatten" in enabled:  # 4
            close = ((z < 0.6).float().mean((-1, -2)) > 0.6) & (stress | (u[:, 0] < 0.5))
            med = self.median(z)
            flat = image(med * (0.3 * torch.randn_like(med)).exp())
            z = torch.where(image(close), flat * shape if "shape" in enabled else flat, z)
        if enabled & {"bleed", "thin"}:
            inv = z.reciprocal()[:, None]
            if "bleed" in enabled:  # 5
                sigma = torch.where(stress, 1.0, 0.3 + 0.7 * u[:, 1])
                k = (-0.5 * (self.edge_offsets[None] / sigma[:, None]).square()).softmax(-1)
                for axis in (3, 2):
                    orient = (3, 3, 0, 0) if axis == 3 else (0, 0, 3, 3)
                    xp = F.pad(inv, orient, mode="reflect")
                    inv = sum(xp.narrow(axis, j, inv.shape[axis]) * k[:, j, None, None, None]
                              for j in range(7))
                dilated = F.max_pool2d(inv, 3, stride=1, padding=1)
                inv = torch.where((stress | (u[:, 2] < 0.5))[:, None, None, None], dilated, inv)
            if "thin" in enabled:  # 6: 2 x 2 opening of 1/z, partial contrast kept next to survivors
                erode = -F.max_pool2d(-F.pad(inv, (0, 1, 0, 1), mode="replicate"), 2, stride=1)
                opened = F.max_pool2d(F.pad(erode, (1, 0, 1, 0), mode="replicate"), 2, stride=1)
                supported = F.max_pool2d(opened, 3, stride=1, padding=1) >= 0.6 * inv
                keep_fraction = torch.where(stress, 0.5, 0.5 + 0.5 * u[:, 8])
                keep = torch.where(supported, keep_fraction[:, None, None, None], 0.0)
                thinned = opened + (inv - opened) * keep
                inv = torch.where((stress | (u[:, 3] < 0.7))[:, None, None, None], thinned, inv)
            z = inv[:, 0].reciprocal()
        if "noise" in enabled:  # 7
            noise = self.blur(torch.randn_like(z)[:, None], self.knoise)[:, 0]
            z = z * self.rescale(noise, 0.06).exp()
        if "windows" in enabled:  # 8: only on walls (not floor, not the generated ceiling)
            a, b, c, d = self.window[:, :, :, None, None].unbind(2)
            rect = ((self.px >= a) & (self.py >= b) & (self.px < c) & (self.py < d)).any(1)
            wall = valid & (world_height > 0.25) & (world_height < 2.25)
            z = torch.where(rect & wall, image(1.8 * self.median(z)), z)
        if "upper" in enabled:  # 9
            z = torch.where(world_height > 1.2, z * image(self.upper), z)
        z = torch.where(valid, z, torch.inf)
        if "failure" in enabled:  # 10: p = 0.03 per step, bursts of 1-3 steps (stress: always 3)
            start = (self.burst == 0) & (u[:, 4] < 0.03)
            steps = torch.where(stress, 3, 1 + (u[:, 5] * 3).long().clamp_max(2))
            self.burst = torch.where(start, steps, self.burst)
            self.mode = torch.where(start, (u[:, 6] * 3).long().clamp_max(2), self.mode)
            self.near = torch.where(start, 0.3 + 0.5 * u[:, 7], self.near)
            failed = torch.where(image(self.mode == 0), torch.inf,
                                 torch.where(image(self.mode == 1), image(self.near), shape**5))
            z = torch.where(image(self.burst > 0), failed, z)
            self.burst = (self.burst - 1).clamp_min(0)
        encoded = encode(z)
        if "latency" not in enabled:
            return encoded
        # 11: the observation is one 100 ms step old. dt is each car's control step: at 70 ms use the frame
        # from two steps back with p = 0.1/dt - 1 (= 3/7) and the previous frame otherwise (mean lag 100 ms);
        # at 100 ms always the previous frame.
        older_probability = (0.1 / self.dt - 1).clamp(0, 1)
        older = (u[:, 9] < older_probability) & (self.age > 1)
        out = torch.where(image(older), self.history[:, 1], self.history[:, 0])
        out = torch.where(image(self.age == 0), encoded, out)
        self.history[:, 1].copy_(self.history[:, 0])
        self.history[:, 0].copy_(encoded)
        self.age.add_(1)
        return out

    @staticmethod
    def median(z: torch.Tensor) -> torch.Tensor:
        # A fixed stratified 30 x 40 slice avoids sorting 19,200 values per car.
        return z[:, 2::4, 2::4].flatten(1).median(dim=1).values

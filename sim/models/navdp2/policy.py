"""Batched NavDP driving: CarObs-like tensors in, signed PWM out. Shared by driver.py (one car, Mac) and the
Isaac harness (many cars, L4) so both run the same code.

Per step: the newest frame goes into an RGB-token memory (upstream: a 16-frame queue, 7 frames sampled evenly
plus the current one); every `plan_every` steps the network samples `samples` trajectories toward the
command's target and the critic picks one; between plans a pure-pursuit tracker follows it on dead reckoning
(gyro heading + the calibrated speed of the last sent PWM).
"""

import torch
from torch.nn import functional as F

from xnavdp.net import MEMORY, XNavDP

SIZE = 224
HISTORY = 16
Z_NEAR, Z_FAR = 0.15, 5.0
# Nominal drive response, sim/isaac/elegoo/calibration.yaml (same model as sim/isaac/elegoo/drive.py).
SPEED_PER_PWM = 0.004
DEADBAND = 12.5
SPIN_PWM = (0.0, 12.5, 80.0, 95.0, 110.0, 145.0, 255.0)
SPIN_DEG = (0.0, 0.0, 130.0, 150.0, 200.0, 265.0, 350.0)
PWM_MAX = 150


def letterbox(images: torch.Tensor) -> torch.Tensor:
    """[B,C,H,W] -> [B,C,224,224]: scale the long side to 224 and zero-pad the short one (upstream's
    process_image / process_depth, cv2 bilinear)."""
    h, w = images.shape[-2:]
    scale = SIZE / max(h, w)
    nh, nw = round(h * scale), round(w * scale)
    x = F.interpolate(images, size=(nh, nw), mode="bilinear", align_corners=False)
    top, left = (SIZE - nh) // 2, (SIZE - nw) // 2
    return F.pad(x, (left, SIZE - nw - left, top, SIZE - nh - top))


def rgb_input(rgb: torch.Tensor) -> torch.Tensor:
    """uint8 [B,H,W,3] -> float [B,224,224,3] in 0..1."""
    return letterbox(rgb.permute(0, 3, 1, 2).float() / 255).permute(0, 2, 3, 1)


def depth_input(encoded: torch.Tensor) -> torch.Tensor:
    """observation_spec.md encoding [B,120,160] -> metres [B,224,224], 0 where far / no return (upstream sets
    inf, > 5 m and < 0.1 m to 0)."""
    z = Z_NEAR * (Z_FAR / Z_NEAR) ** (1 - encoded.float())
    z = torch.where(encoded > 1e-4, z, 0.0)
    z = letterbox(z[:, None])[:, 0]
    return torch.where((z > 5.0) | (z < 0.1), 0.0, z)


def interp(x: torch.Tensor, xs: tuple, ys: tuple) -> torch.Tensor:
    xs_t = torch.tensor(xs, device=x.device, dtype=x.dtype)
    ys_t = torch.tensor(ys, device=x.device, dtype=x.dtype)
    hi = torch.searchsorted(xs_t, x.contiguous()).clamp(1, len(xs) - 1)
    lo = hi - 1
    return torch.lerp(ys_t[lo], ys_t[hi], ((x - xs_t[lo]) / (xs_t[hi] - xs_t[lo])).clamp(0, 1))


def wheel_pwm(v: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    """Inverse of drive.py: forward speed v (m/s) and yaw rate omega (rad/s, CCW +) -> (left, right) PWM."""
    rolling = v.abs() > 1e-3
    mean = torch.where(rolling, v.sign() * (v.abs() / SPEED_PER_PWM + DEADBAND), 0.0)
    turn = omega.abs().rad2deg()
    slope = SPIN_DEG[2] / SPIN_PWM[2]
    diff = torch.where(rolling & (turn < SPIN_DEG[2]), turn / slope, interp(turn.clamp(max=SPIN_DEG[-1]), SPIN_DEG[1:], SPIN_PWM[1:]))
    diff = torch.where(turn < 1.0, 0.0, diff)
    diff = torch.where(rolling, torch.minimum(diff, mean.abs() - 1).clamp_min(0), diff)
    left, right = mean - omega.sign() * diff, mean + omega.sign() * diff
    pwm = torch.stack((left, right), 1)
    pwm = pwm * (PWM_MAX / pwm.abs().amax(1, keepdim=True).clamp_min(PWM_MAX))
    return pwm.round()


class Tracker:
    """Pure pursuit on the latest planned path, in the car frame of the frame it was planned from."""

    def __init__(self, batch: int, device, v_max: float = 0.40, lookahead: float = 0.30, dt: float = 0.1) -> None:
        self.v_max, self.lookahead, self.dt = v_max, lookahead, dt
        self.path = torch.zeros((batch, MEMORY * 3, 2), device=device)
        self.pose = torch.zeros((batch, 3), device=device)
        self.heading0 = torch.zeros(batch, device=device)
        self.valid = torch.zeros(batch, dtype=torch.bool, device=device)

    def reset(self, ids: torch.Tensor) -> None:
        self.valid[ids] = False
        self.pose[ids] = 0

    def set_path(self, ids: torch.Tensor, path: torch.Tensor, heading: torch.Tensor, lead: torch.Tensor) -> None:
        """path [n,24,2] from a frame `lead` = (forward m, left m, yaw rad) behind the car's current pose."""
        self.path[ids] = path.float()
        self.pose[ids] = lead.float()
        self.heading0[ids] = heading[ids] - lead[:, 2]
        self.valid[ids] = True

    def advance(self, heading: torch.Tensor, last_pwm: torch.Tensor) -> None:
        """Dead reckoning over the last step: heading from the gyro, speed from the PWM that was sent."""
        mean = last_pwm.float().mean(1)
        v = mean.sign() * (mean.abs() - DEADBAND).clamp_min(0) * SPEED_PER_PWM
        theta = torch.atan2((heading - self.heading0).sin(), (heading - self.heading0).cos())
        mid = (theta + self.pose[:, 2]) / 2
        self.pose[:, 0] += v * self.dt * mid.cos()
        self.pose[:, 1] += v * self.dt * mid.sin()
        self.pose[:, 2] = theta

    def act(self, speed_limit: torch.Tensor, range_cm: torch.Tensor) -> torch.Tensor:
        c, s = self.pose[:, 2].cos()[:, None], self.pose[:, 2].sin()[:, None]
        dx, dy = self.path[..., 0] - self.pose[:, None, 0], self.path[..., 1] - self.pose[:, None, 1]
        p = torch.stack((c * dx + s * dy, -s * dx + c * dy), -1)
        d = p.norm(dim=-1)
        index = torch.arange(p.shape[1], device=p.device)
        after = index[None] > d.argmin(1, keepdim=True)
        ahead = after & (d >= self.lookahead)
        pick = torch.where(ahead.any(1), ahead.float().argmax(1), torch.full_like(d[:, 0], p.shape[1] - 1).long())
        target = p[torch.arange(len(p), device=p.device), pick]
        dist = target.norm(dim=-1).clamp_min(1e-3)
        alpha = torch.atan2(target[:, 1], target[:, 0])
        remaining = p[:, -1].norm(dim=-1)
        behind = alpha.abs() > 2.0
        reverse = behind & (dist < 0.8)
        spin = (alpha.abs() > 0.9) & ~reverse
        v = self.v_max * speed_limit * (remaining / 0.5).clamp(0.35, 1.0)
        v = torch.where(reverse, -0.15 * torch.ones_like(v), torch.where(spin, 0.0, v))
        v = torch.where((range_cm < 8) & (v > 0), 0.0, v)
        omega = torch.where(spin, alpha.sign() * 1.8, v * 2 * target[:, 1] / dist.square()).clamp(-2.2, 2.2)
        stop = ~self.valid | (remaining < 0.06)
        pwm = wheel_pwm(v, omega)
        return torch.where(stop[:, None], 0.0, pwm)


class NavDPPolicy:
    def __init__(self, net: XNavDP, batch: int, samples: int = 8, plan_every: int = 1, generator=None, **tracker) -> None:
        self.net, self.samples, self.plan_every, self.generator = net, samples, plan_every, generator
        p = next(net.parameters())
        self.device, self.dtype = p.device, p.dtype
        self.tracker = Tracker(batch, self.device, **tracker)
        with torch.no_grad():
            self.black = net.rgbd_encoder.rgb_tokens(rgb_input(torch.zeros((1, 3, 4, 3), dtype=torch.uint8,
                                                                            device=self.device)).to(self.dtype))
        self.ring = self.black.expand(batch, HISTORY, -1, -1).clone()
        self.frame = self.black.expand(batch, -1, -1).clone()
        self.depth = torch.zeros((batch, SIZE, SIZE), device=self.device, dtype=self.dtype)
        self.count = torch.zeros(batch, dtype=torch.long, device=self.device)
        self.sample_index = torch.linspace(0, HISTORY - 1, MEMORY - 1).long().to(self.device)
        self.last_paths = None

    def reset(self, ids: torch.Tensor) -> None:
        self.ring[ids] = self.black
        self.frame[ids] = self.black[0]
        self.depth[ids] = 0
        self.count[ids] = 0
        self.tracker.reset(ids)

    @torch.no_grad()
    def observe(self, rgb: torch.Tensor | None, rgb_valid: torch.Tensor, depth: torch.Tensor | None,
                depth_valid: torch.Tensor) -> torch.Tensor:
        """Push this step's frame (the last good one on a dropped frame); returns the RGB memory
        [B, 8*256, 384] the network would see now."""
        if rgb is not None and bool(rgb_valid.any()):
            ids = rgb_valid.nonzero().squeeze(1)
            self.frame[ids] = self.net.rgbd_encoder.rgb_tokens(rgb_input(rgb[ids]).to(self.dtype))
        if depth is not None and bool(depth_valid.any()):
            ids = depth_valid.nonzero().squeeze(1)
            self.depth[ids] = depth_input(depth[ids]).to(self.dtype)
        memory = torch.cat((self.ring[:, self.sample_index], self.frame[:, None]), 1).flatten(1, 2)
        self.ring = torch.cat((self.ring[:, 1:], self.frame[:, None]), 1)
        return memory

    @torch.no_grad()
    def plan(self, ids: torch.Tensor, rgb_memory: torch.Tensor, target_xy: torch.Tensor):
        """Sample trajectories for cars `ids`; returns (best [n,24,3], all paths [n,S,24,3], q [n,S])."""
        memory = self.net.rgbd_encoder.memory(rgb_memory[ids], self.net.rgbd_encoder.depth_tokens(self.depth[ids]))
        paths, q = self.net.sample(memory, target_xy[ids].to(self.dtype), self.samples, self.generator)
        best = paths[torch.arange(len(ids), device=paths.device), q.argmax(1)]
        return best.float(), paths.float(), q.float()

    @torch.no_grad()
    def step(self, rgb, rgb_valid, depth, depth_valid, target_xy, speed_limit, heading, range_cm, last_pwm,
             rgb_age=None, gyro_rate=None) -> torch.Tensor:
        """One 100 ms control step for every car -> PWM [B,2] (float, integer-valued, |pwm| <= 150)."""
        memory = self.observe(rgb, rgb_valid, depth, depth_valid)
        self.last_memory = memory
        self.tracker.advance(heading, last_pwm)
        due = (self.count % self.plan_every) == 0
        self.last_due = due.nonzero().squeeze(1)
        if len(self.last_due):
            ids = self.last_due
            best, self.last_paths, self.last_q = self.plan(ids, memory, target_xy)
            lead = torch.zeros((len(ids), 3), device=self.device)
            if rgb_age is not None:
                mean = last_pwm[ids].float().mean(1)
                lead[:, 0] = mean.sign() * (mean.abs() - DEADBAND).clamp_min(0) * SPEED_PER_PWM * rgb_age[ids]
                if gyro_rate is not None:
                    lead[:, 2] = gyro_rate[ids] * rgb_age[ids]
            self.tracker.set_path(ids, best[..., :2], heading, lead)
        self.count += 1
        return self.tracker.act(speed_limit, range_cm)


def autocast_dtype(device: str) -> torch.dtype:
    return {"cuda": torch.bfloat16, "mps": torch.float16}.get(torch.device(device).type, torch.float32)


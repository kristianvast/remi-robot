"""DriverEnv in the visual building with the real car's randomisation drawn per episode (Isaac side only).

Composed on train/driver_env.py's DriverEnv (bumper ring, footprint, contact onsets) and nav/env.py's
BuildingNavEnv (visual scene, appearance randomisation, cone sonar). Per episode (draw_episode):
  - car response: every calibration.yaml parameter uniformly in its range (Drive draws them once per process);
  - camera mount: pitch 17-23 deg, height 0.121-0.131 m, forward 0.078-0.086 m on the lens prim, added to
    appearance.py's lens jitter; HFOV 60-70 deg by cropping a 70 deg render (tiled RTX shares one intrinsic);
  - sonar cone 25-35 deg (rays of a 35 deg cone outside the drawn half-angle are ignored);
  - gyro bias, noise and limit from calibration.yaml, plus a bias random walk (drift) every step.
privileged() and truth_* are for scoring only; the harness never hands them to a driver.
"""

from math import radians, tan

import torch
import yaml

from elegoo import appearance as appearance_module
from elegoo.drive import CALIBRATION
from elegoo.nav.env import ConePatternCfg, FLOOR_ECHO_Z, cone_pattern
from elegoo.train.driver_env import BUMPER_HEIGHTS, SONAR_X, DriverEnv, RunSpec
from elegoo.world import CameraSpec

TICKS = 10
RENDER_HFOV, RENDER_W, RENDER_H = 70.0, 720, 540
NOMINAL_PITCH, NOMINAL_HEIGHT, NOMINAL_FORWARD = 20.0, 0.126, 0.0825
SONAR_MAX_CONE = 35.0
GYRO_DRIFT_DEG_S_PER_SQRT_S = 0.01
GYRO_BIAS_LIMIT = 0.5


class MountedAppearance(appearance_module.Appearance):
    def place_lens(self, ids, angles, offset):
        super().place_lens(ids, angles + self.env.mount_angles[ids], offset + self.env.mount_offset[ids])


def cmdtest_spec(count: int, seed: int) -> RunSpec:
    camera = CameraSpec(outputs=("rgb",), width=RENDER_W, height=RENDER_H, pitch_deg=NOMINAL_PITCH,
                        hfov_deg=RENDER_HFOV, height_m=NOMINAL_HEIGHT, forward_m=NOMINAL_FORWARD, far_clip_m=20.0)
    return RunSpec(count=count, camera=camera, randomize=True, seed=seed, visual=True, visual_period=250,
                   split="test", noisy_command=False, step_ticks=(TICKS,), random_camera=False)


class CmdEnv(DriverEnv):
    def __init__(self, sim, spec: RunSpec):
        n, device = spec.count, sim.device
        self.mount_angles = torch.zeros((n, 3), device=device)
        self.mount_offset = torch.zeros((n, 3), device=device)
        self.hfov = torch.full((n,), 65.0, device=device)
        self.sonar_half = torch.full((n,), radians(15.0), device=device)
        self.ranges = yaml.safe_load(CALIBRATION.read_text())["parameters"]
        appearance_module_class = appearance_module.Appearance
        appearance_module.Appearance = MountedAppearance
        try:
            super().__init__(sim, spec)
        finally:
            appearance_module.Appearance = appearance_module_class
        _, directions = cone_pattern(ConePatternCfg(full_cone_deg=SONAR_MAX_CONE, rings=7), self.device)
        self.sonar_ray_angle = torch.acos(directions[:, 0].clamp(-1, 1))
        self.episode_rng = torch.Generator(device=self.device).manual_seed(spec.seed + 7919)

    def scene_config(self, camera):
        cfg = super().scene_config(camera)
        cfg.sonar.pattern_cfg = ConePatternCfg(full_cone_deg=SONAR_MAX_CONE, rings=7)
        return cfg

    def uniform(self, low, high, n):
        return low + (high - low) * torch.rand(n, generator=self.episode_rng, device=self.device)

    def draw_episode(self, ids: torch.Tensor) -> dict:
        n = len(ids)
        for name, (_, low, high) in self.ranges.items():
            self.drive.p[name][ids] = self.uniform(low, high, n)
        self.drive.delay_ticks[ids] = torch.ceil(self.drive.p["command_delay_s"][ids] / self.dt - 1e-5).long()
        self.drive.coast_tau[ids] = self.drive.p["yaw_accel_lag_s"][ids]
        pitch = self.drive.p["camera_pitch_deg"][ids]
        self.mount_angles[ids] = torch.stack((torch.deg2rad(pitch - NOMINAL_PITCH), torch.zeros_like(pitch),
                                              torch.zeros_like(pitch)), 1)
        self.mount_offset[ids] = torch.stack((self.drive.p["camera_forward_m"][ids] - NOMINAL_FORWARD,
                                              torch.zeros_like(pitch),
                                              self.drive.p["camera_height_m"][ids] - NOMINAL_HEIGHT), 1)
        self.hfov[ids] = self.drive.p["camera_hfov_deg"][ids]
        self.sonar_half[ids] = torch.deg2rad(self.drive.p["ultrasonic_full_cone_deg"][ids] / 2)
        keys = ("camera_pitch_deg", "camera_hfov_deg", "camera_height_m", "camera_forward_m",
                "ultrasonic_full_cone_deg", "speed_scale", "turn_scale", "deadband_pwm", "command_delay_s",
                "gyro_bias_deg_s", "gyro_noise_deg_s", "straight_yaw_bias_deg_s")
        return {k: [round(float(v), 4) for v in self.drive.p[k][ids]] for k in keys}

    def start_episode(self, ids: torch.Tensor, xy: torch.Tensor, yaw: torch.Tensor, seed: int) -> dict:
        self.episode_rng.manual_seed(seed)
        draws = self.draw_episode(ids)
        if self.appearance is not None:
            self.appearance.generator.manual_seed(seed)
        self.reset(ids)
        self.place(ids, xy, yaw)
        self.pan_target[ids] = 0
        self.touching[ids] = False
        self.stall_seconds[ids] = 0
        return draws

    def advance(self, action, ticks=None):
        super().advance(action, TICKS if ticks is None else ticks)
        step = self.dt * TICKS
        walk = torch.randn(len(self.ids), generator=self.episode_rng, device=self.device)
        bias = self.drive.p["gyro_bias_deg_s"] + walk * GYRO_DRIFT_DEG_S_PER_SQRT_S * step ** .5
        self.drive.p["gyro_bias_deg_s"].copy_(bias.clamp(-GYRO_BIAS_LIMIT, GYRO_BIAS_LIMIT))

    def sonar_cm(self) -> torch.Tensor:
        data = self.sonar.data
        hits = data.ray_hits_w.torch
        distance = (hits - data.pos_w.torch[:, None]).norm(dim=-1)
        distance = torch.where(hits[..., 2] > FLOOR_ECHO_Z, distance, torch.inf)
        distance = torch.where(self.sonar_ray_angle[None] <= self.sonar_half[:, None] + 1e-4, distance, torch.inf)
        nearest = distance.amin(dim=1).nan_to_num(posinf=10.0)
        cm = nearest * 100 + self.drive.p["ultrasonic_range_noise_cm"] * torch.randn_like(nearest)
        return cm.round().clamp(1, 150).to(torch.int32)

    def crop_frames(self, rgb: torch.Tensor) -> torch.Tensor:
        """[N,540,720,3] 70 deg renders -> [N,480,640,3] at each car's HFOV (centre crop, fx = fy kept)."""
        out = []
        for i in range(len(rgb)):
            frac = tan(radians(float(self.hfov[i])) / 2) / tan(radians(RENDER_HFOV) / 2)
            w, h = round(RENDER_W * frac), round(RENDER_H * frac)
            x0, y0 = (RENDER_W - w) // 2, (RENDER_H - h) // 2
            crop = rgb[i, y0:y0 + h, x0:x0 + w].permute(2, 0, 1)[None].float()
            out.append(torch.nn.functional.interpolate(crop, size=(480, 640), mode="bilinear",
                                                       antialias=True, align_corners=False)[0])
        return torch.stack(out).clamp(0, 255).round().to(torch.uint8).permute(0, 2, 3, 1).contiguous()

    def truth(self) -> dict:
        """Scoring truth: pose, bumper gap, gap ahead of the sonar emitter, tilt, stall, received PWM."""
        p = self.privileged()
        data = self.bumper.data
        hits = data.ray_hits_w.torch
        reach = (hits[..., :2] - data.pos_w.torch[:, None, :2]).norm(dim=-1).nan_to_num(nan=torch.inf)
        phi = self.ray_angle.repeat(len(BUMPER_HEIGHTS))[None] - p["heading_rad"][:, None]
        phi = torch.atan2(phi.sin(), phi.cos())
        gap = reach - self.body_reach(phi)
        ahead = torch.where(phi.abs() <= radians(15), gap, torch.inf).amin(1) + (self.footprint[1] - SONAR_X)
        return {"x": p["pos_w"][:, 0], "y": p["pos_w"][:, 1], "yaw": p["heading_rad"], "bumper": gap.amin(1),
                "ahead_m": ahead, "tilted": p["tilted"], "stall_s": self.stall_seconds.clone(),
                "pwm": self.drive.pwm.clone(), "pan": self.robot.data.joint_pos.torch[:, self.pan_ids[0]]}


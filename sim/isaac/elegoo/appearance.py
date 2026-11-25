"""Appearance randomisation for the RGB camera in the visual building (scene/building_visual.usda).

The building is one prim shared by every car, so its looks change for all cars at once: draw_scene() tints the
generated shell's plain colour slots and the scan's own atlas, and sets lamp,
dome and ambient intensity and colour temperature. The filmed DA3 rooms always show as filmed. It runs at start-up and every
`period` control steps. Per car, at every reset, reset() draws a camera tilt/roll/yaw and position jitter (the
lens prim's local pose) and image-space exposure, white balance, saturation, gamma, blur and noise, which
augment() applies to the rendered frames. neutral() and draw_scene(canonical=True) give the un-randomised view.
"""

import json
import math
from pathlib import Path

import torch

SCENE_DIR = Path(__file__).resolve().parents[1] / "scene"
PROJECT_ROOT = SCENE_DIR.parents[2]
VISUAL_META = SCENE_DIR / "visual_meta.json"
AMBIENT_SETTING = "/rtx/sceneDb/ambientLightIntensity"
BLUR_TAPS = 7


def log_uniform(rng, low, high):
    return math.exp(rng.uniform(math.log(low), math.log(high)))


class Appearance:
    def __init__(self, env, root="/World/Building", seed=0, period=250):
        import omni.usd
        import random

        self.env, self.period = env, period
        self.rng = random.Random(seed)
        self.generator = torch.Generator(device=env.device).manual_seed(seed)
        self.stage = omni.usd.get_context().get_stage()
        self.meta = json.loads(VISUAL_META.read_text())
        self.path = lambda p: root + p[len("/World"):]
        self.steps = 0
        self.last_draw = {}
        n, device = env.spec.count, env.device
        self.gain = torch.ones((n, 3), device=device)
        self.saturation = torch.ones(n, device=device)
        self.gamma = torch.ones(n, device=device)
        self.blur = torch.zeros(n, device=device)
        self.noise = torch.zeros(n, device=device)
        view = env.camera._view
        translations, orientations = view.get_local_poses()
        self.base_pos = translations.torch.clone()
        self.base_quat = orientations.torch.clone()
        self.draw_scene()

    def shader(self, path):
        return self.stage.GetPrimAtPath(self.path(path))

    def tint(self, spread=0.12, brightness=(0.75, 1.2)):
        level = self.rng.uniform(*brightness)
        return tuple(level * (1 + self.rng.uniform(-spread, spread)) for _ in range(3))

    def draw_scene(self, canonical=False):
        import carb
        from pxr import Gf, Sdf

        rng = self.rng
        record = {"canonical": canonical}
        with Sdf.ChangeBlock():
            for look in self.meta["slots"] + self.meta["scan_looks"]:
                tint = (1.0, 1.0, 1.0) if canonical else self.tint(0.08, (0.8, 1.15))
                self.shader(look["shader"]).GetAttribute("inputs:diffuse_tint").Set(Gf.Vec3f(*tint))
        level = 1.0 if canonical else log_uniform(rng, 0.35, 1.8)
        kelvin = self.meta["light_temperature_k"] if canonical else rng.uniform(2700, 6500)
        with Sdf.ChangeBlock():
            for lamp in self.meta["lights"]:
                on = canonical or rng.random() > 0.15
                factor = 1.0 if canonical else rng.uniform(0.6, 1.4)
                prim = self.stage.GetPrimAtPath(self.path(lamp["path"]))
                prim.GetAttribute("inputs:intensity").Set(float(lamp["intensity"] * level * factor * on))
                prim.GetAttribute("inputs:colorTemperature").Set(float(kelvin))
            dome = self.stage.GetPrimAtPath(self.path(self.meta["dome"]["path"]))
            dome.GetAttribute("inputs:intensity").Set(
                float(self.meta["dome"]["intensity"] * (1.0 if canonical else log_uniform(rng, 0.3, 2.0))))
            dome.GetAttribute("inputs:colorTemperature").Set(float(kelvin))
        ambient = 4.0 if canonical else log_uniform(rng, 1.0, 8.0)
        carb.settings.get_settings().set_float(AMBIENT_SETTING, ambient)
        record.update(light_level=round(level, 3), kelvin=round(kelvin), ambient=round(ambient, 2))
        self.last_draw = record
        return record

    def reset(self, ids):
        n, g, device = len(ids), self.generator, self.env.device
        u = torch.rand((n, 12), generator=g, device=device)
        temperature = u[:, 0] * 2 - 1
        self.gain[ids] = (2 ** (u[:, 1:2] * 1.4 - 0.8)) * torch.stack(
            (1 + 0.12 * temperature, 1 + 0.08 * (u[:, 2] - 0.5), 1 - 0.12 * temperature), 1)
        self.saturation[ids] = 0.6 + 0.7 * u[:, 3]
        self.gamma[ids] = 0.85 + 0.35 * u[:, 4]
        self.blur[ids] = torch.where(u[:, 5] < 0.6, 1.5 * u[:, 6], 0.0)
        self.noise[ids] = 0.035 * u[:, 7]
        angles = torch.stack(((u[:, 8] - 0.5) * 4.0, (u[:, 9] - 0.5) * 3.0, (u[:, 10] - 0.5) * 3.0), 1)
        offset = (torch.rand((n, 3), generator=g, device=device) - 0.5) * 0.016
        self.place_lens(ids, torch.deg2rad(angles), offset)

    def neutral(self, ids):
        self.gain[ids], self.saturation[ids], self.gamma[ids] = 1.0, 1.0, 1.0
        self.blur[ids], self.noise[ids] = 0.0, 0.0
        zero = torch.zeros((len(ids), 3), device=self.env.device)
        self.place_lens(ids, zero, zero)

    def place_lens(self, ids, angles, offset):
        """angles: (tilt about the lens x, yaw about y, roll about z) in the camera prim's own frame, radians."""
        import warp as wp
        from isaaclab.utils.math import quat_from_euler_xyz, quat_mul

        delta = quat_from_euler_xyz(angles[:, 0], angles[:, 1], angles[:, 2])
        quat = quat_mul(self.base_quat[ids], delta)
        pos = self.base_pos[ids] + offset
        self.env.camera._view.set_local_poses(wp.from_torch(pos.contiguous()), wp.from_torch(quat.contiguous()),
                                              wp.from_torch(ids.to(torch.int32).contiguous()))

    def tick(self):
        self.steps += 1
        if self.period and self.steps % self.period == 0:
            self.draw_scene()

    def augment(self, rgb):
        """[N,H,W,3] uint8 frames -> the same with each car's exposure, white balance, saturation, gamma,
        Gaussian blur and sensor noise."""
        x = rgb.permute(0, 3, 1, 2).float() / 255
        n, _c, h, w = x.shape
        if bool((self.blur > 0).any()):
            t = torch.arange(BLUR_TAPS, device=x.device) - BLUR_TAPS // 2
            sigma = self.blur.clamp_min(1e-3)[:, None]
            k = torch.exp(-0.5 * (t[None] / sigma) ** 2)
            k = (k / k.sum(1, keepdim=True)).repeat_interleave(3, 0)
            flat = x.reshape(1, n * 3, h, w)
            flat = torch.nn.functional.conv2d(flat, k[:, None, None, :], padding=(0, BLUR_TAPS // 2), groups=n * 3)
            flat = torch.nn.functional.conv2d(flat, k[:, None, :, None], padding=(BLUR_TAPS // 2, 0), groups=n * 3)
            x = flat.reshape(n, 3, h, w)
        x = x * self.gain[:, :, None, None]
        luma = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2])[:, None]
        x = luma + (x - luma) * self.saturation[:, None, None, None]
        x = x.clamp(0, 1) ** self.gamma[:, None, None, None]
        x = x + torch.randn(x.shape, generator=self.generator, device=x.device) * self.noise[:, None, None, None]
        return (x.clamp(0, 1) * 255).round().to(torch.uint8).permute(0, 2, 3, 1).contiguous()

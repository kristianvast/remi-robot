"""Isaac side of the combined model: DriverEnv + truth labels, the simulated camera link, the v2 teacher, the student.

The student only ever sees what DRIVER_API.md's CarObs + DriveCommand carry: the rendered RGB frame after
appearance randomisation, motion blur and a JPEG round trip, delayed 0.10-0.25 s and dropped on 11-14 % of
frames; sonar missing on > 50 % of steps; the drifting gyro; last PWM; the navigator's heading command and
target / goal points from a noisy pose estimate. The teacher (sim/policy/driver_v2.onnx) gets the clean
ray-cast depth and a noise-free lookahead command from the true pose.
"""

import math
from pathlib import Path

import numpy as np
import torch

from ..train.driver_env import DriverEnv
from .common import (AGE_NONE, AREAS, FRAME, HISTORY, IMG_H, IMG_W, Encoder, Heads, encode, jpeg_roundtrip,
                     motion_blur, pack_aux, wheels)

HERE = Path(__file__).resolve().parent
FX_RENDER_PER_PX = 1 / math.tan(math.radians(65) / 2)
AREA07_RIDGE = (-0.05, -7.35, 0.6, -6.7)
RING = 4
LINK_CHUNK = 64


class CombinedEnv(DriverEnv):
    def __init__(self, sim, spec):
        super().__init__(sim, spec)
        self.frame_rgb = None
        if self.appearance is not None:
            augment = self.appearance.augment

            def keep(rgb):
                self.frame_rgb = augment(rgb)
                return self.frame_rgb

            self.appearance.augment = keep
        with np.load(HERE / "labels.npz") as d:
            self.area_map = torch.as_tensor(d["area"].astype(np.int64), device=self.device)
            self.angle_map = torch.as_tensor(d["angle"].astype(np.float32), device=self.device)
            self.goal_ok_map = torch.as_tensor(np.isfinite(d["dist"]) & (d["dist"] > 0.3), device=self.device)
            self.label_origin = (float(d["x0"]), float(d["y0"]), float(d["res"]))
        n = spec.count
        self.teacher_index = torch.zeros(n, dtype=torch.long, device=self.device)
        self.est_offset = torch.zeros((n, 2), device=self.device)
        self.est_yaw = torch.zeros(n, device=self.device)

    def reset(self, ids=None, routes=None):
        super().reset(ids, routes)
        if hasattr(self, "teacher_index"):
            self.reset_estimate(self.ids if ids is None else ids)

    def reset_estimate(self, ids):
        """Pose-estimate error of the tool layer (localiser / dead reckoning): mostly a few cm, 10 % unsure."""
        n = len(ids)
        u = torch.rand((n, 3), device=self.device)
        size = torch.where(u[:, 0] < 0.1, 0.3 + 0.5 * u[:, 1], 0.05 * torch.randn(n, device=self.device).abs())
        self.est_offset[ids] = size[:, None] * torch.stack(((u[:, 2] * 6.283).cos(), (u[:, 2] * 6.283).sin()), 1)
        self.est_yaw[ids] = math.radians(2) * torch.randn(n, device=self.device)
        self.teacher_index[ids] = 0

    def truth(self) -> dict[str, torch.Tensor]:
        p = self.privileged()
        xy, yaw = p["pos_w"][:, :2], p["heading_rad"]
        x0, y0, res = self.label_origin
        r = ((y0 - xy[:, 1]) / res).round().long().clamp(0, self.area_map.shape[0] - 1)
        c = ((xy[:, 0] - x0) / res).round().long().clamp(0, self.area_map.shape[1] - 1)
        rel = self.angle_map[:, r, c].T - yaw[:, None]
        return {"xy": xy, "yaw": yaw, "room": self.area_map[r, c],
                "pose": torch.stack((xy[:, 0] / 10, xy[:, 1] / 10, yaw.sin(), yaw.cos()), 1),
                "goal": torch.stack((rel.sin(), rel.cos()), -1), "goal_ok": self.goal_ok_map[:, r, c].T.float()}

    def teacher_aux(self, aux: torch.Tensor, xy: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
        """Noise-free navigator command from the true pose, every step (DriverEnv.command without its noise)."""
        route, final = self.route, self.counts[self.route] - 1
        candidates = torch.minimum(self.teacher_index[:, None] + torch.arange(10, device=self.device), final[:, None])
        points = self.way[route[:, None], candidates]
        self.teacher_index = candidates.gather(1, (points - xy[:, None]).square().sum(-1).argmin(1)[:, None])[:, 0]
        path_idx = torch.minimum(self.teacher_index[:, None] + torch.arange(1, 13, device=self.device), final[:, None])
        relative = self.way[route[:, None], path_idx] - xy[:, None]
        unit = torch.nn.functional.normalize(relative[:, 2], dim=1)
        cross = (relative[..., 0] * unit[:, 1, None] - relative[..., 1] * unit[:, 0, None]).abs()
        choice = ((cross < 0.10).cumprod(1).sum(1) - 1).clamp(2, 11)
        delta = relative[self.ids, choice]
        err = torch.atan2(delta[:, 1], delta[:, 0]) - yaw
        d = delta.norm(dim=1)
        level = (d >= 0.5).float() * 0.5 + (d > 2.0).float() * 0.5
        return torch.cat((aux[:, :4], torch.stack((err.sin(), err.cos(), level, self.limit), 1)), 1)

    def command_fields(self, aux: torch.Tensor, xy: torch.Tensor, yaw: torch.Tensor) -> dict[str, torch.Tensor]:
        """DriveCommand as the tool layer would hand it over (heading from the navigator, points from the estimate)."""
        self.est_offset += 0.004 * torch.randn_like(self.est_offset)
        he = torch.atan2(aux[:, 4], aux[:, 5])
        target = self.waypoint_distance[:, None] * torch.stack((he.cos(), he.sin()), 1)
        est_yaw = yaw + self.est_yaw
        rel = self.goal_xy - (xy + self.est_offset)
        c, s = est_yaw.cos(), est_yaw.sin()
        goal = torch.stack((c * rel[:, 0] + s * rel[:, 1], -s * rel[:, 0] + c * rel[:, 1]), 1)
        return {"heading_error": he, "how_far": (aux[:, 6] * 2).round(), "speed_limit": aux[:, 7],
                "target_xy": target, "goal_xy": goal}


class V2:
    """sim/policy/driver_v2.onnx rebuilt in torch for a batch of cars on the GPU (same graph: train/model.py
    DriverModel = pack, (x - mean) / div, 3-layer ELU MLP, wheels); checked against onnxruntime at start-up."""

    def __init__(self, path: Path, count: int, device):
        import onnx
        from onnx import numpy_helper

        w = {i.name: torch.as_tensor(numpy_helper.to_array(i).copy(), device=device)
             for i in onnx.load(str(path)).graph.initializer}
        self.mean, self.div = w["normalizer._mean"], w["onnx::Div_103"]
        self.layers = [(w[f"mlp.{k}.weight"], w[f"mlp.{k}.bias"]) for k in (0, 2, 4)]
        self.history = torch.zeros((count, 3, 80), device=device)
        self.device = device
        self.check(path)

    def forward(self, depth, aux, history):
        from ..train.model import pack, wheels

        vector, next_history = pack(depth, aux, history)
        x = (vector - self.mean) / self.div
        for i, (weight, bias) in enumerate(self.layers):
            x = torch.nn.functional.linear(x, weight, bias)
            x = torch.nn.functional.elu(x) if i < 2 else x
        return wheels(x, aux), next_history

    def check(self, path: Path) -> None:
        import onnxruntime as ort

        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        g = torch.Generator().manual_seed(0)
        depth, aux = torch.rand((8, 1, 120, 160), generator=g), torch.rand((8, 8), generator=g) * 2 - 1
        history = torch.rand((8, 3, 80), generator=g)
        with torch.no_grad():
            mine, _ = self.forward(depth.to(self.device), aux.to(self.device), history.to(self.device))
        ref = np.concatenate([session.run(None, {"depth": depth[i:i + 1].numpy(), "aux": aux[i:i + 1].numpy(),
                                                 "history": history[i:i + 1].numpy()})[0] for i in range(8)])
        err = float(np.abs(mine.cpu().numpy() - ref).max())
        print(f"V2_TORCH_PARITY max_abs={err:.2e}", flush=True)
        if err > 1e-4:
            raise RuntimeError(f"torch rebuild of {path} differs from onnxruntime by {err}")

    def reset(self, ids: torch.Tensor) -> None:
        self.history[ids] = 0

    @torch.no_grad()
    def __call__(self, depth: torch.Tensor, aux: torch.Tensor, ids: torch.Tensor | None = None) -> torch.Tensor:
        action, history = self.forward(depth, aux, self.history)
        if ids is None:
            self.history = history
        else:
            self.history[ids] = history[ids]
        return action


class Link:
    """Per car: camera link draws, the capture ring, the delivered-frame history the heads see."""

    def __init__(self, count: int, device, stress: torch.Tensor | None = None, keep_rgb: bool = False,
                 real_path: bool = False):
        self.n, self.device = count, device
        self.real_path = real_path
        self.path_stats = []
        self.stress = torch.zeros(count, dtype=torch.bool, device=device) if stress is None else stress
        z = torch.zeros(count, device=device)
        self.latency, self.drop, self.sonar_drop, self.quality, self.exposure = z.clone(), z.clone(), z.clone(), z.clone(), z.clone()
        self.step = torch.zeros(count, dtype=torch.long, device=device)
        self.ring_feat = torch.zeros((count, RING, FRAME), device=device)
        self.ring_depth = torch.zeros((count, RING, 120, 160), dtype=torch.float16, device=device)
        self.ring_ok = torch.zeros((count, RING), dtype=torch.bool, device=device)
        self.keep_rgb = keep_rgb
        if keep_rgb:
            self.ring_rgb = torch.zeros((count, RING, 3, 120, 160), dtype=torch.uint8, device=device)
            self.delivered_rgb = torch.zeros((count, 3, 120, 160), dtype=torch.uint8, device=device)
        self.history = torch.zeros((count, HISTORY, FRAME), device=device)
        self.ages = torch.full((count, HISTORY), AGE_NONE, device=device)
        self.delivered_depth = torch.zeros((count, 120, 160), device=device)
        self.delivered = torch.zeros(count, dtype=torch.bool, device=device)
        self.last_dropped = torch.zeros(count, dtype=torch.bool, device=device)
        self.reset(torch.arange(count, device=device))

    def reset(self, ids: torch.Tensor) -> None:
        u = torch.rand((len(ids), 5), device=self.device)
        s = self.stress[ids]
        self.latency[ids] = torch.where(s, 0.25, 0.10 + 0.15 * u[:, 0])
        self.drop[ids] = torch.where(s, 0.14, 0.11 + 0.03 * u[:, 1])
        self.sonar_drop[ids] = torch.where(s, 0.75, 0.50 + 0.15 * u[:, 2])
        self.quality[ids] = torch.where(s, 60.0, (60 + 31 * u[:, 3]).floor())
        self.exposure[ids] = torch.where(s, 0.04, 0.01 + 0.02 * u[:, 4])
        self.step[ids] = 0
        self.ring_ok[ids] = False
        self.history[ids] = 0
        self.ages[ids] = AGE_NONE
        self.last_dropped[ids] = False

    @torch.no_grad()
    def capture(self, encoder: Encoder, rgb_hwc: torch.Tensor, gyro_rate: torch.Tensor, lap=lambda name: None) -> None:
        """Encode this step's rendered frames (after the camera / JPEG path) into the capture ring."""
        burst = self.last_dropped & self.stress
        dropped = torch.rand(self.n, device=self.device) < torch.where(burst, 0.4, self.drop)
        self.last_dropped = dropped
        slot = self.step % RING
        cars = torch.arange(self.n, device=self.device)
        self.ring_ok[cars, slot] = ~dropped
        # Every car is encoded (dropped frames are discarded afterwards) so the batch shape never changes and
        # cudnn.benchmark keeps its tuned kernels; a varying batch re-tunes every step (4x slower).
        kept = ~dropped
        dtype = next(encoder.parameters()).dtype
        blur = gyro_rate.abs() * self.exposure
        imgs, real_imgs = [], []
        for s in range(0, self.n, LINK_CHUNK):
            part = slice(s, s + LINK_CHUNK)
            full = camera_frame(rgb_hwc[part], blur[part])
            imgs.append(train_link(full, self.quality[part]))
            if self.real_path:
                real_imgs.append(real_link(full, self.quality[part]))
        img = torch.cat(imgs)
        lap("enc_link")
        if self.real_path:
            sim_img, img = img, torch.cat(real_imgs)
            sim_depth, sim_feat = encoder(sim_img.to(dtype))
            depth, feat = encoder(img.to(dtype))
            self.path_stats.append(torch.stack((
                torch.nn.functional.cosine_similarity(feat.float(), sim_feat.float(), dim=1),
                (depth.float() - sim_depth.float()).abs().mean((1, 2)),
                (img - sim_img).abs().mean((1, 2, 3))), 1)[kept].cpu())
        else:
            depth, feat = encoder(img.to(dtype))
        lap("enc_da2")
        keep = kept[:, None]
        self.ring_feat[cars, slot] = torch.where(keep, feat.float(), self.ring_feat[cars, slot])
        self.ring_depth[cars, slot] = torch.where(keep[:, :, None], depth.half(), self.ring_depth[cars, slot])
        if self.keep_rgb:
            small = torch.nn.functional.interpolate(img, size=(120, 160), mode="area")
            self.ring_rgb[cars, slot] = small.round().clamp(0, 255).to(torch.uint8)

    def deliver(self, dt: torch.Tensor) -> None:
        """Hand the frame captured `latency` ago (if it was not dropped) to the history; age the rest."""
        k = (self.latency / dt).round().long().clamp(1, RING - 1)
        src = self.step - k
        slot = src % RING
        cars = torch.arange(self.n, device=self.device)
        ok = (src >= 0) & self.ring_ok[cars, slot]
        self.ages = torch.where(self.ages < AGE_NONE, (self.ages + dt[:, None]).clamp_max(AGE_NONE - 1e-3), self.ages)
        age = self.latency + 0.02 * (torch.rand(self.n, device=self.device) - 0.5)
        shifted = torch.cat((self.ring_feat[cars, slot][:, None], self.history[:, :-1]), 1)
        shifted_age = torch.cat((age[:, None], self.ages[:, :-1]), 1)
        self.history = torch.where(ok[:, None, None], shifted, self.history)
        self.ages = torch.where(ok[:, None], shifted_age, self.ages)
        self.delivered = ok
        self.delivered_depth = torch.where(ok[:, None, None], self.ring_depth[cars, slot].float(), self.delivered_depth)
        if self.keep_rgb:
            self.delivered_rgb = torch.where(ok[:, None, None, None], self.ring_rgb[cars, slot], self.delivered_rgb)
        self.step += 1

    def sonar(self, range_cm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ok = (torch.rand(self.n, device=self.device) >= self.sonar_drop).float()
        return range_cm, ok


def camera_frame(rgb_hwc: torch.Tensor, blur_rad: torch.Tensor) -> torch.Tensor:
    """Rendered frames -> the car camera's 640 x 480 image before compression: upsampled if rendered smaller,
    horizontal motion blur of the yaw swept during the exposure (blur_rad x focal length in px)."""
    x = rgb_hwc.permute(0, 3, 1, 2).float()
    if x.shape[2:] != (480, 640):
        x = torch.nn.functional.interpolate(x, size=(480, 640), mode="bilinear", align_corners=False)
    return motion_blur(x, blur_rad * 320 * FX_RENDER_PER_PX, taps=31)


def train_link(full: torch.Tensor, quality: torch.Tensor) -> torch.Tensor:
    """Training path: GPU JPEG round trip at 640 x 480, then area downsample to the encoder size."""
    x = jpeg_roundtrip(full, quality)
    return torch.nn.functional.interpolate(x, size=(IMG_H, IMG_W), mode="area")


def real_link(full: torch.Tensor, quality: torch.Tensor) -> torch.Tensor:
    """driver.py's path on the real car's input: libjpeg encode/decode (cv2) at 640 x 480, cv2.INTER_AREA resize."""
    import cv2

    frames = full.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    out = []
    for frame, q in zip(frames, quality.tolist()):
        ok, buf = cv2.imencode(".jpg", frame[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, int(q)])
        decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., ::-1]
        out.append(cv2.resize(decoded, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA))
    return torch.as_tensor(np.stack(out), device=full.device).permute(0, 3, 1, 2).float()


def student_inputs(env: CombinedEnv, link: Link, aux: torch.Tensor, xy, yaw) -> torch.Tensor:
    cmd = env.command_fields(aux, xy, yaw)
    range_cm, ok = link.sonar(aux[:, 0] * 150)
    newest = link.ages[:, 0]
    return pack_aux(range_cm, ok, torch.deg2rad(aux[:, 1] * 250), env.last[:, 0] * 255, env.last[:, 1] * 255,
                    cmd["heading_error"], cmd["how_far"], cmd["speed_limit"], cmd["target_xy"], cmd["goal_xy"], newest)


def load_heads(path: Path | None, device, dropout: float = 0.0) -> Heads:
    heads = Heads(dropout=dropout).to(device)
    if path is not None:
        heads.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return heads


def act(heads: Heads, link: Link, aux_s: torch.Tensor):
    with torch.no_grad():
        out = heads(link.history, link.ages, aux_s)
    return wheels(out[0], aux_s), out


__all__ = ["AREA07_RIDGE", "AREAS", "CombinedEnv", "Link", "V2", "act", "encode", "load_heads", "student_inputs"]

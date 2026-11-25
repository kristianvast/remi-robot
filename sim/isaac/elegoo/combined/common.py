"""Combined camera model (p4 option C): DA2-S encoder + driving / place / goal-direction heads.

Pure torch (plus transformers for the encoder), used by training and evaluation on the L4 and by export.py.
The Mac driver (sim/models/combined/driver.py) runs the exported ONNX graphs and repeats pack_aux() and
the frame-history bookkeeping in numpy; keep the two in step.

Per new camera frame (encoder graph):
  image  float32 [B,3,168,224] RGB 0..255 (640x480 JPEG resized with area interpolation)
  -> depth  [B,120,160] observation_spec encoding (DA2 Metric-Indoor-Small metres, inverse-depth area resize)
  -> frame  [B,FRAME] = CLS token + 2x4 mean-pooled last-layer patch tokens (9 x 384) + depth summary (80)
Per control step (heads graph):
  history [B,4,FRAME] newest first (zeros where no frame yet), ages [B,4] s since capture (AGE_NONE if empty),
  aux [B,AUX] from CarObs + DriveCommand (pack_aux)
  -> residual [B,2] on the heading prior (wheels), room logits [B,10], pose [B,6] = (x/10, y/10, sin, cos,
     log sd xy (m), log sd yaw (rad)), goal [B,10,2] = (sin, cos) of the direction to each place, car frame.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

IMG_W, IMG_H = 224, 168
DEPTH_W, DEPTH_H = 160, 120
TOKENS = 384
GRID = (2, 4)
ROWS, COLS = 4, 10
SUMMARY = 2 * ROWS * COLS
FRAME = TOKENS * (1 + GRID[0] * GRID[1]) + SUMMARY
HISTORY = 4
AGE_NONE = 2.0
AUX = 15
AREAS = tuple(f"area_{i:02d}" for i in range(1, 11))
PWM_UNIT = 150 / 255
Z_NEAR, Z_FAR = 0.15, 5.0
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
HF_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
HF_REVISION = "8078d68a9c75a972131914f6afd0c1723be0da7f"


def encode(z: torch.Tensor) -> torch.Tensor:
    z = torch.where(torch.isfinite(z), z, Z_FAR).clamp(Z_NEAR, Z_FAR)
    return 1 - (z / Z_NEAR).log() / math.log(Z_FAR / Z_NEAR)


def summary(depth: torch.Tensor) -> torch.Tensor:
    """train/model.py's 80-number depth summary: nearest and mean over a 4 x 10 grid. depth [B,120,160]."""
    d = depth[:, None]
    cells = (DEPTH_H // ROWS, DEPTH_W // COLS)
    return torch.cat((F.max_pool2d(d, cells).flatten(1), F.avg_pool2d(d, cells).flatten(1)), 1)


def area_matrix(n_in: int, n_out: int) -> torch.Tensor:
    """[n_out, n_in] averaging weights of adaptive average pooling along one axis (PyTorch's bin edges), so a
    fixed-size area resize is two matmuls and exports to ONNX."""
    m = torch.zeros(n_out, n_in)
    for i in range(n_out):
        start, end = (i * n_in) // n_out, -((-(i + 1) * n_in) // n_out)
        m[i, start:end] = 1.0 / (end - start)
    return m


class Encoder(nn.Module):
    """DA2 Metric-Indoor-Small: one backbone pass gives the frame feature and the metric depth."""

    def __init__(self, model):
        super().__init__()
        self.m = model
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1) * 255)
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1) * 255)
        self.register_buffer("rows", area_matrix(IMG_H, DEPTH_H))
        self.register_buffer("cols", area_matrix(IMG_W, DEPTH_W))

    @classmethod
    def pretrained(cls, path: str = HF_ID):
        from transformers import AutoModelForDepthEstimation

        kw = {} if "/" not in path or path.startswith("/") else {"revision": HF_REVISION}
        model = AutoModelForDepthEstimation.from_pretrained(path, **kw).eval()
        # The DepthAnything wrapper refuses sdpa, but its DINOv2 backbone dispatches on its own config.
        model.backbone.config._attn_implementation = "sdpa"
        return cls(model)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = (image - self.mean.to(image.dtype)) / self.std.to(image.dtype)
        ph, pw = x.shape[2] // 14, x.shape[3] // 14
        maps = self.m.backbone.forward_with_filtered_kwargs(x, output_hidden_states=False,
                                                             output_attentions=False).feature_maps
        z = self.m.head(self.m.neck(maps, ph, pw), ph, pw)  # [B,H,W] metres
        inv = self.rows.float() @ (1.0 / z.float().clamp_min(Z_NEAR)) @ self.cols.float().T
        depth = encode(1.0 / inv.clamp_min(1e-6))
        last = maps[-1].float()
        cls_token = last[:, 0]
        patches = last[:, 1:].reshape(-1, ph, pw, TOKENS).permute(0, 3, 1, 2)
        grid = F.avg_pool2d(patches, (ph // GRID[0], pw // GRID[1])).flatten(2).transpose(1, 2).flatten(1)
        return depth, torch.cat((cls_token, grid, summary(depth)), 1)


class Heads(nn.Module):
    def __init__(self, width: int = 256, dropout: float = 0.0):
        """dropout (train mode only; no parameters, so state dicts and ONNX graphs are unchanged): on the projected
        frame features and after each trunk layer, against the place head memorising training start cells (p7)."""
        super().__init__()
        self.frame = nn.Sequential(nn.LayerNorm(FRAME), nn.Linear(FRAME, width), nn.GELU())
        trunk_in = HISTORY * (width + 2) + AUX
        self.trunk = nn.Sequential(nn.Linear(trunk_in, 512), nn.GELU(), nn.Linear(512, 384), nn.GELU())
        self.drop = nn.Dropout(dropout)
        self.drive = nn.Sequential(nn.Linear(384 + AUX, 128), nn.GELU(), nn.Linear(128, 2))
        self.place = nn.Linear(384, len(AREAS) + 6)
        self.goal = nn.Linear(384, 2 * len(AREAS))
        with torch.no_grad():
            self.drive[-1].weight.mul_(0.05)
            self.drive[-1].bias.zero_()

    def forward(self, history: torch.Tensor, ages: torch.Tensor, aux: torch.Tensor):
        present = (ages < AGE_NONE - 1e-3).float()
        f = self.drop(self.frame(history)) * present[..., None]
        f = torch.cat((f, present[..., None], ages[..., None].clamp(0, AGE_NONE) / AGE_NONE), -1).flatten(1)
        h = self.drop(self.trunk(torch.cat((f, aux), 1)))
        residual = self.drive(torch.cat((h, aux), 1))
        p = self.place(h)
        n = len(AREAS)
        room, pose = p[:, :n], p[:, n:]
        pose = torch.cat((pose[:, :2], F.normalize(pose[:, 2:4], dim=1, eps=1e-6), pose[:, 4:6].clamp(-4, 3)), 1)
        goal = F.normalize(self.goal(h).view(-1, n, 2), dim=-1, eps=1e-6)
        return residual, room, pose, goal


def wheels(residual: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
    """train/model.py's heading-following prior + residual; signed PWM/255 per side, |PWM| <= 150 * limit."""
    sin, cos, limit = aux[:, 5], aux[:, 6], aux[:, 8]
    v = (0.6 * cos.clamp_min(0).square() + residual[:, 0]).clamp(-0.4, 1.0)
    turn = (torch.where(cos > 0, 0.6 * sin, 0.6 * torch.sign(sin)) + residual[:, 1]).clamp(-0.8, 0.8)
    side = torch.stack((v - turn, v + turn), 1)
    peak = side.abs().amax(1, keepdim=True)
    scale = torch.minimum(torch.ones_like(peak), limit[:, None] / peak.clamp_min(1e-6))
    return side * scale * PWM_UNIT


def pack_aux(range_cm, range_ok, gyro_rate, last_left, last_right, heading_error, how_far, speed_limit,
             target_xy, goal_xy, frame_age):
    """CarObs + DriveCommand -> aux [B,AUX] (torch tensors, batch first). range_cm is ignored where range_ok is 0;
    last_* in PWM; gyro_rate rad/s; target_xy / goal_xy (forward, left) m; frame_age s of the newest frame."""
    t = torch.as_tensor(target_xy).clamp(-3, 3) / 3
    g = torch.as_tensor(goal_xy)
    dist = g.norm(dim=1)
    bearing = torch.atan2(g[:, 1], g[:, 0])
    return torch.stack((
        torch.where(range_ok > 0, range_cm.clamp(0, 150) / 150, torch.zeros_like(range_cm)), range_ok.float(),
        gyro_rate / math.radians(250), last_left / 255, last_right / 255,
        torch.sin(heading_error), torch.cos(heading_error), how_far / 2, speed_limit,
        t[:, 0], t[:, 1], torch.log1p(dist) / 2, torch.sin(bearing), torch.cos(bearing),
        frame_age.clamp(0, AGE_NONE) / AGE_NONE,
    ), 1).float()


def losses(out, aux, teacher_pwm, room, pose, goal, goal_ok):
    """Imitation + place + goal-direction losses (means); returns (total, parts dict)."""
    residual, room_logits, pose_hat, goal_hat = out
    pwm = wheels(residual, aux)
    drive = F.smooth_l1_loss(pwm / PWM_UNIT, teacher_pwm / PWM_UNIT, beta=0.1)
    ce = F.cross_entropy(room_logits, room)
    log_sd_xy, log_sd_yaw = pose_hat[:, 4], pose_hat[:, 5]
    err2 = ((pose_hat[:, :2] - pose[:, :2]) * 10).square().sum(1)
    nll_xy = 0.5 * err2 * torch.exp(-2 * log_sd_xy) + 2 * log_sd_xy
    dyaw = 1 - (pose_hat[:, 2:4] * pose[:, 2:4]).sum(1)
    nll_yaw = dyaw * torch.exp(-2 * log_sd_yaw) + log_sd_yaw
    g = (1 - (goal_hat * goal).sum(-1)) * goal_ok
    goal_loss = g.sum() / goal_ok.sum().clamp_min(1)
    total = drive + 0.3 * ce + 0.05 * (nll_xy.mean() + nll_yaw.mean()) + 0.3 * goal_loss
    return total, {"drive": drive.item(), "room_ce": ce.item(), "nll_xy": nll_xy.mean().item(),
                   "nll_yaw": nll_yaw.mean().item(), "goal": goal_loss.item(),
                   "room_acc": (room_logits.argmax(1) == room).float().mean().item(),
                   "xy_err_m": err2.sqrt().median().item()}


_Q_LUMA = torch.tensor([
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99], dtype=torch.float32).view(8, 8)
_Q_CHROMA = torch.full((8, 8), 99.0)
_Q_CHROMA[:4, :4] = torch.tensor([[17, 18, 24, 47], [18, 21, 26, 66], [24, 26, 56, 99], [47, 66, 99, 99]])


def _dct_matrix(device):
    k = torch.arange(8, device=device, dtype=torch.float32)
    m = torch.cos((2 * k[None] + 1) * k[:, None] * math.pi / 16) * math.sqrt(2 / 8)
    m[0] /= math.sqrt(2)
    return m


def _blocks_codec(x: torch.Tensor, table: torch.Tensor, scale: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """x [B,H,W] level-shifted channel; table [8,8]; scale [B] libjpeg quality scale. The 8 x 8 block DCT is a
    stride-8 convolution with the 64 basis images (and the inverse its transpose), which runs as one GEMM."""
    basis = (d[:, None, :, None] * d[None, :, None, :]).reshape(64, 1, 8, 8)
    coef = F.conv2d(x[:, None], basis, stride=8)
    q = ((table.reshape(1, 64) * scale[:, None] + 50) / 100).floor().clamp(1, 255)[:, :, None, None]
    coef = torch.round(coef / q) * q
    return F.conv_transpose2d(coef, basis, stride=8)[:, 0]


def jpeg_roundtrip(rgb: torch.Tensor, quality: torch.Tensor) -> torch.Tensor:
    """Baseline JPEG (4:2:0, standard tables) encode + decode on the GPU. rgb [B,3,H,W] 0..255 float; edges are
    replicated to whole 16 x 16 macroblocks like an encoder does, then cropped back."""
    h, w = rgb.shape[2:]
    rgb = F.pad(rgb, (0, -w % 16, 0, -h % 16), mode="replicate")
    d = _dct_matrix(rgb.device)
    q = quality.float().clamp(1, 100)
    scale = torch.where(q < 50, 5000 / q, 200 - 2 * q)
    r, g, b = rgb.unbind(1)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b
    y = _blocks_codec(y - 128, _Q_LUMA.to(rgb.device), scale, d) + 128
    chroma = []
    for c in (cb, cr):
        small = F.avg_pool2d(c[:, None], 2)[:, 0]
        small = _blocks_codec(small, _Q_CHROMA.to(rgb.device), scale, d)
        chroma.append(F.interpolate(small[:, None], scale_factor=2, mode="bilinear", align_corners=False)[:, 0])
    cb, cr = chroma
    out = torch.stack((y + 1.402 * cr, y - 0.344136 * cb - 0.714136 * cr, y + 1.772 * cb), 1)
    return out[:, :, :h, :w].clamp(0, 255).round()


def motion_blur(rgb: torch.Tensor, length_px: torch.Tensor, taps: int = 13) -> torch.Tensor:
    """Horizontal box blur of per-image length (px) - camera yaw during the exposure. rgb [B,3,H,W]."""
    b, c, h, w = rgb.shape
    k = torch.arange(taps, device=rgb.device, dtype=rgb.dtype) - taps // 2
    weight = (k.abs()[None] <= (length_px[:, None].to(rgb.dtype) / 2)).to(rgb.dtype)
    weight = (weight / weight.sum(1, keepdim=True)).repeat_interleave(c, 0)[:, None, None, :]
    x = F.pad(rgb.reshape(1, b * c, h, w), (taps // 2, taps // 2, 0, 0), mode="replicate")
    return F.conv2d(x, weight, groups=b * c).reshape(b, c, h, w)

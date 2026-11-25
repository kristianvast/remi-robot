"""X-NavDP point-goal network, ported from InternRobotics/NavDP baselines/x-navdp/eval/src (MIT, 2026 Tianyu Yang
and X-NavDP contributors) for batched inference and fine-tuning in plain PyTorch.

Parameter names equal the released checkpoint (x-navdp_posttrain.ckpt), so `load_state_dict` works unchanged.
Differences from upstream, all at inference time:
- the DDPM sampler is written out here (diffusers DDPMScheduler: 10 steps, squaredcos_cap_v2, epsilon, clip 1,
  fixed_small variance), so neither diffusers nor scipy is needed;
- no RTC prefix guidance and no B-spline smoothing of the sampled steps (both are upstream post-processing);
- RGB tokens are computed once per frame and kept (the RGB encoder is frozen, so this equals re-encoding the
  8-frame memory every step as upstream does).
Embodiment index 0 (Dingo, wheeled) is used throughout.
"""

import math

import torch
from torch import nn

from .dinov2 import DINOv2

DIM, TOKENS, PREDICT, MEMORY = 384, 256, 24, 8
FT_STEP, TRAIN_STEPS = 6, 10
WHEELED = 0


class LearnablePositionalEncoding(nn.Module):
    """Returns only the encoding (upstream semantics), not x + encoding."""

    def __init__(self, embed_dim: int, max_len: int) -> None:
        super().__init__()
        self.position_embedding = nn.Embedding(max_len, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.position_embedding.weight[: x.shape[1]][None].expand(x.shape[0], -1, -1)


def time_embedding(t: torch.Tensor, dim: int = DIM) -> torch.Tensor:
    half = dim // 2
    freq = torch.exp(torch.arange(half, device=t.device) * (-math.log(10000) / (half - 1)))
    arg = t.float()[:, None] * freq[None]
    return torch.cat((arg.sin(), arg.cos()), -1)


def alphas_cumprod(steps: int = TRAIN_STEPS) -> torch.Tensor:
    """diffusers betas_for_alpha_bar('cosine') = beta_schedule 'squaredcos_cap_v2'."""
    bar = lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2  # noqa: E731
    betas = torch.tensor([min(1 - bar((i + 1) / steps) / bar(i / steps), 0.999) for i in range(steps)],
                         dtype=torch.float64)
    return torch.cumprod(1 - betas, 0).float()


class Backbone(nn.Module):
    """Two DINOv2 ViT-S/14 encoders (Depth Anything V2 Small layout) + a 2-layer memory transformer."""

    def __init__(self, memory_size: int = MEMORY) -> None:
        super().__init__()
        self.memory_size = memory_size
        self.rgb_model = DINOv2("vits")
        self.depth_model = DINOv2("vits")
        self.former_query = LearnablePositionalEncoding(DIM, memory_size * 16)
        self.former_pe = LearnablePositionalEncoding(DIM, (memory_size + 1) * TOKENS)
        self.former_net = nn.TransformerDecoder(nn.TransformerDecoderLayer(DIM, 8, batch_first=True), 2)
        self.project_layer = nn.Linear(DIM, DIM)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def rgb_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """[B,224,224,3] float in 0..1 -> [B,256,384]."""
        x = (images.permute(0, 3, 1, 2) - self.mean) / self.std
        return self.rgb_model.get_intermediate_layers(x)[0]

    def depth_tokens(self, depths: torch.Tensor) -> torch.Tensor:
        """[B,224,224] metres (0 = invalid) -> [B,256,384]."""
        return self.depth_model.get_intermediate_layers(depths[:, None].expand(-1, 3, -1, -1))[0]

    def memory(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """rgb [B, memory*256, 384] (oldest first, current last), depth [B,256,384] -> [B, memory*16, 384]."""
        tokens = torch.cat((rgb, depth), 1)
        tokens = tokens + self.former_pe(tokens)
        query = self.former_query(tokens[:, : self.memory_size * 16])
        return self.project_layer(self.former_net(query, tokens))


class XNavDP(nn.Module):
    def __init__(self, memory_size: int = MEMORY, predict_size: int = PREDICT, depth: int = 16, heads: int = 8) -> None:
        super().__init__()
        self.predict_size = predict_size
        self.rgbd_encoder = Backbone(memory_size)
        self.point_encoder = nn.Linear(3, DIM)
        self.decoder_layer = nn.TransformerDecoderLayer(DIM, heads, 4 * DIM, activation="gelu", batch_first=True,
                                                        norm_first=True)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, depth)
        self.input_embed = nn.Linear(3, DIM)
        self.cond_pos_embed = LearnablePositionalEncoding(DIM, memory_size * 16 + 4)
        self.out_pos_embed = LearnablePositionalEncoding(DIM, predict_size)
        self.layernorm = nn.LayerNorm(DIM)
        self.action_head = nn.Linear(DIM, 3)
        self.embodiment_embedding = nn.Embedding(3, DIM)
        self.embody_tgt_delta = nn.Sequential(nn.Linear(DIM, DIM), nn.GELU(), nn.Linear(DIM, DIM))
        self.embody_out_film = nn.Linear(DIM, 2 * DIM)
        self.decoder_ft = nn.TransformerDecoder(self.decoder_layer, depth)
        self.layernorm_ft = nn.LayerNorm(DIM)
        self.action_head_ft = nn.Linear(DIM, 3)
        self.input_embed_ft = nn.Linear(3, DIM)
        self.decoder_q_layer = nn.TransformerDecoderLayer(DIM, heads, 4 * DIM, activation="gelu", dropout=0.0,
                                                          batch_first=True, norm_first=True)
        self.decoder_q = nn.TransformerDecoder(self.decoder_q_layer, depth)
        self.layernorm_q = nn.LayerNorm(DIM)
        self.q_pool_mlp = nn.Sequential(nn.Linear(predict_size * DIM, DIM), nn.GELU(), nn.Linear(DIM, DIM))
        self.q1_heads = nn.Linear(DIM, 1)
        self.q2_heads = nn.Linear(DIM, 1)
        self.input_embed_q = nn.Linear(3, DIM)
        self.q1_target_heads = nn.Linear(DIM, 1)
        self.q2_target_heads = nn.Linear(DIM, 1)
        causal = torch.triu(torch.full((predict_size, predict_size), float("-inf")), 1)
        self.register_buffer("tgt_mask", causal, persistent=False)
        critic = torch.zeros((predict_size, 5 + memory_size * 16))
        critic[:, 0] = float("-inf")
        self.register_buffer("critic_mask", critic, persistent=False)
        self.register_buffer("alphas_cumprod", alphas_cumprod(), persistent=False)

    def _embodiment(self, n: int, device) -> torch.Tensor:
        return self.embodiment_embedding(torch.full((n,), WHEELED, dtype=torch.long, device=device))

    def _cond(self, first: torch.Tensor, goal: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        cond = torch.cat((first, goal, goal, goal, memory), 1)
        return cond + self.cond_pos_embed(cond)

    def predict_noise(self, x: torch.Tensor, t: torch.Tensor, goal: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """Epsilon for actions x [B,24,3] at timesteps t [B]; t >= 6 uses the base decoder, t < 6 the
        embodiment-modulated fine-tune decoder (upstream's split)."""
        out = torch.empty_like(x)
        base = t >= FT_STEP
        for use_base in (True, False):
            m = base if use_base else ~base
            if not bool(m.any()):
                continue
            xs, ts, gs, ms = x[m], t[m], goal[m], memory[m]
            cond = self._cond(time_embedding(ts)[:, None].to(gs.dtype), gs, ms)
            if use_base:
                h = self.input_embed(xs)
                h = self.decoder(tgt=h + self.out_pos_embed(h), memory=cond, tgt_mask=self.tgt_mask)
                out[m] = self.action_head(self.layernorm(h)).to(out.dtype)
            else:
                e = self._embodiment(len(xs), x.device)
                h = self.input_embed_ft(xs)
                h = h + self.out_pos_embed(h) + self.embody_tgt_delta(e)[:, None]
                h = self.decoder_ft(tgt=h, memory=torch.cat((cond, e[:, None]), 1), tgt_mask=self.tgt_mask)
                gamma, beta = self.embody_out_film(e).chunk(2, -1)
                h = (1 + gamma[:, None]) * self.layernorm_ft(h) + beta[:, None]
                out[m] = self.action_head_ft(h).to(out.dtype)
        return out

    def twin_q(self, path: torch.Tensor, goal: torch.Tensor, memory: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Twin Q heads for cumulative paths [B,24,3] and encoded goals [B,1,384] -> ([B], [B])."""
        e = self._embodiment(len(path), path.device)
        h = self.input_embed_q(path.to(memory.dtype))
        h = h + self.out_pos_embed(h) + self.embody_tgt_delta(e)[:, None]
        cond = torch.cat((self._cond(goal, goal, memory), e[:, None]), 1)
        h = self.decoder_q(tgt=h, memory=cond, memory_mask=self.critic_mask)
        gamma, beta = self.embody_out_film(e).chunk(2, -1)
        pooled = self.q_pool_mlp(((1 + gamma[:, None]) * self.layernorm_q(h) + beta[:, None]).flatten(1))
        return self.q1_heads(pooled)[:, 0], self.q2_heads(pooled)[:, 0]

    def encode_goal(self, goal_xy: torch.Tensor, dtype) -> torch.Tensor:
        goal = torch.cat((goal_xy, torch.zeros_like(goal_xy[:, :1])), 1).to(dtype)
        return self.point_encoder(self.clip_goal(goal))[:, None]

    def ddpm_step(self, x: torch.Tensor, eps: torch.Tensor, t: int, generator=None) -> torch.Tensor:
        ac = self.alphas_cumprod
        a_t = ac[t]
        a_prev = ac[t - 1] if t > 0 else torch.ones_like(a_t)
        alpha_t = a_t / a_prev
        x0 = ((x - (1 - a_t).sqrt() * eps) / a_t.sqrt()).clamp(-1, 1)
        mean = (a_prev.sqrt() * (1 - alpha_t) / (1 - a_t)) * x0 + (alpha_t.sqrt() * (1 - a_prev) / (1 - a_t)) * x
        if t == 0:
            return mean
        var = ((1 - a_prev) / (1 - a_t) * (1 - alpha_t)).clamp_min(1e-20)
        return mean + var.sqrt() * torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)

    @torch.no_grad()
    def sample(self, memory: torch.Tensor, goal_xy: torch.Tensor, samples: int, generator=None):
        """memory [B,128,384], goal_xy [B,2] (forward, left) m -> paths [B,S,24,3] (cumulative, car frame
        at the frame time), q [B,S]."""
        b = len(goal_xy)
        goal = self.encode_goal(goal_xy, memory.dtype).repeat_interleave(samples, 0)
        memory = memory.repeat_interleave(samples, 0)
        x = torch.randn((b * samples, self.predict_size, 3), device=memory.device, dtype=memory.dtype,
                        generator=generator)
        for t in range(TRAIN_STEPS - 1, -1, -1):
            steps = torch.full((len(x),), t, device=x.device, dtype=torch.long)
            x = self.ddpm_step(x, self.predict_noise(x, steps, goal, memory), t, generator)
        path = torch.cumsum(x / 4.0, 1)
        q1, q2 = self.twin_q(path, goal, memory)
        return path.view(b, samples, self.predict_size, 3), ((q1 + q2) / 2).view(b, samples)

    @staticmethod
    def clip_goal(goal: torch.Tensor) -> torch.Tensor:
        norm = goal.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return goal * (25.0 / norm).clamp(max=1.0)

    def imitation_loss(self, memory: torch.Tensor, goal_xy: torch.Tensor, path: torch.Tensor) -> torch.Tensor:
        """Epsilon-matching loss (NavDP's diffusion objective) for expert cumulative paths [B,24,2|3]."""
        if path.shape[-1] == 2:
            path = torch.cat((path, torch.zeros_like(path[..., :1])), -1)
        seg = torch.diff(path, dim=1, prepend=torch.zeros_like(path[:, :1])) * 4.0
        x0 = seg.clamp(-1, 1).to(memory.dtype)
        goal = self.encode_goal(goal_xy, memory.dtype)
        t = torch.randint(0, TRAIN_STEPS, (len(x0),), device=x0.device)
        a = self.alphas_cumprod[t].view(-1, 1, 1).to(x0.dtype)
        noise = torch.randn_like(x0)
        eps = self.predict_noise(a.sqrt() * x0 + (1 - a).sqrt() * noise, t, goal, memory)
        return (eps.float() - noise.float()).square().mean()

    def critic_loss(self, memory: torch.Tensor, goal_xy: torch.Tensor, paths: torch.Tensor,
                    target: torch.Tensor) -> torch.Tensor:
        """Regress both Q heads to a privileged score for cumulative paths [B,24,3]."""
        q1, q2 = self.twin_q(paths, self.encode_goal(goal_xy, memory.dtype), memory)
        return (q1.float() - target).square().mean() + (q2.float() - target).square().mean()


def load(path, device="cpu", dtype=torch.float32) -> XNavDP:
    """Build the network and load a checkpoint: the upstream file (a bare state dict) or ours ({'model': ...})."""
    net = XNavDP()
    state = torch.load(path, map_location="cpu", weights_only=True)
    state = state.get("model", state)
    missing, unexpected = net.load_state_dict(state, strict=False)
    missing = [k for k in missing if not k.endswith(("mask", "alphas_cumprod", "mean", "std"))]
    if missing:
        raise KeyError(f"checkpoint lacks {len(missing)} weights, e.g. {missing[:5]}")
    net.unexpected_keys = unexpected
    return net.to(device=device, dtype=dtype).eval()

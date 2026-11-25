"""Convert a Depth Anything V2 (HF transformers) checkpoint to a Core ML package.

usage: convert_coreml.py <hf_model_id> <width> <height> <out.mlpackage>
Input "image": float32 (1,3,H,W) RGB in 0..255; normalisation is inside the graph.
Output "depth": (1,H,W) float - relative inverse depth (Small) or metres (Metric).
"""
import sys
import coremltools as ct
import numpy as np
import torch
from transformers import AutoModelForDepthEstimation

mid, W, H, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
assert W % 14 == 0 and H % 14 == 0


class Wrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255)

    def forward(self, x):
        return self.m(pixel_values=(x - self.mean) / self.std).predicted_depth


m = Wrap(AutoModelForDepthEstimation.from_pretrained(mid).eval())
x = torch.rand(1, 3, H, W) * 255
# The input size is fixed, so the bicubic position-embedding resize (unsupported by
# Core ML) is a constant: compute it once and bake it in.
emb = m.m.backbone.embeddings
with torch.no_grad():
    tokens = torch.zeros(1, (H // 14) * (W // 14) + 1, emb.position_embeddings.shape[-1])
    pos = emb.interpolate_pos_encoding(tokens, H, W).clone()
emb.register_buffer("fixed_pos", pos)
emb.interpolate_pos_encoding = lambda embeddings, height, width: emb.fixed_pos
with torch.no_grad():
    ref = m(x)
    ep = torch.export.export(m, (x,)).run_decompositions({})
ml = ct.convert(ep, inputs=[ct.TensorType(name="image", shape=x.shape)], outputs=[ct.TensorType(name="depth")],
                compute_precision=ct.precision.FLOAT16, minimum_deployment_target=ct.target.macOS15,
                convert_to="mlprogram")
ml.save(out)
y = ml.predict({"image": x.numpy()})["depth"]
r = ref.numpy()
print(out, "rel err vs torch fp32:", float(np.median(np.abs(y - r) / (np.abs(r) + 1e-6))))

# Weights: source and licence

## Encoder (inside encoder.onnx)
- Model: Depth Anything V2 Metric-Indoor-Small, transformers format
  https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf
- Revision: 8078d68a9c75a972131914f6afd0c1723be0da7f (HF API `sha`, read 2026-11-24)
- File: model.safetensors, 99,173,660 bytes,
  sha256 e990eb82fbf11b05b7813261196a2b841bdcf5a05f64396724a8987fa90504a3 (matches the LFS blob id)
- Licence: Apache-2.0. Source of that statement: the upstream repository README
  (https://github.com/DepthAnything/Depth-Anything-V2, section LICENSE: "Depth-Anything-V2-Small model is under
  the Apache-2.0 license. Depth-Anything-V2-Base/Large/Giant models are under the CC-BY-NC-4.0 license.").
  The Hugging Face model card of this metric-indoor checkpoint carries no licence tag, so "Apache-2.0 for the
  metric-indoor Small fine-tune" is inferred from the upstream statement for Small models, not read from the card.
- Used frozen; exported unchanged (the bicubic position-embedding resize is baked in for the fixed 224 x 168 input).

## Heads (heads.onnx, heads.pt)
Trained for this project (sim/isaac/elegoo/combined/train.py). Their teacher, sim/policy/driver_v2.onnx, is this
project's own driver policy.

Hashes of every file in this folder: SHA256SUMS.

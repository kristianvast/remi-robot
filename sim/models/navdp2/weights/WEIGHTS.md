# Weights, sources, hashes, licences

## Starting checkpoint (not stored here, 873 MB)
- Source: https://huggingface.co/InternRobotics/X-NavDP/blob/main/x-navdp_posttrain.ckpt (public, not gated;
  downloaded 2026-11-24 to elegoo-sim-8 with
  `curl -L https://huggingface.co/InternRobotics/X-NavDP/resolve/main/x-navdp_posttrain.ckpt`).
- Size 873,100,357 B; sha256 `267089a81bbbe7a913debda6603f3f1b66a79520370ce953b2d888d793b89f24`
  (equals the Hugging Face LFS oid; checked on the VM with sha256sum).
- What it is: X-NavDP post-trained policy (NavDP diffusion policy + RL post-training, GQRM), arXiv 2607.28560.
- Licence: the model card states none. The code in `baselines/x-navdp` is MIT
  (`LICENSE-MIT-X-NavDP.txt`, "Copyright (c) 2026 Tianyu Yang and X-NavDP contributors"); its THIRD_PARTY_NOTICES
  put the Depth Anything V2 / DINOv2 parts under Apache-2.0 (`LICENSE-Apache-2.0-DINOv2-DepthAnythingV2.txt`).
  The NavDP repository root states CC BY-NC-SA 4.0 for "the open-sourced code"; the bare NavDP checkpoint needs a
  Google form and was not used. So: weights licence none stated; treat as research / non-commercial until the
  authors say otherwise.

## Fine-tuned weights (this folder)
- `navdp_elegoo.pt`: `{"model": fp16 state dict, "step": n}` of `xnavdp/net.py:XNavDP`, fine-tuned by
  `sim/isaac/elegoo/navdp/train.py` (recipe in `.omo/waves/p4/navdp.md`): 2507 steps, final checkpoint.
  347,166,584 B, sha256 `4ffdc2cfdc5f4aba534a4f0fed4fa1c147b451d628fdd6515c2b9f8eb66717f8`. Derived from the X-NavDP
  checkpoint above, so the same licence situation applies.

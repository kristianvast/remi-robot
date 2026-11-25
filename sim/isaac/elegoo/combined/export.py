"""Export the combined model for the Mac as two ONNX graphs and check them against PyTorch (container python, no Kit).

  encoder.onnx  image [1,3,168,224] float RGB 0..255 -> depth [1,120,160] (spec encoding), frame [1,FRAME]
  heads.onnx    history [1,4,FRAME], ages [1,4], aux [1,AUX] -> residual [1,2], room [1,10], pose [1,6], goal [1,10,2]

  /opt/isaaclab-venv/bin/python p4c/sim/isaac/elegoo/combined/export.py --heads p4c/run/heads.pt --output p4c/export
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.append(str(Path(__file__).resolve().parents[4] / "pylib"))
from elegoo.combined.common import AGE_NONE, AUX, FRAME, HISTORY, IMG_H, IMG_W, Encoder, Heads


def fix_position_embedding(encoder: Encoder) -> None:
    """The input size is fixed, so bake the bicubic position-embedding resize in (as sim/realdepth/convert_coreml.py)."""
    emb = encoder.m.backbone.embeddings
    with torch.no_grad():
        tokens = torch.zeros(1, (IMG_H // 14) * (IMG_W // 14) + 1, emb.position_embeddings.shape[-1])
        pos = emb.interpolate_pos_encoding(tokens, IMG_H, IMG_W).clone()
    emb.register_buffer("fixed_pos", pos)
    emb.interpolate_pos_encoding = lambda embeddings, height, width: emb.fixed_pos


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heads", type=Path, required=True)
    parser.add_argument("--da2", default="p4c/da2")
    parser.add_argument("--sample", type=Path, help="record.npz with real rendered frames for the parity check")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import onnxruntime as ort

    args.output.mkdir(parents=True, exist_ok=True)
    encoder = Encoder.pretrained(args.da2).eval()
    fix_position_embedding(encoder)
    heads = Heads().eval()
    heads.load_state_dict(torch.load(args.heads, map_location="cpu", weights_only=True))
    image = torch.rand(1, 3, IMG_H, IMG_W) * 255
    history, ages = torch.randn(1, HISTORY, FRAME), torch.tensor([[0.15, 0.25, 0.35, AGE_NONE]])
    aux = torch.randn(1, AUX) * 0.3
    torch.onnx.export(encoder, (image,), args.output / "encoder.onnx", input_names=["image"],
                      output_names=["depth", "frame"], opset_version=17, dynamo=False)
    torch.onnx.export(heads, (history, ages, aux), args.output / "heads.onnx", input_names=["history", "ages", "aux"],
                      output_names=["residual", "room", "pose", "goal"], opset_version=17, dynamo=False)
    report = {}
    images = [image]
    if args.sample is not None:
        rgb = np.load(args.sample)["rgb"][0, ::40]
        for frame in rgb[:8]:
            x = torch.as_tensor(frame.astype(np.float32))[None]
            images.append(torch.nn.functional.interpolate(x, size=(IMG_H, IMG_W), mode="bilinear", antialias=True))
    enc = ort.InferenceSession(str(args.output / "encoder.onnx"), providers=["CPUExecutionProvider"])
    errs_d, errs_f = [], []
    with torch.no_grad():
        for x in images:
            d, f = encoder(x)
            od, of = enc.run(None, {"image": x.numpy()})
            errs_d.append(float(np.abs(od - d.numpy()).max()))
            errs_f.append(float(np.abs(of - f.numpy()).max() / (np.abs(f.numpy()).max() + 1e-6)))
        hd = ort.InferenceSession(str(args.output / "heads.onnx"), providers=["CPUExecutionProvider"])
        ref = heads(history, ages, aux)
        got = hd.run(None, {"history": history.numpy(), "ages": ages.numpy(), "aux": aux.numpy()})
    report["encoder_depth_max_abs"] = max(errs_d)
    report["encoder_frame_max_rel"] = max(errs_f)
    report["heads_max_abs"] = max(float(np.abs(g - r.numpy()).max()) for g, r in zip(got, ref))
    report["frames_checked"] = len(images)
    (args.output / "export_check.json").write_text(json.dumps(report, indent=1))
    print("EXPORT_OK", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

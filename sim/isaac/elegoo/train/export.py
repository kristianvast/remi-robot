"""Export the trained driver to ONNX and replay recorded observations through PyTorch on CPU.

  sim/cloud/run_headless.sh drv/elegoo/train/export.py --checkpoint /workspace/elegoo/drv/run/ppo.pt \
      --replay /workspace/elegoo/drv/eval/replay.npz --output /workspace/elegoo/drv/export
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from elegoo.train.model import AUX, FEATURES, HISTORY, driver_model, load_actor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    model = driver_model(load_actor(args.checkpoint, "cpu")).eval()
    inputs = (torch.zeros(1, 1, 120, 160), torch.zeros(1, AUX), torch.zeros(1, HISTORY, FEATURES))
    target = args.output / "driver.onnx"
    torch.onnx.export(model, inputs, str(target), input_names=["depth", "aux", "history"],
                      output_names=["action", "next_history"], opset_version=17, dynamo=False)
    with np.load(args.replay) as recording:
        depth, aux, reset, pwm = (recording[k] for k in ("depth", "aux", "reset", "pwm"))
    history = inputs[2]
    found = []
    with torch.inference_mode():
        for image, features, episode_reset in zip(depth, aux, reset, strict=True):
            if episode_reset:
                history = torch.zeros_like(history)
            action, history = model(torch.from_numpy(image)[None, None], torch.from_numpy(features)[None], history)
            found.append(np.rint(np.clip(action[0].numpy(), -1, 1) * 255).astype(np.int16))
    difference = int(np.abs(np.asarray(found) - pwm).max())
    result = {"frames": len(depth), "max_pwm_difference_cpu_vs_gpu_eval": difference, "onnx": str(target)}
    (args.output / "export.json").write_text(json.dumps(result))
    print("EXPORT_DONE", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

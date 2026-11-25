"""Candidate depth models on Apple Silicon: latency benchmark and depth dumps.

  probe.py bench <config> [--frames N]   -> one JSON line (run one config per process,
                                            so peak memory is per model)
  probe.py dump <config> <out_dir> <img...> -> <out_dir>/<config>/<stem>.npy

Every runner takes a 640x480 RGB uint8 frame and returns a float32 map at the
model's own resolution: kind "inv" (relative inverse depth, larger = nearer)
or "metric" (metres along the optical axis).
"""
import glob
import json
import os
import resource
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_HOME", os.path.join(HERE, "models", "hf"))
CAR_FX = 502.3  # px at 640x480, ~65 deg HFOV (sim/isaac calibration)


def coreml_runner(pkg, units):
    import coremltools as ct

    cu = {"ALL": ct.ComputeUnit.ALL, "NE": ct.ComputeUnit.CPU_AND_NE,
          "GPU": ct.ComputeUnit.CPU_AND_GPU, "CPU": ct.ComputeUnit.CPU_ONLY}[units]
    m = ct.models.MLModel(os.path.join(HERE, "models", "coreml", pkg), compute_units=cu)
    shp = m.get_spec().description.input[0].type.multiArrayType.shape
    H, W = int(shp[2]), int(shp[3])

    def run(rgb):
        x = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)
        x = np.ascontiguousarray(x.transpose(2, 0, 1)[None], dtype=np.float32)
        return m.predict({"image": x})["depth"][0].astype(np.float32)
    return run


def apple_coreml_runner(units):
    import coremltools as ct
    from PIL import Image

    cu = {"ALL": ct.ComputeUnit.ALL, "GPU": ct.ComputeUnit.CPU_AND_GPU}[units]
    m = ct.models.MLModel(os.path.join(HERE, "models", "apple-coreml", "DepthAnythingV2SmallF16.mlpackage"),
                          compute_units=cu)

    def run(rgb):
        im = Image.fromarray(cv2.resize(rgb, (518, 392), interpolation=cv2.INTER_AREA))
        return np.asarray(m.predict({"image": im})["depth"], dtype=np.float32)
    return run


def torch_da2_runner(mid, W, H):
    import torch
    from transformers import AutoModelForDepthEstimation

    dev = torch.device("mps")
    m = AutoModelForDepthEstimation.from_pretrained(mid).eval().to(dev, torch.float16)
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

    @torch.no_grad()
    def run(rgb):
        x = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(x).to(dev).permute(2, 0, 1)[None].float().div(255)
        y = m(pixel_values=((x - mean) / std).half()).predicted_depth
        return y[0].float().cpu().numpy()
    return run


def depthpro2_runner(W, H, precision):
    import torch
    from depth_pro_2 import DepthPro2
    from depth_pro_2.model import to_normalized_tensor

    dev = torch.device("mps")
    dt = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
    m = DepthPro2.from_pretrained("apple/DepthPro2-S", precision=dt, fov=False).to(dev).eval()
    f_px = CAR_FX * W / 640

    def run(rgb):
        x = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)
        d = m.infer(to_normalized_tensor(x, dev), f_px=f_px)["depth"]
        return d.float().cpu().numpy()
    return run


DA2_REL = "depth-anything/Depth-Anything-V2-Small-hf"
DA2_MET = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
CONFIGS = {
    "apple_da2rel_518_coreml_all": (lambda: apple_coreml_runner("ALL"), "inv"),
    "da2rel_518_coreml_all": (lambda: coreml_runner("da2_rel_518x392.mlpackage", "ALL"), "inv"),
    "da2rel_518_coreml_gpu": (lambda: coreml_runner("da2_rel_518x392.mlpackage", "GPU"), "inv"),
    "da2rel_336_coreml_all": (lambda: coreml_runner("da2_rel_336x252.mlpackage", "ALL"), "inv"),
    "da2rel_336_coreml_ne": (lambda: coreml_runner("da2_rel_336x252.mlpackage", "NE"), "inv"),
    "da2rel_336_coreml_gpu": (lambda: coreml_runner("da2_rel_336x252.mlpackage", "GPU"), "inv"),
    "da2rel_280_coreml_all": (lambda: coreml_runner("da2_rel_280x210.mlpackage", "ALL"), "inv"),
    "da2rel_518_mps_fp16": (lambda: torch_da2_runner(DA2_REL, 518, 392), "inv"),
    "da2rel_336_mps_fp16": (lambda: torch_da2_runner(DA2_REL, 336, 252), "inv"),
    "da2met_518_coreml_all": (lambda: coreml_runner("da2_metric_518x392.mlpackage", "ALL"), "metric"),
    "da2met_518_coreml_gpu": (lambda: coreml_runner("da2_metric_518x392.mlpackage", "GPU"), "metric"),
    "da2met_336_coreml_all": (lambda: coreml_runner("da2_metric_336x252.mlpackage", "ALL"), "metric"),
    "da2met_336_coreml_gpu": (lambda: coreml_runner("da2_metric_336x252.mlpackage", "GPU"), "metric"),
    "da2met_280_coreml_all": (lambda: coreml_runner("da2_metric_280x210.mlpackage", "ALL"), "metric"),
    "da2met_518_mps_fp16": (lambda: torch_da2_runner(DA2_MET, 518, 392), "metric"),
    "dp2s_640_mps_bf16": (lambda: depthpro2_runner(640, 480, "bf16"), "metric"),
    "dp2s_320_mps_bf16": (lambda: depthpro2_runner(320, 240, "bf16"), "metric"),
    "dp2s_320_mps_fp16": (lambda: depthpro2_runner(320, 240, "fp16"), "metric"),
}


def real_frames():
    car = sorted(glob.glob(os.path.join(HERE, "..", "..", "patrol-photos", "bench", "*.jpg")))
    film = sorted(glob.glob(os.path.join(HERE, "data", "film640", "*.jpg")))[::20]
    return [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in car + film]


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20  # bytes on macOS


def bench(name, n=200, warm=20):
    frames = real_frames()
    base = rss_mb()
    t0 = time.perf_counter()
    run = CONFIGS[name][0]()
    load_s = time.perf_counter() - t0
    for i in range(warm):
        run(frames[i % len(frames)])
    ts = []
    for i in range(n):
        f = frames[i % len(frames)]
        t = time.perf_counter()
        y = run(f)
        cv2.resize(y, (160, 120), interpolation=cv2.INTER_AREA)
        ts.append((time.perf_counter() - t) * 1000)
    ts = np.array(ts)
    out = {"config": name, "out_shape": list(y.shape), "n": n, "median_ms": round(float(np.median(ts)), 2),
           "p95_ms": round(float(np.percentile(ts, 95)), 2), "fps_median": round(1000 / float(np.median(ts)), 1),
           "load_s": round(load_s, 1), "peak_rss_mb": round(rss_mb(), 0), "rss_before_model_mb": round(base, 0)}
    if "mps" in name:
        import torch
        out["mps_driver_mb"] = round(torch.mps.driver_allocated_memory() / 2**20, 0)
    print(json.dumps(out))


def dump(name, out_dir, paths):
    run = CONFIGS[name][0]()
    d = os.path.join(out_dir, name)
    os.makedirs(d, exist_ok=True)
    for p in paths:
        rgb = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        np.save(os.path.join(d, os.path.splitext(os.path.basename(p))[0] + ".npy"), run(rgb))
    print(name, "dumped", len(paths))


if __name__ == "__main__":
    if sys.argv[1] == "bench":
        bench(sys.argv[2], int(sys.argv[4]) if len(sys.argv) > 4 else 200)
    elif sys.argv[1] == "dump":
        dump(sys.argv[2], sys.argv[3], sys.argv[4:])

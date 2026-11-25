# Combined camera model "C" - training and evaluation (p4)

Delivered model: sim/models/combined/ (driver.py, weights/). Report: .omo/waves/p4/combined.md.

| File | Runs on | What |
|---|---|---|
| `common.py` | anywhere | Encoder (DA2 Metric-Indoor-Small: depth + frame feature), Heads, `wheels`, `pack_aux`, losses, GPU JPEG codec, motion blur |
| `env.py` | L4 (Kit) | `CombinedEnv(DriverEnv)` + truth labels, camera `Link` (latency, drops, 640x480 JPEG path), batched torch rebuild of driver_v2 (teacher), student inputs |
| `train.py` | L4 (Kit) | Stage 1 DAgger: teacher labels every step, per-episode teacher/student driver, replay of cached frame features, heads trained between chunks; `--profile` per-stage timings |
| `evaluate.py` | L4 (Kit) | Held-out test routes: student nominal / stress, v2 nominal / stress, teacher; `--real-path` = driver.py's input path + match statistics |
| `export.py` | Mac / container python | `encoder.onnx` + `heads.onnx` and an ONNX-vs-PyTorch check on recorded frames |
| `videos.py` | Mac / container python | success / failure videos and contact sheets from `record.npz` |
| `labels.py` | elegoo-sim-3 | `labels.npz`: room raster + 10 goal-direction fields (geodesic, clearance weighted) |
| `loc_compare.py` | elegoo-sim-3 | place head vs sim/localize on the same recorded frames (room prior and cold start) |
| `summarize.py` | Mac | results tables (`results/summary_*`) and `results/learning_curves.png` |
| `results/` | | learning.csv, per-eval episodes.csv / place.npz / path_stats.npy, loc summary, plots |

## Reproduce (VM layout: ~/elegoo-sim/work/p4c mirrors the project: sim/isaac/{elegoo,scene}, sim/rooms/bank, da2/ = the HF snapshot, driver_v2.onnx, pylib/ = onnxruntime + opencv-python-headless)
```
sim/cloud/run_headless.sh elegoo-sim-9 p4c/sim/isaac/elegoo/combined/train.py --num-envs 384 --budget-sec 4500 \
    --steps-per-iter 32 --updates-per-iter 48 --batch 1024 --buffer 300000 --teacher-steps 250 --anneal-steps 1200 \
    --beta-min 0.15 --output /workspace/elegoo/p4c/run --viz none
E=p4c/sim/isaac/elegoo/combined; H=/workspace/elegoo/p4c/run/heads.pt
run_headless.sh elegoo-sim-9 $E/evaluate.py --groups nominal/v2,stress/v2,nominal/teacher --output /workspace/elegoo/p4c/eval_base --viz none
run_headless.sh elegoo-sim-9 $E/evaluate.py --groups nominal/student --heads $H --output /workspace/elegoo/p4c/eval_nominal --viz none
run_headless.sh elegoo-sim-9 $E/evaluate.py --groups stress/student --heads $H --record-per-area 0 --output /workspace/elegoo/p4c/eval_stress --viz none
run_headless.sh elegoo-sim-9 $E/evaluate.py --groups nominal/student --heads $H --per-area 6 --record-per-area 0 --real-path \
    --rgb-size 640 480 --output /workspace/elegoo/p4c/eval_real --viz none
sim-3 (cwd ~/elegoo): PYTHONPATH=~/elegoo sim/macsim/.venv/bin/python ~/p4-combined/loc_compare.py record_loc.npz labels.npz out/
Mac (sim/isaac): sim/realdepth/.venv/bin/python elegoo/combined/export.py --heads ... --da2 <snapshot> --sample record.npz \
    --output ../models/combined/weights     (needs the `onnx` package on the path)
Mac: sim/realdepth/.venv/bin/python -m sim.models.combined.driver --bench 5
```
Run containers with `PYTHONDONTWRITEBYTECODE=1` or remove the root-owned `__pycache__` before the next push.

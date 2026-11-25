"""Replay the recorded Gemini tool calls through a driver with real-car inputs only, and score on truth.

    # on the L4 (inside the derived Isaac Lab image, see run.sh / remote/Dockerfile):
    python sim/cmdtest/harness.py --backend isaac --drivers hand policy_v2_veto --slots 8 --out sim/cmdtest/results
    # anywhere, no GPU: the same tool layer, replay and scoring on macsim (smoke test of the plumbing only;
    # its depth is macsim's simulated corruption, not Depth Pro, so its numbers are not results)
    python sim/cmdtest/harness.py --backend macsim --drivers hand --requests fwd_1m --seeds 101

`slots` cars run at once in the shared building (no car-to-car contact, each camera sees only its own car).
Every trial: spawn, per-episode draws (seeded by the trial seed), the request's calls replayed in a thread by
tools.replay, CarObs built by inputs.RealInputs + the depth estimator, the driver reached only through
AuditedDriver (CarObs + DriveCommand, checked by type), truth logged per step for score.py.
"""

import argparse
import json
import sys
import threading
import time
import traceback
import zlib
from collections import defaultdict
from dataclasses import replace
from math import cos, degrees, pi, radians, sin
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(ROOT), str(ROOT / "sim" / "isaac")):
    if p not in sys.path:
        sys.path.insert(0, p)

from sim.cmdtest import score  # noqa: E402
from sim.cmdtest.api import CONTRACT_FIELDS, CarObs, DriveCommand, clip_pwm, load_driver  # noqa: E402
from sim.cmdtest.inputs import DT, RealInputs  # noqa: E402
from sim.cmdtest.tools import CLOCK, Aborted, CarTools, Link, replay  # noqa: E402

import places  # noqa: E402

SCRIPT = HERE / "gemini_script.json"
SEEDS = (101, 202, 303, 404, 505)
TRIAL_CAP_S = 420.0
LOG_KEYS = ("t", "x", "y", "yaw", "bumper", "ahead_m", "tilted", "stall_s", "pwm_l", "pwm_r", "pan", "est_x", "est_y")


class Audit:
    def __init__(self, driver_name):
        self.driver = driver_name
        self.calls = 0
        self.arg_types = set()
        self.rgb = set()
        self.depth = set()
        self.modes = defaultdict(int)
        self.act_ms = []

    def check(self, obs, cmd):
        if type(obs) is not CarObs or type(cmd) is not DriveCommand:
            raise TypeError(f"driver input must be CarObs + DriveCommand, got {type(obs)} {type(cmd)}")
        self.calls += 1
        self.arg_types.add((type(obs).__name__, type(cmd).__name__))
        self.rgb.add("None" if obs.rgb is None else f"{obs.rgb.dtype}{obs.rgb.shape}")
        self.depth.add("None" if obs.depth is None else f"{obs.depth.dtype}{obs.depth.shape}")
        self.modes[cmd.mode] += 1


class AuditedDriver:
    """The only reference the tool layer has to the driver under test."""

    def __init__(self, inner, audit):
        self._inner, self._audit = inner, audit
        self.name = getattr(inner, "name", audit.driver)
        self.command_style = getattr(inner, "command_style", "line_of_sight")

    @property
    def front(self):
        return float(getattr(self._inner, "front", float("inf")))

    def reset(self, seed=None):
        self._inner.reset(seed)

    def act(self, obs, cmd):
        self._audit.check(obs, cmd)
        began = time.perf_counter()
        out = self._inner.act(obs, cmd)
        self._audit.act_ms.append((time.perf_counter() - began) * 1000)
        return clip_pwm(out)


class Spawner:
    """go_to requests start at home (+-8 cm, +-8 deg hand placement, goto.md); others on a free cell of home's room."""

    def __init__(self, is_free):
        from sim.localize.range_map import range_map
        from sim.macsim.world import floor
        from sim.navigate.planner import Planner

        self.home = places.home()
        fl, clearance = floor(), range_map(False).clearance
        cells = []
        for x, y in Planner().xy:
            r, c = fl.cells(x, y)
            if fl.area(x, y) == self.home.area and clearance[r, c] >= .35 and is_free(x, y):
                cells.append((float(x), float(y)))
        self.cells = cells

    def __call__(self, request, seed):
        rng = np.random.default_rng([seed, zlib.crc32(request["id"].encode())])
        if any(c["name"] == "go_to" for t in request["turns"] for c in t["calls"]):
            h = self.home
            return (h.x + rng.uniform(-.08, .08), h.y + rng.uniform(-.08, .08)), h.yaw + radians(rng.uniform(-8, 8))
        x, y = self.cells[int(rng.integers(len(self.cells)))]
        return (x, y), float(rng.uniform(-pi, pi))


class IsaacBackend:
    def __init__(self, sim, slots, seed):
        import torch

        from sim.cmdtest.env import CmdEnv, cmdtest_spec

        self.torch = torch
        self.env = CmdEnv(sim, cmdtest_spec(slots, seed))
        self.n = slots
        self.parts = defaultdict(float)
        for _ in range(20):
            self.env.advance(torch.zeros((slots, 2), device=self.env.device))
            self.env.observe()
        self.depth = DepthPro(torch)

    def is_free(self, x, y):
        t = self.torch.tensor([[x, y]], device=self.env.device)
        return bool(self.env.on_free(t)[0])

    def start(self, slot, xy, yaw, seed):
        t = self.torch
        ids = t.tensor([slot], device=self.env.device)
        return self.env.start_episode(ids, t.tensor([xy], device=self.env.device, dtype=t.float32),
                                      t.tensor([yaw], device=self.env.device, dtype=t.float32), seed)

    def step(self, cmds, heads):
        t, env = self.torch, self.env
        action = t.tensor([[c[0] / 255, c[1] / 255] for c in cmds], device=env.device, dtype=t.float32)
        env.pan_target[:, 0] = t.tensor([radians(h - 90) for h in heads], device=env.device)
        mark = time.perf_counter()
        env.advance(action)
        self.parts["advance_s"] += time.perf_counter() - mark
        mark = time.perf_counter()
        obs = env.observe()
        truth = {k: v.cpu().numpy() for k, v in env.truth().items()}
        self.parts["observe_s"] += time.perf_counter() - mark
        mark = time.perf_counter()
        frames = env.crop_frames(obs["rgb"]).cpu().numpy()
        self.parts["frames_s"] += time.perf_counter() - mark
        yaw_rate = env.robot.data.root_com_vel_w.torch[:, 5].cpu().numpy()
        rate = np.radians(obs["gyro_rate_deg_s"].cpu().numpy())
        heading = np.radians(obs["yaw_deg"].cpu().numpy())
        rng_cm = obs["range_cm"].cpu().numpy()
        hfov = env.hfov.cpu().numpy()
        return [dict(frame=frames[i], true_yaw_rate=float(yaw_rate[i]), hfov=float(hfov[i]), range_cm=float(rng_cm[i]),
                     gyro_rate=float(rate[i]), gyro_heading=float(heading[i]),
                     truth={k: v[i] for k, v in truth.items()}) for i in range(self.n)]


class DepthPro:
    """sim/realdepth/realdepth.py (Depth Pro 2-S, 640x480 -> 320x240, bfloat16) on CUDA, batched over all cars' frames.

    depth_pro_2 treats a (T, 3, H, W) batch as one clip: its cross-frame blocks attend over every frame's patches.
    Here the encoder's cross_frame_attention is called once per frame (each a one-frame clip, as observe() runs it)
    while the other blocks, the reassembly and the per-frame decoder run batched. check() measures the result against
    observe() frame by frame; it goes into the audit."""

    def __init__(self, torch):
        import cv2
        from depth_pro_2.model import to_normalized_tensor

        from sim.realdepth import realdepth

        realdepth._device = torch.device("cuda")
        self.torch, self.cv2, self.rd, self.norm = torch, cv2, realdepth, to_normalized_tensor
        self.model = realdepth._load()
        self.encoders = [m for m in self.model.modules() if hasattr(m, "cross_frame_attention")]
        self.ms = []
        self.check_result = None
        from concurrent.futures import ThreadPoolExecutor

        self.stream = torch.cuda.Stream()
        self.worker = ThreadPoolExecutor(1, thread_name_prefix="depth")

    def submit(self, frames):
        """Estimate on a side CUDA stream in the worker thread, overlapping the simulator's next step."""
        def run():
            with self.torch.cuda.stream(self.stream):
                return self.batch(frames)
        return self.worker.submit(run)

    def _independent(self, on):
        for m in self.encoders:
            if not on:
                m.__dict__.pop("cross_frame_attention", None)
                continue
            joint = type(m).cross_frame_attention.__get__(m)
            m.cross_frame_attention = lambda block, tokens, rope, joint=joint, cat=self.torch.cat: cat(
                [joint(block, tokens[i:i + 1], rope) for i in range(tokens.shape[0])])

    def batch(self, frames):
        rd, t = self.rd, self.torch
        began = time.perf_counter()
        x = t.stack([self.norm(self.cv2.resize(f, (rd.IN_W, rd.IN_H), interpolation=self.cv2.INTER_AREA), rd._device)
                     for f in frames])
        self._independent(True)
        try:
            with t.no_grad():
                z = self.model.infer(x, f_px=rd.CAR_FX_640 * rd.IN_W / 640)["depth"].float().cpu().numpy()
        finally:
            self._independent(False)
        out = []
        for zi in z:
            inv = 1.0 / np.clip(zi, rd.Z_NEAR, np.inf)
            inv = inv.reshape(rd.OUT_H, rd.IN_H // rd.OUT_H, rd.OUT_W, rd.IN_W // rd.OUT_W).mean(axis=(1, 3))
            out.append(rd.encode(1.0 / np.maximum(inv, 1e-6)))
        self.ms.append((time.perf_counter() - began) * 1000 / len(frames))
        return out

    def check(self, frames):
        batched = self.batch(frames)
        single = [self.rd.observe(f) for f in frames]
        diff = max(float(np.abs(a - b).max()) for a, b in zip(batched, single))
        self.check_result = dict(frames=len(frames), max_abs_diff_encoded=diff,
                                 mean_abs_diff=float(np.mean([np.abs(a - b).mean() for a, b in zip(batched, single)])))
        return self.check_result


class MacsimBackend:
    """Smoke test only: macsim's 2-D world and its simulated depth; a grey frame stands in for the camera."""

    def __init__(self, slots, seed):
        from sim.macsim.simulator import Simulator
        from sim.macsim.world import floor

        self.sims = [Simulator(seed=seed + i, randomize=True, corrupt=True) for i in range(slots)]
        self.n = slots
        self.floor = floor()
        self.prev = [0.0] * slots
        self.depth = None

    def is_free(self, x, y):
        return bool(self.floor.valid(x, y))

    def start(self, slot, xy, yaw, seed):
        self.sims[slot].reset((xy[0], xy[1], yaw))
        self.prev[slot] = 0.0
        return {}

    def step(self, cmds, heads):
        out = []
        for i, sim in enumerate(self.sims):
            sim.set_head(radians(heads[i] - 90))
            sim.step(*cmds[i], dt=DT)
            ob = sim.observe()
            depth = ob.depth if ob.depth is not None else np.zeros((120, 160), np.float32)
            x, y, yaw = sim.pose
            rate = (ob.gyro - self.prev[i]) / DT
            self.prev[i] = ob.gyro
            truth = dict(x=x, y=y, yaw=yaw, bumper=1.0 if not sim.bumped else 0.0, ahead_m=1.0, tilted=False,
                         stall_s=0.0, pwm=np.array(sim.drive.pwm if hasattr(sim.drive, "pwm") else cmds[i]),
                         pan=sim.head)
            out.append(dict(frame=np.full((480, 640, 3), 128, np.uint8), true_yaw_rate=rate, hfov=65.0,
                            range_cm=float(ob.range_cm), gyro_rate=rate, gyro_heading=ob.gyro, truth=truth,
                            depth_direct=depth))
        return out


def run_driver(backend, driver_name, requests, seeds, out_dir, video, log, pool=None, driver_workers=0,
               driver_kw=None):
    driver_kw = driver_kw or {}
    audit = Audit(driver_name)
    spawner = Spawner(backend.is_free)
    goal_of = lambda args: places.find_place(args["place"])[1]
    pending = sorted(((r, s) for r in requests for s in seeds),
                     key=lambda rs: not any(c["name"] == "go_to" for t in rs[0]["turns"] for c in t["calls"]))
    slots = [None] * backend.n
    if driver_workers:
        from sim.cmdtest.driverpool import DriverPool

        hosted = DriverPool(driver_name, driver_workers, driver_kw)
        drivers = [AuditedDriver(hosted.driver(i), audit) for i in range(backend.n)]
    else:
        drivers = [AuditedDriver(load_driver(driver_name, **driver_kw), audit) for _ in range(backend.n)]
    rows, trials, inputs_stats = [], [], []
    began_wall, steps, car_steps = time.perf_counter(), 0, 0
    parts = defaultdict(float)
    last_report = began_wall
    depth_job = None
    while pending or any(slots):
        for i in range(backend.n):
            if slots[i] is None and pending:
                slots[i] = start_trial(backend, i, *pending.pop(0), drivers[i], driver_name, spawner, video, pool)
        active = [s for s in slots if s is not None]
        mark = time.perf_counter()
        for s in active:
            s.link.ready.wait()
        parts["tools_s"] += time.perf_counter() - mark
        for i, s in enumerate(slots):
            if s is not None and s.link.finished:
                s.thread.join()
                trial_rows, ok = score.score_trial(s.trial, goal_of)
                s.trial["pass"] = ok
                rows += trial_rows
                trials.append(s.trial)
                c = s.inputs.counts
                inputs_stats.append(dict(request=s.trial["request"], seed=s.trial["seed"], steps=c["steps"],
                                         rgb_none=c["rgb_none"] / max(c["steps"], 1),
                                         range_none=c["range_none"] / max(c["steps"], 1),
                                         mean_age_s=c["age_sum"] / max(c["steps"] - c["rgb_none"], 1),
                                         blur_px_max=c["blur_px_max"], **s.trial["draws"]))
                if s.video is not None:
                    write_video(s, out_dir, driver_name)
                log(f"{driver_name} {s.trial['request']} seed {s.trial['seed']}: {'PASS' if ok else 'FAIL'} "
                    f"({CLOCK.now - s.trial['t_start']:.0f} s sim) "
                    + "; ".join(f"{r['tool']}={r['status']}/{r['passed']}" for r in trial_rows))
                slots[i] = None
        active = [(i, s) for i, s in enumerate(slots) if s is not None]
        if not active:
            continue
        cmds = [slots[i].link.cmd if slots[i] is not None else (0, 0) for i in range(backend.n)]
        heads = [slots[i].link.head_servo if slots[i] is not None else 90 for i in range(backend.n)]
        mark = time.perf_counter()
        raws = backend.step(cmds, heads)
        CLOCK.now += DT
        steps += 1
        car_steps += len(active)
        parts["env_s"] += time.perf_counter() - mark
        mark = time.perf_counter()
        built = []
        for i, s in active:
            raw = raws[i]
            built.append(s.inputs.observe(CLOCK.now, raw["frame"], raw["true_yaw_rate"], raw["hfov"],
                                          raw["range_cm"], raw["gyro_rate"], raw["gyro_heading"],
                                          s.link.cmd, s.link.head_moved))
        parts["inputs_s"] += time.perf_counter() - mark
        mark = time.perf_counter()
        need = [k for k, (obs, _, _) in enumerate(built) if obs.rgb is not None]
        if backend.depth is None:
            depths = {k: raws[active[k][0]]["depth_direct"] for k in need}
        else:
            ready = {}
            if depth_job is not None:
                future, owners = depth_job
                ready = {id(o): d for o, d in zip(owners, future.result())}
            depths = {k: ready[id(active[k][1].inputs)] for k in need}
            owners = [s.inputs for _, s in active if s.inputs.upcoming is not None]
            frames = [o.upcoming[1] for o in owners]
            if backend.depth.check_result is None and len(frames) >= 2:
                log(f"DEPTH_BATCH_CHECK {json.dumps(backend.depth.check(frames))}")
            depth_job = (backend.depth.submit(frames), owners) if frames else None
        parts["depth_wait_s"] += time.perf_counter() - mark
        for k, (i, s) in enumerate(active):
            raw = raws[i]
            obs, jpeg, captured = built[k]
            if k in depths:
                obs = replace(obs, depth=depths[k])
            s.link.obs = obs
            if jpeg is not None:
                s.link.frame = (captured, jpeg)
            s.link.head_moved = False
            tr = raw["truth"]
            fix = s.tools.where.fix
            est = fix.pose if fix is not None and fix.live else (np.nan, np.nan, np.nan)
            values = (CLOCK.now, tr["x"], tr["y"], tr["yaw"], tr["bumper"], tr["ahead_m"], bool(tr["tilted"]),
                      tr["stall_s"], tr["pwm"][0], tr["pwm"][1], tr["pan"], est[0], est[1])
            for k, v in zip(LOG_KEYS, values):
                s.trial["log"][k].append(float(v))
            if s.video is not None:
                s.video.append((jpeg, s.link.cmd, obs.range_cm))
            if CLOCK.now - s.trial["t_start"] > TRIAL_CAP_S:
                s.link.abort = True
            s.link.ready.clear()
            s.link.go.set()
        if time.perf_counter() - last_report > 60:
            last_report = time.perf_counter()
            wall = last_report - began_wall
            log(f"PROGRESS {driver_name} done={len(trials)} left={len(pending)} active={len(active)} steps={steps} "
                f"rtf_aggregate={car_steps * DT / wall:.2f} ms/step " + " ".join(
                    f"{k[:-2]}={1000 * v / max(steps, 1):.0f}"
                    for k, v in {**parts, **getattr(backend, "parts", {})}.items()))
    wall = time.perf_counter() - began_wall
    sim_car_s = car_steps * DT
    perf = dict(wall_s=wall, steps=steps, car_steps=car_steps, sim_car_s=sim_car_s,
                rtf_per_car=steps * DT / wall if wall else 0, rtf_aggregate=sim_car_s / wall if wall else 0,
                depth_ms_per_frame_median=float(np.median(backend.depth.ms)) if getattr(backend.depth, "ms", None)
                else None, act_ms_median=float(np.median(audit.act_ms)) if audit.act_ms else None,
                ms_per_step={k[:-2]: round(1000 * v / max(steps, 1), 1) for k, v in parts.items()},
                depth_batch_check=getattr(backend.depth, "check_result", None), slots=backend.n)
    return rows, trials, audit, inputs_stats, perf


def start_trial(backend, slot, request, seed, driver, driver_name, spawner, video, pool=None):
    xy, yaw = spawner(request, seed)
    draws = backend.start(slot, xy, yaw, seed)
    rng = np.random.default_rng([seed, zlib.crc32(request["id"].encode()), 1])
    inputs = RealInputs(rng)
    link = Link()
    driver.reset(seed)
    tools = CarTools(link, driver, seed)
    trial = dict(driver=driver_name, request=request["id"], kind=request["kind"], seed=seed, t_start=CLOCK.now,
                 events=[], log=defaultdict(list), draws={**draws, **inputs.draws()},
                 gemini_matches_request=request["gemini_matches_request"],
                 spawn=[round(float(xy[0]), 3), round(float(xy[1]), 3), round(float(yaw), 4)])

    def body():
        CLOCK.local.link = link
        if pool is not None:
            pool.bind(slot)
        try:
            link.wait(0.5)
            replay(tools, request, trial["events"])
        except Aborted:
            trial["error"] = f"aborted at the {TRIAL_CAP_S:.0f} s trial cap"
        except Exception:  # noqa: BLE001 - a crashing trial is recorded as an ERROR row, never dropped
            trial["error"] = traceback.format_exc()
        finally:
            link.finished = True
            link.ready.set()

    thread = threading.Thread(target=body, daemon=True, name=f"car{slot}")
    want = f"{request['id']}:{seed}" in video
    s = SimpleNamespace(trial=trial, link=link, inputs=inputs, tools=tools, thread=thread, video=[] if want else None)
    thread.start()
    return s


def write_video(s, out_dir, driver_name):
    import cv2

    from sim.macsim.world import floor

    name = f"{driver_name}_{s.trial['request']}_{s.trial['seed']}"
    path = out_dir / f"{name}_car_eye.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (640, 480))
    last = np.zeros((480, 640, 3), np.uint8)
    log = s.trial["log"]
    for k, (jpeg, cmd, rng) in enumerate(s.video):
        if jpeg is not None:
            last = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        frame = last.copy()
        text = f"t={log['t'][k] - s.trial['t_start']:5.1f}s pwm={cmd} range={rng if rng is not None else '-'}"
        cv2.putText(frame, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2)
        cv2.putText(frame, "new frame" if jpeg is not None else "dropped / no new frame", (8, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0) if jpeg is not None else (0, 0, 255), 1)
        writer.write(frame)
    writer.release()
    fl = floor()
    image = cv2.cvtColor(fl.image, cv2.COLOR_GRAY2BGR)
    xs, ys = np.array(log["x"]), np.array(log["y"])
    ex, ey = np.array(log["est_x"]), np.array(log["est_y"])
    goals = [places.find_place(e["args"]["place"])[1] for e in s.trial["events"] if e["name"] == "go_to"]
    allx = np.concatenate((xs, ex[np.isfinite(ex)], [g[0] for g in goals]))
    ally = np.concatenate((ys, ey[np.isfinite(ey)], [g[1] for g in goals]))
    r0, c0 = fl.cells(allx.min() - 1, ally.max() + 1)
    r1, c1 = fl.cells(allx.max() + 1, ally.min() - 1)
    r0, c0 = max(r0, 0), max(c0, 0)

    def px(x, y):
        r, c = fl.cells(np.asarray(x), np.asarray(y))
        return np.column_stack((c - c0, r - r0)).astype(np.int32)

    crop = image[r0:r1, c0:c1].copy()
    cv2.polylines(crop, [px(xs, ys)], False, (0, 160, 0), 2)
    ok = np.isfinite(ex)
    if ok.any():
        cv2.polylines(crop, [px(ex[ok], ey[ok])], False, (0, 0, 220), 1)
    for g in goals:
        cv2.circle(crop, tuple(px([g[0]], [g[1]])[0]), 25, (220, 0, 0), 2)
    x0, y0, yaw0 = s.trial["spawn"]
    cv2.arrowedLine(crop, tuple(px([x0], [y0])[0]), tuple(px([x0 + .4 * cos(yaw0)], [y0 + .4 * sin(yaw0)])[0]),
                    (0, 120, 255), 2)
    scale = max(1, 900 // max(crop.shape[:2]))
    crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    cv2.putText(crop, f"{name}: green true path, red localiser estimate, blue 0.5 m goal ring, orange start",
                (6, 18), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 0), 1)
    cv2.imwrite(str(out_dir / f"{name}_topdown.png"), crop)


def write_outputs(out_dir, driver_name, rows, trials, audit, inputs_stats, perf, seeds, script, prefix):
    header = [f"cmdtest {prefix} for driver {driver_name}; Gemini script {script['model']} recorded "
              f"{script['recorded_utc']}; seeds {list(seeds)}; one row per tool call"] + score.RULES.splitlines()
    score.write_csv(out_dir / f"{prefix}_{driver_name}.csv", rows, header)
    table = score.summary(rows, trials)
    (out_dir / f"summary_{prefix}_{driver_name}.md").write_text(
        f"# {prefix} {driver_name}\n\n{table}\n\nperf: {json.dumps(perf)}\n")
    stats = {k: [d[k] for d in inputs_stats] for k in inputs_stats[0]} if inputs_stats else {}

    def spread(key):
        v = np.array([x for x in stats.get(key, []) for x in (x if isinstance(x, list) else [x])], float)
        return [round(float(v.min()), 4), round(float(v.mean()), 4), round(float(v.max()), 4)] if len(v) else None

    record = dict(driver=driver_name, contract_fields=CONTRACT_FIELDS,
                  driver_received_types=sorted(map(list, audit.arg_types)), act_calls=audit.calls,
                  rgb_seen=sorted(audit.rgb), depth_seen=sorted(audit.depth), modes=dict(audit.modes),
                  per_trial_min_mean_max={k: spread(k) for k in (
                      "rgb_none", "range_none", "mean_age_s", "blur_px_max", "jpeg_quality", "drop_p", "exposure_s",
                      "sonar_drop_p", "camera_pitch_deg", "camera_hfov_deg", "camera_height_m", "camera_forward_m",
                      "ultrasonic_full_cone_deg", "speed_scale", "turn_scale", "deadband_pwm", "command_delay_s",
                      "gyro_bias_deg_s", "gyro_noise_deg_s", "straight_yaw_bias_deg_s")},
                  trials=len(trials), perf=perf)
    (out_dir / f"audit_{prefix}_{driver_name}.json").write_text(json.dumps(record, indent=1) + "\n")
    with (out_dir / f"trials_{prefix}_{driver_name}.jsonl").open("w") as f:
        for t in trials:
            line = {k: v for k, v in t.items() if k != "log"}
            line["log"] = {k: [round(v, 3) for v in t["log"][k]] for k in ("t", "x", "y", "yaw", "est_x", "est_y")}
            f.write(json.dumps(line, default=str) + "\n")
    return table, record


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=("isaac", "macsim"), default="isaac")
    ap.add_argument("--drivers", nargs="+", default=["hand", "policy_v2_veto"])
    ap.add_argument("--requests", nargs="*", help="request ids (default: all)")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--slots", type=int, default=8)
    ap.add_argument("--prefix", default="baseline", help="output name prefix: baseline_<driver>.csv, final_...")
    ap.add_argument("--record", nargs="*", default=[], help="request:seed pairs to record as car-eye video + plot")
    ap.add_argument("--out", type=Path, default=HERE / "results")
    ap.add_argument("--loc-workers", action="store_true", help="each car's localiser in its own process")
    ap.add_argument("--driver-workers", type=int, default=0,
                    help="host the driver under test in this many worker processes (0 = in the harness process)")
    ap.add_argument("--driver-kwargs", type=lambda text: {k: json.loads(v) for k, v in (p.split("=", 1) for p in text.split(","))},
                    default={}, help="make_driver() keyword arguments as key=value[,key=value], values JSON, e.g. "
                                     "threads=1 (execution settings only)")
    known, _ = ap.parse_known_args()
    if known.backend == "isaac":
        from isaaclab.app import add_launcher_args

        add_launcher_args(ap)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    script = json.loads(SCRIPT.read_text())
    requests = [r for r in script["requests"] if not args.requests or r["id"] in args.requests]
    logf = (args.out / f"run_{args.prefix}.log").open("a")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    pool = None
    if args.loc_workers:
        from sim.cmdtest.locpool import Pool

        pool = Pool(args.slots)
        pool.install()

    def go(backend):
        log(f"CMDTEST_READY backend={args.backend} slots={backend.n} requests={len(requests)} seeds={args.seeds} "
            f"loc_workers={pool is not None}")
        for name in args.drivers:
            rows, trials, audit, stats, perf = run_driver(backend, name, requests, args.seeds, args.out,
                                                          set(args.record), log, pool, args.driver_workers,
                                                          args.driver_kwargs)
            perf["driver_workers"] = args.driver_workers
            perf["driver_kwargs"] = args.driver_kwargs
            table, record = write_outputs(args.out, name, rows, trials, audit, stats, perf, args.seeds, script,
                                          args.prefix)
            log(f"CMDTEST_DRIVER_DONE {name}\n{table}\nAUDIT {json.dumps(record['driver_received_types'])} "
                f"fields={json.dumps(record['contract_fields'])}\nPERF {json.dumps(perf)}")

    if args.backend == "macsim":
        go(MacsimBackend(args.slots, args.seeds[0]))
        return
    import warp as wp
    from isaaclab.app import launch_simulation
    from isaaclab.sim import SimulationCfg, SimulationContext
    from isaaclab_physx.physics import PhysxCfg

    args.enable_cameras = True
    wp.config.enable_backward = False
    cfg = SimulationCfg(dt=0.01, device=args.device, physics=PhysxCfg())
    with launch_simulation(cfg, args):
        go(IsaacBackend(SimulationContext(cfg), args.slots, args.seeds[0]))
    log("CMDTEST_ALL_DONE")


if __name__ == "__main__":
    main()

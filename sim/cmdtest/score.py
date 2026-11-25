"""Per-call scoring on simulator truth (the only place truth is read), and the CSV / summary writers."""

import csv
import json
from collections import defaultdict
from math import degrees, hypot

import numpy as np

RULES = """\
Scoring per tool call, on simulator truth only (sim/cmdtest/score.py):
- go_to: PASS = the tool reported arrived AND the true pose is within 0.5 m of the place AND within the 300 s
  navigator limit. false_arrival = reported arrived while the true pose is > 1.0 m from the place.
- drive_forward / drive_back N cm: PASS = true path length within +-20 % of N (at least +-8 cm) AND true heading
  change < 15 deg; or (forward only) a justified early stop: the tool stopped for an obstacle and the true gap from
  the sonar emitter to the nearest obstacle within +-15 deg ahead is < 25 cm.
- turn D deg: PASS = true signed yaw change within +-15 deg of D.
- explore / patrol: PASS = every full 10 s window has >= 0.5 m of true path (a final window of >= 3 s pro rata)
  AND no stall (wheels driving straight >= 55 PWM, < 2.5 cm/s true speed) lasting > 3 s.
- stop: PASS = the PWM the wheels actually run is (0, 0) within 0.5 s of the stop call.
- look_direction D deg: PASS = true head pan within +-10 deg of D after the call AND < 3 cm true path.
- Calls cut by a later call of the script (a spoken stop, or the next call after a background job) are
  "interrupted": explore / patrol are still scored over the time they ran, other kinds are not scored.
  Calls dropped by car_agent.py's repeat guard are "skipped_repeat", not scored. where_am_i / task_status: no motion.
- A request PASSES when every scored call passes (and at least one call was scored).
- contacts = onsets of the bumper-ring gap <= 1 cm (train/driver_env.py's measure), per metre of true path;
  tilts = onsets of roll or pitch > 6 deg.
"""

FIELDS = ["driver", "request", "kind", "seed", "turn", "call", "tool", "args", "status", "why", "passed", "score_note",
          "t0_s", "dur_s", "path_m", "contacts", "tilts", "false_arrival", "goal_err_m", "est_err_m", "target",
          "measured", "request_pass", "gemini_matches_request", "result"]


def unwrap(yaw):
    return np.unwrap(np.asarray(yaw, float))


def window(log, t0, t1):
    t = log["t"]
    return np.flatnonzero((t >= t0 - 1e-6) & (t <= t1 + 1e-6))


def onsets(flags):
    flags = np.asarray(flags, bool)
    return int(np.sum(flags[1:] & ~flags[:-1]) + (flags[0] if len(flags) else 0))


def score_call(ev, log, goal_of):
    idx = window(log, ev["t0"], ev["t1"])
    if len(idx) == 0:
        idx = np.array([max(0, np.searchsorted(log["t"], ev["t0"]) - 1)])
    x, y, yaw = log["x"][idx], log["y"][idx], unwrap(log["yaw"])[idx]
    path = float(np.sum(np.hypot(np.diff(x), np.diff(y)))) if len(idx) > 1 else 0.0
    out = dict(path_m=path, contacts=onsets(log["bumper"][idx] <= .01), tilts=onsets(log["tilted"][idx]),
               false_arrival="", goal_err_m="", est_err_m="", target="", measured="", passed="", score_note="")
    name, status = ev["name"], ev.get("status", "")
    if status in ("skipped_repeat", "no_motion", "unsupported", "refused") and name != "stop":
        out["score_note"] = status
        if status in ("unsupported", "refused"):
            out["passed"] = False
        return out
    if status == "interrupted" and name not in ("explore", "patrol", "stop"):
        out["score_note"] = "interrupted"
        return out
    args = ev["args"]
    if name == "go_to":
        gx, gy = goal_of(args)
        err = hypot(x[-1] - gx, y[-1] - gy)
        said = status == "arrived"
        out.update(goal_err_m=round(err, 3), target=f"{gx:.2f},{gy:.2f}", measured=f"{x[-1]:.2f},{y[-1]:.2f}",
                   false_arrival=said and err > 1.0)
        if "est" in ev and np.all(np.isfinite(ev["est"][:2])):
            out["est_err_m"] = round(hypot(ev["est"][0] - x[-1], ev["est"][1] - y[-1]), 3)
        out["passed"] = said and err <= .5 and ev.get("nav_s", 0) <= 300
    elif name in ("drive_forward", "drive_back"):
        lo, hi = (5, 300) if name == "drive_forward" else (5, 40)
        want = max(lo, min(hi, int(args["centimeters"]))) / 100
        turned = abs(degrees(yaw[-1] - yaw[0]))
        tol = max(.2 * want, .08)
        ok = abs(path - want) <= tol and turned < 15
        justified = (name == "drive_forward" and ev.get("why", "").startswith("obstacle")
                     and float(log["ahead_m"][idx[-1]]) < .25)
        out.update(target=f"{want * 100:.0f}cm", measured=f"{path * 100:.0f}cm/{turned:.0f}deg",
                   passed=bool(ok or justified), score_note="justified_stop" if justified and not ok else "")
    elif name == "turn":
        want = max(5, min(180, abs(int(args["degrees"])))) * (1 if args["direction"] == "left" else -1)
        got = degrees(yaw[-1] - yaw[0])
        out.update(target=f"{want}deg", measured=f"{got:.0f}deg", passed=abs(got - want) <= 15)
    elif name in ("explore", "patrol"):
        t = log["t"][idx]
        seg = np.concatenate(([0.], np.cumsum(np.hypot(np.diff(x), np.diff(y)))))
        ok, worst = True, np.inf
        for start in np.arange(t[0], t[-1], 10.):
            end = min(start + 10., t[-1])
            if end - start < 3. and start > t[0]:
                break
            moved = np.interp(end, t, seg) - np.interp(start, t, seg)
            need = .5 * (end - start) / 10.
            worst = min(worst, moved / max(need, 1e-6))
            ok &= moved >= need
        stall = float(np.max(log["stall_s"][idx]))
        out.update(target=">=0.5m/10s", measured=f"{path:.2f}m/{t[-1] - t[0]:.0f}s stall{stall:.1f}s",
                   passed=bool(ok and stall <= 3.))
    elif name == "stop":
        later = np.flatnonzero((log["t"] >= ev["t0"] - 1e-6) & (log["pwm_l"] == 0) & (log["pwm_r"] == 0))
        delay = float(log["t"][later[0]] - ev["t0"]) if len(later) else np.inf
        out.update(target="<=0.5s", measured=f"{delay:.2f}s", passed=delay <= .5)
    elif name == "look_direction":
        want = 0 if args["direction"] == "forward" else max(10, min(80, abs(int(args.get("degrees", 45))))) * (
            1 if args["direction"] == "left" else -1)
        pan = degrees(float(log["pan"][idx[-1]]))
        out.update(target=f"{want}deg", measured=f"{pan:.0f}deg", passed=abs(pan - want) <= 10 and path < .03)
    else:
        out["score_note"] = "not scored"
    return out


def score_trial(trial, goal_of):
    rows = []
    log = {k: np.asarray(v) for k, v in trial["log"].items()}
    for ev in trial["events"]:
        if "t1" not in ev:
            ev["t1"] = float(log["t"][-1]) if len(log["t"]) else ev["t0"]
            ev.setdefault("status", "unfinished")
        s = score_call(ev, log, goal_of)
        rows.append(dict(driver=trial["driver"], request=trial["request"], kind=trial["kind"], seed=trial["seed"],
                         turn=ev["turn"], call=ev["index"], tool=ev["name"],
                         args=json.dumps(ev["args"], ensure_ascii=False), status=ev.get("status", ""),
                         why=ev.get("why", ev.get("reason", "")), t0_s=round(ev["t0"] - trial["t_start"], 1),
                         dur_s=round(ev["t1"] - ev["t0"], 1), result=ev.get("result", ""), **s))
    if trial.get("error"):
        rows.append(dict(driver=trial["driver"], request=trial["request"], kind=trial["kind"], seed=trial["seed"],
                         tool="ERROR", status="error", passed=False, result=trial["error"][-300:]))
    scored = [r["passed"] for r in rows if r.get("passed") != ""]
    ok = bool(scored) and all(scored)
    for r in rows:
        r["request_pass"] = ok
        r["gemini_matches_request"] = trial["gemini_matches_request"]
        for k in ("path_m",):
            if isinstance(r.get(k), float):
                r[k] = round(r[k], 3)
    return rows, ok


def write_csv(path, rows, header_lines):
    with path.open("w", newline="") as f:
        for line in header_lines:
            f.write(f"# {line}\n")
        w = csv.DictWriter(f, FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def summary(rows, trials):
    by_kind = defaultdict(lambda: dict(req=0, req_ok=0, calls=0, calls_ok=0, path=0., contacts=0, fa=0, tilts=0,
                                       dur=[]))
    for t in trials:
        k = by_kind[t["kind"]]
        k["req"] += 1
        k["req_ok"] += t["pass"]
    for r in rows:
        k = by_kind[r["kind"]]
        if r.get("passed") != "":
            k["calls"] += 1
            k["calls_ok"] += bool(r["passed"])
        if isinstance(r.get("path_m"), float):
            k["path"] += r["path_m"]
        k["contacts"] += int(r.get("contacts") or 0)
        k["tilts"] += int(r.get("tilts") or 0)
        k["fa"] += r.get("false_arrival") is True
        if r.get("dur_s") not in (None, "") and r.get("tool") not in ("stop", "ERROR"):
            k["dur"].append(float(r["dur_s"]))
    lines = ["| command type | requests passed | calls passed | contacts/m | false arrivals | tilts | median call s |",
             "|---|---|---|---|---|---|---|"]
    total = dict(req=0, req_ok=0, calls=0, calls_ok=0, path=0., contacts=0, fa=0, tilts=0, dur=[])
    for kind in sorted(by_kind):
        k = by_kind[kind]
        for key in total:
            total[key] += k[key]
        lines.append(row_line(kind, k))
    lines.append(row_line("**all**", total))
    return "\n".join(lines)


def row_line(kind, k):
    cpm = k["contacts"] / k["path"] if k["path"] > 0 else 0.
    med = float(np.median(k["dur"])) if k["dur"] else 0.
    pct = 100 * k["req_ok"] / max(k["req"], 1)
    return (f"| {kind} | {k['req_ok']}/{k['req']} ({pct:.0f} %) | {k['calls_ok']}/{k['calls']} | {cpm:.3f} | "
            f"{k['fa']} | {k['tilts']} | {med:.1f} |")

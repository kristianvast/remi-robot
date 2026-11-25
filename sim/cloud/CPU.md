# elegoo-sim-3: the cloud CPU machine for simulator work (2026-11-24)

Owner's rule: heavy runs (evaluations, sweeps, test batches, anything multi-process or longer than about a minute)
run here, NOT on the Mac. The Mac keeps editing, reading results, single short commands, and anything that needs the
microphone, speaker, or the owner's Google / ElevenLabs credentials.

- Machine: `elegoo-sim-3`, project your-gcp-project, zone us-central1-c, c3d-highcpu-60 (60 cores, no GPU),
  Ubuntu 24.04. List price roughly 2 to 2.5 USD an hour (not read from billing). It powers itself off after
  45 minutes with load under 0.5. No Google or ElevenLabs credentials are on it: do not copy any there.
- The project is mirrored at `~/elegoo` with the same layout as the Mac's project root (sim/macsim, sim/localize,
  sim/navigate, sim/isaac/elegoo, sim/isaac/scene, scans/merged, .omo/waves/p1 and p3, and
  the entire car/tools/brain tree, including `dashboard/`). No .venv, cache,
  .env or video is copied.
- Python: `sim/macsim/.venv` (built with `uv sync` from the lock file) and
  `car/tools/brain/.venv` (requirements.txt + aiohttp, person detector model installed).
- Checked by the lead: `python -m sim.macsim.verify` there gives the same numbers as on the Mac (27.0 / 39.0 cm/s,
  130 deg/s, 0.4 s, depth median error 3 mm). One core runs 179 simulator steps a second (the Mac: 298), so use
  many processes: about 56 workers.

## Use (from the Mac, project root)
```
sim/cloud/cpu.sh status                      # RUNNING or TERMINATED
sim/cloud/cpu.sh start                       # if it powered itself off
sim/cloud/cpu.sh push                        # copy current code and data to ~/elegoo (do this after every edit)
sim/cloud/cpu.sh run 'sim/macsim/.venv/bin/python -m sim.navigate.evaluate_nav'    # runs in ~/elegoo
sim/cloud/cpu.sh pull .omo/waves/p3/navigate-eval.csv sim/navigate/plots           # copy results back
```
- `run` blocks until the command ends. For anything over a minute, start it detached and poll a log, for example:
  `cpu.sh run 'nohup sim/macsim/.venv/bin/python -m sim.navigate.evaluate_nav > nav.log 2>&1 & echo started'`
  then `cpu.sh run 'tail -3 nav.log'`.
- If a script has a fixed worker count (14 was used on the Mac), raise it for this run (an argument or an
  environment variable) instead of leaving 46 cores idle.
- `push` overwrites files on the machine with the Mac's copy and never deletes; `pull` overwrites the Mac's copy.
  Two workers share the machine: keep to your own folders and ports (navigate: sim/navigate; stall: tools/brain,
  .omo/waves/p3/stall, ports 18200-18399).
- Stand-in car tests: start the stand-in and the test in ONE `run` command (background the stand-in with `&`, kill
  it at the end), as the stand-in dies when the ssh command ends.
- Tests that call Gemini or ElevenLabs (verify_autonomy.py's object search, smoke_voice.py) cannot run there. Run
  those once on the Mac; they are light.
- Do not stop the machine when you finish: another worker may be using it. The idle rule and the lead handle that.
- Never touch elegoo-sim-1 (training) or elegoo-sim-2, or any machine not named elegoo-sim-*.
- Report the machine minutes you used.

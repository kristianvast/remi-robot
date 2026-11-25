# Real-car session, 2026-11-24 (owner present)

Before starting: port 8765 on the Mac is taken by a `python -m http.server` serving recon/ (not started by the lead),
so the dashboard runs on 8766. Nothing else may read the car's camera during the session (one reader only).

1. Home spot: the owner names the room and the exact spot; the lead records it in sim/navigate/places.yaml as `home`.
2. Camera calibration, about 5 min (car parked, agent NOT running, so the calibration is the camera's only reader):
   sim/realdepth/.venv/bin/python sim/realcar/calibrate.py capture --url http://elegoo-car.local/capture
   About 15 pictures of the A4 board (sim/realcar/checkerboard_A4_9x6_25mm.svg printed at 100 %), near and far,
   tilted, in every part of the picture. Then:
   sim/realdepth/.venv/bin/python sim/realcar/calibrate.py solve      -> sim/realcar/camera.json (rms under 0.5 px)
3. Stall recording, about 5 min: tools/brain/stall_probe.py (detector thresholds in tools/brain/stall.py; command in .omo/waves/p3/stall.md once written):
   car held still, pushed by hand, driven freely from the pane, driven into a soft obstacle.
4. Voice on Gemini Live, about 15 min, from tools/brain (command in .omo/waves/p3/gemini-live.md once written), with
   DASHBOARD_PORT=8766 CAR_HOST=elegoo-car.local: talk, single moves, look around, explore 30 s, follow the owner,
   find an object, mute with M, emergency stop with Space.
5. Afterwards: stop the agent, save its log to .omo/waves/p3/realcar-session-2.log, commit what passed.

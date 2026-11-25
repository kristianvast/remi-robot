#!/usr/bin/env bash
# Add one processed clip to the texture bank and rebuild the visual scene (sim/isaac/scene/building_visual.usda).
#   sim/rooms/add_to_scene.sh IMG_4479
# Needs sim/rooms/<clip>/ from process.sh (room_aligned.usd, cam2world_aligned.npy, transform.json, sim_ready.json).
# Bank crops come from sim/rooms/bank/<clip>_bank.json (process.sh makes them on the GPU machine); if they are
# missing they are made on elegoo-sim-3 from ~/rooms/out/<clip> (read only). texture_bank.py and build_visual.py
# then run on elegoo-sim-3 in ~/visuals (CPU, under a minute) and the scene, metadata and bank are copied back.
# A clip that is not sim-ready still adds its crops to the bank but no filmed room.
set -euo pipefail
clip=$1
P=$(cd "$(dirname "$0")/../.." && pwd)
VM=elegoo-sim-3; G="--zone us-central1-c --project your-gcp-project"
W='~/visuals'
ssh_vm() { gcloud compute ssh $VM $G --quiet --command "$1" 2>&1 | grep -v '^Warning\|^Updating\|^Existing host' || true; }
[ -f "$P/sim/rooms/$clip/transform.json" ] || { echo "no sim/rooms/$clip/transform.json: run process.sh first"; exit 1; }
[ "$("$P/sim/cloud/cpu.sh" status | cut -f1)" = RUNNING ] || "$P/sim/cloud/cpu.sh" start

ssh_vm "export PATH=\$HOME/.local/bin:\$PATH; mkdir -p $W/proj; cd $W && (test -x venv/bin/python || (uv venv -q venv -p 3.12 && uv pip install -q -p venv/bin/python numpy scipy trimesh usd-core opencv-python-headless shapely pillow pyyaml)) && echo venv ready"
if ! gcloud compute ssh $VM $G --quiet --command "test -f $W/proj/scans/merged/building.glb" 2>/dev/null; then
  (cd "$P" && COPYFILE_DISABLE=1 tar -czf - scans/merged/building.glb sim/isaac/scene/textures sim/isaac/scene/cache/samples.npz) |
    gcloud compute ssh $VM $G --quiet --command "tar -C $W/proj -xzf -"
fi
rooms=(); for d in "$P"/sim/rooms/IMG_*/; do for f in room_aligned.usd cam2world_aligned.npy transform.json sim_ready.json; do
  [ -f "$d$f" ] && rooms+=("sim/rooms/$(basename "$d")/$f"); done; done
(cd "$P" && COPYFILE_DISABLE=1 tar -czf - scans/merged/floor_map.png scans/merged/floor_map.yaml scans/merged/rooms_next.json \
  sim/isaac/scene/*.py sim/isaac/scene/scene_meta.yaml sim/isaac/scene/layers/shell.usda \
  sim/rooms/texture_bank.py sim/rooms/vm/bank_crops.py sim/rooms/bank "${rooms[@]}") |
  gcloud compute ssh $VM $G --quiet --command "tar -C $W/proj -xzf - && echo uploaded"

if [ ! -f "$P/sim/rooms/bank/${clip}_bank.json" ]; then
  ssh_vm "test -d ~/rooms/out/$clip/frames/images || { echo 'NO_FRAMES: ~/rooms/out/$clip is not on $VM; re-run process.sh (it makes the crops on the GPU machine)'; exit 1; }
          cd $W/proj && ../venv/bin/python sim/rooms/vm/bank_crops.py ~/rooms/out/$clip sim/rooms/bank --name $clip | grep -E 'BANK_OK|BANK_MISMATCH|Error'"
fi
ssh_vm "cd $W/proj && ../venv/bin/python sim/rooms/texture_bank.py && cd sim/isaac/scene && ../../../../venv/bin/python build_visual.py | grep -E 'CLIP|ROOM|VISUAL_OK|Error'"
(cd "$P" && gcloud compute ssh $VM $G --quiet --command "cd $W/proj && tar -czf - sim/isaac/scene/building_visual.usda sim/isaac/scene/layers/visual \
  sim/isaac/scene/visual_meta.json sim/isaac/scene/proof/visual_poses.json sim/isaac/scene/cache/visual_plan.png sim/rooms/bank" 2>/dev/null | tar -xzf -)
python3 -c "import json;m=json.load(open('$P/sim/isaac/scene/visual_meta.json'));b=json.load(open('$P/sim/rooms/bank/index.json'));print('ADDED $clip: bank', sum(e['clip']=='$clip' for e in b['entries']), 'crops of', len(b['entries']), '| filmed rooms', [f['clip'] for f in m['filmed']])"

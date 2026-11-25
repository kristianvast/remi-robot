#!/usr/bin/env bash
# One clip -> metric DA3 room aligned into the building scan, written to sim/rooms/<clip>/.
#   sim/rooms/process.sh ~/Downloads/IMG_4479.MOV ["office"]
# The area name (or area_NN) is optional; without it the whole building is searched.
# Heavy work runs on elegoo-sim-2 (started if needed, stopped at the end unless KEEP_VM=1).
# Another GPU machine: VM=elegoo-sim-4 ZONE=us-east4-c sim/rooms/process.sh ...
# FORCE=1 redoes extraction, DA3 and alignment for a clip that was already processed on that machine.
# GPU: L4 24 GB or A100 40 GB. A 16 GB T4 cannot run DA3 multi-view (tested 2026-11-24: out of memory even at 55 frames).
# DA3_FRAMES="150 120 90" are the frame counts tried in order on GPU out-of-memory.
set -euo pipefail
clip=$1; area=${2:-}
name=$(basename "$clip"); name=${name%.*}
[ "$name" = IMG_4474 ] && { echo "IMG_4474 (whole-building walk) is excluded by the owner's decision"; exit 1; }
P=$(cd "$(dirname "$0")/../.." && pwd)
VM=${VM:-elegoo-sim-2}; Z=${ZONE:-us-central1-c}; PR=your-gcp-project
[[ $VM == elegoo-sim-* && $VM != elegoo-sim-1 && $VM != elegoo-sim-3 ]] || { echo "refusing machine $VM"; exit 1; }
G="--zone $Z --project $PR"
status() { gcloud compute instances describe $VM $G --format='value(status)'; }
ssh_vm() { gcloud compute ssh $VM $G --command "$1" -- -o ServerAliveInterval=30 -o ServerAliveCountMax=20; }

if [ "$(status)" != RUNNING ]; then
  for i in $(seq 1 11); do
    gcloud compute instances start $VM $G >/dev/null 2>&1 || true
    [ "$(status)" = RUNNING ] && break
    echo "start attempt $i: $(status) (L4 stockout?), retrying in 2 min"; sleep 120
  done
  [ "$(status)" = RUNNING ] || { echo "$VM did not start"; exit 1; }
fi
t0=$(date +%s)
finish() { echo "machine seconds for this clip: $(( $(date +%s) - t0 ))"
  [ "${KEEP_VM:-0}" = 1 ] || { gcloud compute instances stop $VM $G >/dev/null 2>&1; echo "$VM: $(status)"; }; }
trap finish EXIT
for i in $(seq 1 30); do ssh_vm true >/dev/null 2>&1 && break; sleep 10; done
ssh_vm "mkdir -p ~/rooms/vm ~/rooms/clips ~/rooms/scan ~/logs"
gcloud compute scp $G "$P"/sim/rooms/vm/*.py "$P"/sim/rooms/vm/*.sh "$P"/recon/vm/common.py "$P"/recon/render.py $VM:rooms/vm/
if ! ssh_vm "test -f ~/rooms/scan/building.glb"; then
  gcloud compute scp $G "$P"/scans/merged/{building.glb,floor_map.png,floor_map.yaml,rooms_next.json} \
    "$P"/scans/merged/views/poses_rendered.json "$P"/sim/navigate/places.yaml $VM:rooms/scan/
fi
ssh_vm "test -f ~/rooms/clips/$(basename "$clip")" || gcloud compute scp $G "$clip" $VM:rooms/clips/
for d in "$P"/sim/rooms/IMG_*/; do
  c=$(basename "$d"); [ "$c" = "$name" ] && continue
  [ -f "$d/assess/occ.npz" ] && [ -f "$d/sim_ready.json" ] || continue
  ssh_vm "mkdir -p ~/rooms/out/$c/assess ~/rooms/out/$c/aligned"
  gcloud compute scp $G "$d/assess/occ.npz" $VM:rooms/out/$c/assess/ && gcloud compute scp $G "$d/sim_ready.json" $VM:rooms/out/$c/aligned/
done
ssh_vm "bash ~/rooms/vm/setup_vm.sh > ~/logs/rooms_setup.log 2>&1 || { tail -20 ~/logs/rooms_setup.log; exit 1; }"
ssh_vm "FORCE='${FORCE:-0}' DA3_FRAMES='${DA3_FRAMES:-150 120 90}' DA3_MODEL='${DA3_MODEL:-depth-anything/DA3NESTED-GIANT-LARGE-1.1}' bash ~/rooms/vm/run_clip.sh $name '$area' 2>&1 | tee ~/logs/rooms_$name.log | grep -E 'EXTRACT|DA3_OK|ALIGN_OK|FUSE_OK|RUN_|Error|failed'"

out="$P/sim/rooms/$name"
mkdir -p "$out/assess"
R="rooms/out/$name"
gcloud compute scp $G $VM:$R/frames/frames.json $VM:$R/da3/{stats.json,render.png,cam2world.npy,cloud.ply} \
  $VM:$R/aligned/{transform.json,align_topdown.png,cam2world_aligned.npy,cloud_aligned.ply,room_aligned.glb,room_aligned.usd,sim_ready.json} "$out/"
gcloud compute scp --recurse $G $VM:$R/assess/{assess.json,occ.npz,views_sheet.jpg,occupancy.png,slice.png,views} "$out/assess/"
gcloud compute scp $G $VM:rooms/out/maps/{floor_map_rooms.png,floor_map_rooms.yaml,floor_map_rooms_add.png,floor_map_rooms_add.yaml,floor_map_rooms.json,map_compare_$name.png} "$P/sim/rooms/" || echo "no merged map entry for $name (not sim-ready)"
mkdir -p "$P/sim/rooms/bank"
gcloud compute scp $G "$VM:$R/bank/${name}_*" "$P/sim/rooms/bank/" || echo "no bank crops for $name"
python3 -c "import json;t=json.load(open('$out/transform.json'));print('area',t['area_by_alignment'],'scale',round(t['scale'],3),'residual cm',{k:round(v,1) for k,v in t['residual_cm'].items() if isinstance(v,float)},'coverage',t['coverage'])"
echo "DONE $out"
[ "${NO_SCENE:-0}" = 1 ] || "$P/sim/rooms/add_to_scene.sh" "$name"

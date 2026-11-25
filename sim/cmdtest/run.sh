#!/usr/bin/env bash
# The command test in one command: push, run on L4 machines in parallel, pull, stop the machines.
#   sim/cmdtest/run.sh hand policy_v2_veto                          all drivers on one machine, one after another
#   sim/cmdtest/run.sh --vm elegoo-sim-6 hand --vm elegoo-sim-10 policy_v2_veto     one group per machine, in parallel
#   sim/cmdtest/run.sh smoke [harness args]                          macsim plumbing test on elegoo-sim-3, no GPU
#   sim/cmdtest/run.sh hash | stamp <dir> <machine>                   print the harness hash | stamp pulled results
# Drivers: hand, policy_v2_veto (today's), or any sim/models/<name>/driver.py exposing make_driver().
# Environment: CMDTEST_VM (machine for drivers before any --vm, default elegoo-sim-6), CMDTEST_ARGS (extra harness
# args, e.g. "--prefix final --record goto_kitchen:101"), KEEP_VM=1 to leave the machines running.
set -euo pipefail
P="$(cd "$(dirname "$0")/../.." && pwd)"
source "$P/sim/cloud/_common.sh"
IMAGE=elegoo-cmdtest:1
WORK=p4cmd
BRAIN=car/tools/brain

stage() {
  local extra=()
  for d in "$@"; do [ -d "$P/sim/models/$d" ] && extra+=("sim/models/$d"); done
  COPYFILE_DISABLE=1 tar -C "$P" --no-xattrs --exclude .venv --exclude .uv-cache --exclude __pycache__ --exclude results --exclude plots \
    --exclude '*.log' --exclude cache --exclude proof --exclude benchmark --exclude source --exclude patrols \
    --exclude 'sim/isaac/elegoo/train/results' --exclude run.sh.new -czf - \
    sim/cmdtest sim/navigate sim/localize sim/macsim sim/policy/driver.onnx sim/policy/driver_v2.onnx \
    sim/policy/run_driver.py sim/realdepth/realdepth.py sim/realdepth/observation_spec.md sim/isaac/elegoo \
    sim/isaac/scene sim/rooms/bank scans/merged/collision.obj scans/merged/floor_map.png scans/merged/floor_map.yaml \
    scans/merged/rooms.json scans/merged/rooms_next.json $BRAIN ${extra[@]+"${extra[@]}"}
}

harness_hash() {   # the harness, the Gemini script, and the shared code it imports or reads (not the driver lanes' own dirs)
  (cd "$P" && {
    ls sim/cmdtest/*.py sim/cmdtest/gemini_script.json sim/cmdtest/remote/Dockerfile sim/navigate/places.yaml \
       sim/policy/*.onnx sim/policy/run_driver.py sim/realdepth/realdepth.py sim/isaac/elegoo/calibration.yaml \
       sim/isaac/elegoo/train/routes.npz sim/isaac/elegoo/nav/building_plan.json sim/isaac/elegoo/nav/free_map.npz \
       "$BRAIN"/{drive,behaviors,robot,navigation,stall,places,depth_feed,safe_drive,follow,patrol,find_object,trips}.py \
       "$BRAIN"/{telemetry,pilot,persona,voice,car_camera,car,car_agent,fake_world,fake_camera}.py
    find sim/navigate sim/localize sim/macsim sim/isaac/elegoo -name '*.py' -not -path '*/.venv/*' \
      -not -path '*/.uv-cache/*' -not -path '*/source/*' -not -path '*/benchmark/*' \
      -not -path 'sim/isaac/elegoo/navdp/*' -not -path 'sim/isaac/elegoo/combined/*'
  } | LC_ALL=C sort | xargs shasum -a 256 | shasum -a 256 | cut -c1-16)
}

if [ "${1:-}" = smoke ]; then
  shift
  CPU=elegoo-sim-3
  gcloud compute ssh "$CPU" --project "$PROJECT" --zone us-central1-c --quiet --command "rm -rf ~/p4-cmdtest/sim/cmdtest && mkdir -p ~/p4-cmdtest" </dev/null
  stage | gcloud compute ssh "$CPU" --project "$PROJECT" --zone us-central1-c --quiet --command "tar -C ~/p4-cmdtest -xzf - && echo pushed"
  gcloud compute ssh "$CPU" --project "$PROJECT" --zone us-central1-c --quiet --command \
    "cd ~/p4-cmdtest && OMP_NUM_THREADS=1 ~/elegoo/$BRAIN/.venv/bin/python sim/cmdtest/harness.py --backend macsim --out /tmp/p4-cmdtest-smoke $*" </dev/null
  exit 0
fi

run_vm() {   # run_vm <machine> <driver>... : the whole pipeline for one machine; results land in sim/cmdtest/results
  pick_name "$1"
  shift
  require_zone
  local tag="$NAME" hash="$HASH" pull
  if [ "$(status_of)" != RUNNING ]; then
    gcloud compute instances start "$NAME" --zone "$ZONE" --project "$PROJECT" --quiet
    for _ in $(seq 1 30); do gssh --command true 2>/dev/null && break; sleep 10; done
  fi
  echo "$(date -u +%H:%M:%S)Z $tag ($ZONE) running: $* (harness $hash)"
  gcloud compute scp --project "$PROJECT" --zone "$ZONE" --quiet "$P/sim/cmdtest/remote/Dockerfile" "$NAME":/tmp/cmdtest.Dockerfile
  gssh --command "sudo docker image inspect $IMAGE >/dev/null 2>&1 || sudo docker build -t $IMAGE -f /tmp/cmdtest.Dockerfile /tmp 2>&1 | tail -3"
  if ! gssh --command "test -d ~/elegoo-sim/hf/hub/models--apple--DepthPro2-S"; then
    COPYFILE_DISABLE=1 tar -C "$P/sim/realdepth/models" --no-xattrs -czf - hf/hub | gssh --command "mkdir -p ~/elegoo-sim && tar -C ~/elegoo-sim -xzf -"
  fi
  gssh --command "sudo rm -rf ~/elegoo-sim/work/$WORK && mkdir -p ~/elegoo-sim/work/$WORK"
  stage "$@" | gssh --command "tar -C ~/elegoo-sim/work/$WORK -xzf - && chmod -R a+rwX ~/elegoo-sim/work/$WORK && echo pushed"
  local run_log="~/elegoo-sim/work/$WORK-run.log"
  gssh --command "nohup sudo docker run --rm --gpus all --network=host --name cmdtest \
    -e ACCEPT_EULA=Y -e NVIDIA_DRIVER_CAPABILITIES=all -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 -e OMP_NUM_THREADS=1 \
    -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
    -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
    -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
    -v ~/elegoo-sim/hf:/hf:rw -v ~/elegoo-sim/work:/workspace/elegoo:rw -w /workspace/elegoo/$WORK \
    --entrypoint bash $IMAGE -c '/workspace/isaaclab/isaaclab.sh -p sim/cmdtest/harness.py --backend isaac \
    --drivers $* --viz none --slots 24 --loc-workers ${CMDTEST_ARGS:-}' > $run_log 2>&1 < /dev/null & echo started"
  echo "WORKING: cmdtest - $tag running $*"
  while sleep 60; do
    state="$(gssh --command "sudo docker ps -q -f name=cmdtest | wc -l; grep -E 'PROGRESS|PASS|FAIL|Traceback|Error' $run_log | tail -n 1" 2>/dev/null || true)"
    echo "$(date -u +%H:%M:%S)Z $tag $(echo "$state" | tail -n 1)"
    [ "$(echo "$state" | head -n 1)" = 0 ] && break
  done
  pull="$(mktemp -d)"
  gcloud compute scp --project "$PROJECT" --zone "$ZONE" --quiet --recurse \
    "$NAME":"~/elegoo-sim/work/$WORK/sim/cmdtest/results/*" "$pull/" || true
  gcloud compute scp --project "$PROJECT" --zone "$ZONE" --quiet "$NAME":"$run_log" "$pull/$WORK-run-$tag.log" || true
  stamp_and_collect "$pull" "$tag" "$hash"
  if [ "${KEEP_VM:-0}" != 1 ]; then
    gcloud compute instances stop "$NAME" --zone "$ZONE" --project "$PROJECT" --quiet
    echo "$tag: $(status_of)"
  fi
  grep -q CMDTEST_ALL_DONE "$pull/$WORK-run-$tag.log" || die "$tag: run did not finish; see sim/cmdtest/results/$WORK-run-$tag.log"
}

driver_hashes() {   # driver_hashes <name>: sha256 of a sim/models driver's driver.py, policy.py and weights files
  (cd "$P/sim/models/$1" && find . -maxdepth 2 -type f \( -name driver.py -o -name policy.py -o -path './weights/*.pt' \
    -o -path './weights/*.onnx' -o -path './weights/*.pth' \) | sed 's|^\./||' | LC_ALL=C sort \
    | xargs shasum -a 256 | awk '{printf "%s=%s ", $2, $1}') || true
}

stamp_and_collect() {   # stamp_and_collect <pulled dir> <machine> <hash>: hash into headers, per-machine log names
  local dir="$1" tag="$2" hash="$3" f
  for f in "$dir"/*.csv; do
    [ -e "$f" ] || continue
    { echo "# harness_hash $hash (sim/cmdtest/run.sh harness_hash: harness, Gemini script, imported shared code); machine $tag"; cat "$f"; } > "$f.tmp" && mv "$f.tmp" "$f"
  done
  for f in "$dir"/*.csv; do
    [ -e "$f" ] || continue
    local base drv line
    base="$(basename "$f" .csv)"; drv="${base#*_}"
    [ -d "$P/sim/models/$drv" ] || continue
    line="$(driver_hashes "$drv")"
    { head -n 1 "$f"; echo "# driver $drv sha256 $line"; tail -n +2 "$f"; } > "$f.tmp" && mv "$f.tmp" "$f"
  done
  for f in "$dir"/summary_*.md; do [ -e "$f" ] && printf '\nharness_hash %s, machine %s\n' "$hash" "$tag" >> "$f"; done
  for f in "$dir"/audit_*.json; do
    [ -e "$f" ] && python3 -c "import json,sys; p=sys.argv[1]; d=json.load(open(p)); d['harness_hash']=sys.argv[2]; d['machine']=sys.argv[3]; open(p,'w').write(json.dumps(d, indent=1)+'\n')" "$f" "$hash" "$tag"
  done
  for f in "$dir"/run_*.log; do [ -e "$f" ] && mv "$f" "${f%.log}-$tag.log"; done
  mkdir -p "$P/sim/cmdtest/results"
  cp -R "$dir"/. "$P/sim/cmdtest/results/"
}

case "${1:-}" in
  hash) harness_hash; exit 0 ;;
  driverhash) driver_hashes "$2"; echo; exit 0 ;;
  stamp) stamp_and_collect "$2" "$3" "$(harness_hash)"; exit 0 ;;
esac
[ $# -ge 1 ] || { sed -n 2,9p "$0"; exit 2; }
groups=()
current="${CMDTEST_VM:-elegoo-sim-6}"
drivers=""
while [ $# -gt 0 ]; do
  if [ "$1" = --vm ]; then
    [ -n "$drivers" ] && groups+=("$current $drivers")
    current="$2"; drivers=""; shift 2; continue
  fi
  case "$1" in hand|policy_v2_veto) ;; *) [ -f "$P/sim/models/$1/driver.py" ] || die "no sim/models/$1/driver.py" ;; esac
  drivers="$drivers $1"; shift
done
[ -n "$drivers" ] && groups+=("$current $drivers")
HASH="$(harness_hash)"
echo "harness_hash $HASH; groups: ${groups[*]}"
pids=()
for g in "${groups[@]}"; do
  read -r -a parts <<< "$g"
  ( run_vm "${parts[@]}" ) > "/tmp/cmdtest-run-${parts[0]}.log" 2>&1 &
  pids+=("$!")
  echo "${parts[0]}: ${parts[*]:1} -> /tmp/cmdtest-run-${parts[0]}.log"
done
failed=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then echo "OK: ${groups[$i]}"; else echo "FAILED: ${groups[$i]}"; failed=1; fi
done
[ "$failed" = 0 ] && echo "results in sim/cmdtest/results/ (harness_hash $HASH)"
exit "$failed"

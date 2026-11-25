# sim/rooms: phone clip of one room -> Depth Anything 3 room in the building scan's frame

## Run a new clip (from the project root)

```sh
sim/rooms/process.sh ~/Downloads/IMG_4479.MOV            # area found by alignment
sim/rooms/process.sh ~/Downloads/IMG_4479.MOV "office"   # or restrict to a named area / area_NN
VM=elegoo-sim-4 ZONE=us-east4-c sim/rooms/process.sh ...  # another elegoo-sim-* GPU machine
KEEP_VM=1 sim/rooms/process.sh ...                       # leave the machine running for the next clip
```

It starts the GPU machine if needed, uploads the scripts, the scan files and the clip, installs DA3 once
(`vm/setup_vm.sh`), runs `vm/run_clip.sh` (extract -> DA3 -> align -> fuse -> export -> assess -> map merge),
copies the results to `sim/rooms/<clip>/` and stops the machine. IMG_4474 (the whole-building walk) is refused.
Area names are the provisional ones in `sim/navigate/places.yaml`.

## Per clip (`sim/rooms/<clip>/`)

| File | What |
|---|---|
| `frames.json` | decoded frames, the selected ones (sharpest per window), HDR flag, blurry windows |
| `stats.json`, `render.png`, `cloud.ply`, `cam2world.npy` | raw DA3 result in its own frame (render like recon/out/da3_b) |
| `transform.json` | DA3 -> scan 4x4, scale, yaw, residuals, floor coverage, area, all candidates |
| `align_topdown.png` | room on the scan, both coloured and as outline |
| `cloud_aligned.ply`, `cam2world_aligned.npy` | points and cameras in the scan frame |
| `room_aligned.glb` | fused coloured mesh (TSDF 2 cm) in the scan frame, +Z up like `scans/merged/building.glb` |
| `room_aligned.usd` | same mesh for Isaac Sim: `/Room/mesh`, displayColor, metersPerUnit 1, upAxis Z |
| `sim_ready.json` | gate: median point-to-scan residual under 10 cm |
| `assess/` | car-eye pairs (scan left, DA3 right), occupancy and slice pictures, `assess.json` |

`sim/rooms/floor_map_rooms.png/.yaml`: `scans/merged/floor_map.png` where every sim-ready room's car-height
occupancy comes from DA3 (cells DA3 did not see keep the scan's value). `floor_map_rooms_add.png/.yaml`: the same,
except DA3 never removes a scan obstacle (it only adds obstacles and turns unknown cells it saw as floor into free);
use this one for driving, because DA3 misses most chair and table legs. Same size, origin and resolution as
floor_map.png, so either is a drop-in replacement for the map server and planners. `map_compare_<clip>.png` shows
old / replace / diff / add-only. DA3 obstacles count only with points from 3 or more frames, 0.8-3.5 m from the
clip's camera path (this drops the person filming and far noise), inside areas where DA3 saw 30 % or more of the floor.

## Adding a room to the Isaac scene (not done here; building.usda was not edited)

`room_aligned.usd` is already in the scan frame, so it needs no transform. Add, inside `def Xform "World"` of
`sim/isaac/scene/building.usda`, one prim per room (path relative to `sim/isaac/scene/`):

```usda
    def Xform "Room_IMG_4475" (
        prepend references = @../../rooms/IMG_4475/room_aligned.usd@
    )
    {
    }
```

Visual only: it has no collision and should not get any (the scan's `Collision` layer and the floor slab stay
the physics). The DA3 surfaces lie within a few cm of the scan's in that room, so both will show and fight where
they overlap. To see only the DA3 room there, either hide the scan for that area (needs cutting `scan_visual`
by the room polygon in `build_scene.py`) or put the room in a variant set like `Shell` so it can be switched on
and off per experiment.

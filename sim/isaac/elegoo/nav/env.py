"""Many ELEGOO cars in the scanned building, each with a spawn pose and a goal point.

All cars share one stage and one copy of the building. Every env origin is (0, 0, 0) and the
cars are placed in stage coordinates. The building is a global prim (collision_group=-1), so
every car collides with it, while Isaac Lab's inverted PhysX collision groups switch off
car-to-car contacts. RTX scene partitioning (on by default in Isaac Lab 3.0) tags each env
root, so each car camera renders the shared building and its own car only.

The generated shell is visual only. Where the scan has no wall (the ClosureWalls lines) this
env adds invisible 5 cm thick, 0.5 m tall static colliders, and privileged()["off_map"] flags
a car whose centre leaves the floor map's free cells (dilated 6 cm), for episode termination.
"""

from dataclasses import dataclass
from math import atan2, hypot, radians, tan
from pathlib import Path
import json

import isaaclab.sim as sim
import numpy as np
import torch
import yaml
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import MultiMeshRayCasterCfg, RayCasterCfg
from isaaclab.sensors.ray_caster.patterns import PatternBaseCfg
from isaaclab.utils import configclass

from ..environment import ElegooEnv, EnvSpec
from ..world import CameraSpec, scene_config

HERE = Path(__file__).resolve().parent
SCENE_DIR = HERE.parents[1] / "scene"
BUILDING_USD = SCENE_DIR / "building.usda"
BUILDING_VISUAL_USD = SCENE_DIR / "building_visual.usda"
META = yaml.safe_load((SCENE_DIR / "scene_meta.yaml").read_text())
CLOSURE = json.loads((HERE / "building_plan.json").read_text())["closure"]
COLLIDER = "/World/Building/Collision/collision_mesh"
SONAR_EMITTER = (0.0247, 0.0, 0.0225)
CLOSURE_HEIGHT_M = 0.5
CLOSURE_THICKNESS_M = 0.05
# The collider floor is z = 0. A grazing floor return does not echo back to the transducer.
FLOOR_ECHO_Z = 0.015
# Where the real car usually starts: beside the middle desk of the red office (desk centre (6.94, -0.54) in the
# IMG_4475 DA3 room), on the open aisle spawn pose with 0.5 m clearance. Under the desk the collider has legs that
# tip a car spawned at a random heading. Home starts sit on free-map cells within HOME_RADIUS_M, any heading.
OFFICE_DESK_XY = (6.23, -1.23)
HOME_RADIUS_M = 0.15


def cone_pattern(cfg: "ConePatternCfg", device: str) -> tuple[torch.Tensor, torch.Tensor]:
    angle = torch.arange(cfg.spokes, device=device) * (2 * torch.pi / cfg.spokes)
    ring = torch.arange(1, cfg.rings + 1, device=device) / cfg.rings
    plane = (ring[:, None, None] * torch.stack((angle.cos(), angle.sin()), -1)).reshape(-1, 2)
    plane = torch.cat((torch.zeros((1, 2), device=device), plane)) * tan(radians(cfg.full_cone_deg / 2))
    directions = torch.nn.functional.normalize(torch.cat((torch.ones_like(plane[:, :1]), plane), -1), dim=-1)
    return torch.zeros_like(directions), directions


@configclass
class ConePatternCfg(PatternBaseCfg):
    func: callable = cone_pattern
    full_cone_deg: float = 30.0
    rings: int = 4
    spokes: int = 32


@dataclass(frozen=True, slots=True)
class NavSpec(EnvSpec):
    obstacles: bool = False
    seed: int = 0
    min_goal_m: float = 2.0
    goal_radius_m: float = 0.25
    ambient_light: float = 4.0
    visual: bool = False             # textured visual scene + appearance randomisation for the RGB camera
    visual_period: int = 250         # control steps between building-wide look changes (visual only)
    home_probability: float = 0.0    # share of resets that start at the office middle desk instead of a spawn pose


def yaw_of(quat_xyzw: torch.Tensor) -> torch.Tensor:
    x, y, z, w = quat_xyzw.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def closure_colliders(cfg: InteractiveSceneCfg) -> None:
    for i, (x0, y0, x1, y1) in enumerate(CLOSURE):
        yaw = atan2(y1 - y0, x1 - x0)
        setattr(cfg, f"closure_{i}", AssetBaseCfg(
            prim_path=f"/World/ClosureColliders/segment_{i}",
            spawn=sim.CuboidCfg(
                size=(hypot(x1 - x0, y1 - y0) + CLOSURE_THICKNESS_M, CLOSURE_THICKNESS_M, CLOSURE_HEIGHT_M),
                collision_props=sim.CollisionPropertiesCfg(), visible=False,
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=((x0 + x1) / 2, (y0 + y1) / 2, CLOSURE_HEIGHT_M / 2),
                rot=(0.0, 0.0, float(np.sin(yaw / 2)), float(np.cos(yaw / 2))),
            ),
            collision_group=-1,
        ))


class BuildingNavEnv(ElegooEnv):
    """ElegooEnv in the building: spawns from scene_meta.yaml, a goal per car, cone sonar on the collider."""

    def __init__(self, sim_context: sim.SimulationContext, spec: NavSpec) -> None:
        device = sim_context.device
        self.spawns = torch.tensor([[p["x"], p["y"], p["yaw"]] for p in META["spawn_poses"]], device=device)
        self.goals = torch.tensor([[g["x"], g["y"]] for g in META["goals"]], device=device)
        self.goal_xy = torch.zeros((spec.count, 2), device=device)
        self.stall_seconds = torch.zeros(spec.count, device=device)
        self.generator = torch.Generator(device=device).manual_seed(spec.seed)
        free = np.load(HERE / "free_map.npz")
        shape = tuple(free["shape"])
        self.free = torch.tensor(np.unpackbits(free["allowed"])[: shape[0] * shape[1]].reshape(shape),
                                 dtype=torch.bool, device=device)
        self.free_origin = (float(free["x0"]), float(free["y0"]), float(free["res"]))
        x0, y0, res = self.free_origin
        rows, cols = torch.nonzero(self.free, as_tuple=True)
        cells = torch.stack((x0 + cols * res, y0 - rows * res), 1)
        self.home_cells = cells[(cells - torch.tensor(OFFICE_DESK_XY, device=device)).norm(dim=1) <= HOME_RADIUS_M]
        self.appearance = None
        super().__init__(sim_context, spec)
        self.sonar = self.scene["sonar"]
        if spec.visual and self.camera is not None:
            from ..appearance import Appearance

            self.appearance = Appearance(self, seed=spec.seed, period=spec.visual_period)
            self.appearance.reset(self.ids)

    def scene_config(self, camera: CameraSpec) -> InteractiveSceneCfg:
        spec = self.spec
        cfg = scene_config(spec.count, camera, obstacles=False)
        cfg.env_spacing = 0.0
        cfg.floor = None
        cfg.light = None
        cfg.building = AssetBaseCfg(
            prim_path="/World/Building",
            spawn=sim.UsdFileCfg(usd_path=str(BUILDING_VISUAL_USD if spec.visual else BUILDING_USD)),
            collision_group=-1,
        )
        closure_colliders(cfg)
        cfg.sonar = MultiMeshRayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/Geometry/base_footprint/sensor_pan",
            spawn=None,
            mesh_prim_paths=[
                COLLIDER,
                MultiMeshRayCasterCfg.RaycastTargetCfg(
                    prim_expr="/World/ClosureColliders", track_mesh_transforms=False,
                ),
            ],
            pattern_cfg=ConePatternCfg(),
            offset=RayCasterCfg.OffsetCfg(pos=SONAR_EMITTER),
            ray_alignment="base",
            max_distance=3.0,
        )
        if camera.outputs:
            cfg.camera.renderer_cfg.global_settings.ambient_light_intensity = spec.ambient_light
        return cfg

    def reset(self, ids: torch.Tensor | None = None) -> None:
        ids = self.ids if ids is None else ids
        super().reset(ids)
        if self.appearance is not None:
            self.appearance.reset(ids)
        n, count = len(ids), len(self.spawns)
        if n <= count:
            pick = torch.randperm(count, generator=self.generator, device=self.device)[:n]
        else:
            pick = torch.randint(count, (n,), generator=self.generator, device=self.device)
        start = self.spawns[pick].clone()
        if self.spec.home_probability > 0:  # no draw at 0, so older runs keep their random streams
            home = torch.rand(n, generator=self.generator, device=self.device) < self.spec.home_probability
            k = int(home.sum())
            if k:
                cell = torch.randint(len(self.home_cells), (k,), generator=self.generator, device=self.device)
                start[home, :2] = self.home_cells[cell]
                start[home, 2] = (torch.rand(k, generator=self.generator, device=self.device) * 2 - 1) * torch.pi
        self.place(ids, start[:, :2], start[:, 2])
        candidates = torch.randint(len(self.goals), (n, 16), generator=self.generator, device=self.device)
        distance = (self.goals[candidates] - start[:, None, :2]).norm(dim=-1)
        far_enough = distance >= self.spec.min_goal_m
        choice = torch.where(far_enough.any(1), far_enough.float().argmax(1), distance.argmax(1))
        self.goal_xy[ids] = self.goals[candidates[torch.arange(n, device=self.device), choice]]

    def place(self, ids: torch.Tensor, xy: torch.Tensor, yaw: torch.Tensor) -> None:
        """Put cars at stage-frame (x, y, yaw) just above the collider floor, at rest."""
        pose = torch.zeros((len(ids), 7), device=self.device)
        pose[:, :2] = xy
        pose[:, 2] = self.robot.data.default_root_pose.torch[ids, 2] + 0.005
        pose[:, 5], pose[:, 6] = (yaw / 2).sin(), (yaw / 2).cos()
        self.robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
        self.robot.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros((len(ids), 6), device=self.device), env_ids=ids,
        )
        self.drive.reset(ids)
        self.sensors.yaw_deg[ids] = 0
        self.stall_seconds[ids] = 0

    def advance(self, action: torch.Tensor | None, ticks: int) -> None:
        start = self.robot.data.root_link_pose_w.torch[:, :2].clone()
        super().advance(action, ticks)
        elapsed = ticks * self.dt
        travel = (self.robot.data.root_link_pose_w.torch[:, :2] - start).norm(dim=1)
        translating = self.drive.pwm.mean(dim=1).abs() >= 55
        self.stall_seconds = torch.where(
            translating & (travel < 0.025 * elapsed), self.stall_seconds + elapsed, 0.0,
        )

    def sonar_cm(self) -> torch.Tensor:
        data = self.sonar.data
        hits = data.ray_hits_w.torch
        distance = (hits - data.pos_w.torch[:, None]).norm(dim=-1)
        distance = torch.where(hits[..., 2] > FLOOR_ECHO_Z, distance, torch.inf)
        nearest = distance.amin(dim=1).nan_to_num(posinf=10.0)
        cm = nearest * 100 + self.drive.p["ultrasonic_range_noise_cm"] * torch.randn_like(nearest)
        return cm.round().clamp_min(0).minimum(self.drive.p["ultrasonic_cap_cm"]).to(torch.int32)

    def observe(self) -> dict[str, torch.Tensor]:
        obs = super().observe()
        obs["range_cm"] = self.sonar_cm()
        if self.appearance is not None and "rgb" in obs:
            obs["rgb"] = self.appearance.augment(obs["rgb"])
            self.appearance.tick()
        return obs

    def privileged(self) -> dict[str, torch.Tensor]:
        """Ground truth for planners, rewards, terminations and tests; not part of the car's observations."""
        root = self.robot.data.root_link_pose_w.torch
        x, y, z, w = root[:, 3:].unbind(-1)
        pitch = (2 * (w * y - z * x)).clamp(-1, 1).asin()
        roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        heading = yaw_of(root[:, 3:])
        to_goal = self.goal_xy - root[:, :2]
        bearing = torch.atan2(to_goal[:, 1], to_goal[:, 0]) - heading
        distance = to_goal.norm(dim=1)
        x0, y0, res = self.free_origin
        col = ((root[:, 0] - x0) / res).round().long().clamp(0, self.free.shape[1] - 1)
        row = ((y0 - root[:, 1]) / res).round().long().clamp(0, self.free.shape[0] - 1)
        return {
            "pos_w": root[:, :3], "heading_rad": heading, "goal_xy": self.goal_xy,
            "goal_dist_m": distance, "goal_bearing_rad": torch.atan2(bearing.sin(), bearing.cos()),
            "reached": distance < self.spec.goal_radius_m, "off_map": ~self.free[row, col],
            "tilted": pitch.abs().maximum(roll.abs()) > radians(6),
            "stalled": self.stall_seconds >= 0.8,
        }

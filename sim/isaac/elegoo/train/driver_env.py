"""Route-following episodes for the depth + heading-command driver, on BuildingNavEnv.

The policy sees only policy_obs(): corrupted encoded depth and aux (range, gyro rate, last sent PWM,
heading-error command). True pose and the collider are used only to generate the command (the
navigator's job), for reward, terminations and metrics.

Depth comes from a Warp ray-cast pinhole camera on the visual scan and the generated shell (the geometry
the RTX camera rendered before), so no RT cores are needed and every car has its own lens: pitch, height,
forward offset and field of view are drawn per episode within calibration.yaml's ranges.
Every episode also draws its control step (70 or 100 ms). All cars share the 10 ms physics clock; a car
whose step is shorter than the longest one is frozen (pose, drive state, delay queue, gyro) for the rest.
"""

from dataclasses import dataclass
from math import cos, pi, radians, sin, tan
from pathlib import Path

import numpy as np
import torch
import warp as wp
import yaml
from isaaclab.sensors.ray_caster import MultiMeshRayCasterCameraCfg, MultiMeshRayCasterCfg
from isaaclab.sensors.ray_caster.patterns import PatternBaseCfg, PinholeCameraPatternCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply

from ..drive import CALIBRATION
from ..nav.env import COLLIDER, SONAR_EMITTER, BuildingNavEnv, NavSpec
from ..world import PAN_PIVOT, CameraSpec
from .depth import NAMES, DepthCorruptor

COLLISION_TILT, COLLISION_STALL, COLLISION_OFF_MAP = 1, 2, 4
WIDTH, HEIGHT = 160, 120
CAMERA_KEYS = ("camera_pitch_deg", "camera_hfov_deg", "camera_height_m", "camera_forward_m")
VISUAL = ("/World/Building/Scan", "/World/Building/Shell")
BUMPER_HEIGHTS = (0.025, 0.07, 0.12)
BUMPER_RAYS = 72
BUMPER_RANGE_M = 0.6
CONTACT_M = 0.01
SONAR_X = PAN_PIVOT[0] + SONAR_EMITTER[0]
GROUND_ROWS = 80          # depth rows 80..119 look within ~13 deg of the horizon at the nominal 20 deg pitch


def ring_pattern(cfg: "RingPatternCfg", device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Horizontal rays in every direction from the car origin at a few heights (world-aligned)."""
    angle = torch.arange(cfg.rays, device=device) * (2 * torch.pi / cfg.rays)
    direction = torch.stack((angle.cos(), angle.sin(), torch.zeros_like(angle)), -1)
    starts = torch.zeros((len(cfg.heights), cfg.rays, 3), device=device)
    starts[..., 2] = torch.tensor(cfg.heights, device=device)[:, None]
    return starts.reshape(-1, 3), direction.repeat(len(cfg.heights), 1)


@configclass
class RingPatternCfg(PatternBaseCfg):
    func: callable = ring_pattern
    heights: tuple = BUMPER_HEIGHTS
    rays: int = BUMPER_RAYS


def lens_offset(height_m, forward_m):
    """Camera position relative to the pan link (world.py's convention)."""
    return forward_m - PAN_PIVOT[0], 0.0, height_m - PAN_PIVOT[2]


@dataclass(frozen=True, slots=True)
class RunSpec(NavSpec):
    split: str = "train"
    depth_corruptions: frozenset[str] = NAMES
    goal_radius_m: float = 0.30
    noisy_command: bool = True
    collision_penalty: float = 15.0
    step_ticks: tuple[int, ...] = (7,)     # control step per episode, in 10 ms physics ticks
    random_camera: bool = False            # per-episode lens within calibration.yaml's ranges (else nominal)
    margin_m: float = 0.10                 # proximity cost below this clearance
    proximity_weight: float = 0.0
    contact_weight: float = 0.0
    facing_fraction: float = 0.0           # episodes that start 10-20 cm in front of an obstacle (train only)
    exclude: tuple[tuple[float, float, float, float], ...] = ()   # (x0, y0, x1, y1): routes touching these are not drawn
    home_fraction: float = 0.0             # train: episodes on home_train routes (start beside the office middle desk)
    short_fraction: float = 0.0            # train: episodes on short_train routes (straight 0.3-1.5 m moves)


class DriverEnv(BuildingNavEnv):
    def __init__(self, sim, spec: RunSpec):
        self.ready = False
        super().__init__(sim, spec)
        self.depth_camera = self.scene["depth_camera"]
        self.bumper = self.scene["bumper"]
        mixed = spec.split == "train" and (spec.home_fraction > 0 or spec.short_fraction > 0)
        splits = (spec.split, "home_train", "short_train") if mixed else (spec.split,)
        with np.load(Path(__file__).with_name("routes.npz")) as data:
            width = max(data[f"{s}_xy"].shape[1] for s in splits)
            way = [np.pad(data[f"{s}_xy"], ((0, 0), (0, width - data[f"{s}_xy"].shape[1]), (0, 0)), mode="edge")
                   for s in splits]
            self.way = torch.as_tensor(np.concatenate(way), device=self.device)
            self.counts = torch.as_tensor(np.concatenate([data[f"{s}_count"] for s in splits]), device=self.device)
            self.yaw = torch.as_tensor(np.concatenate([data[f"{s}_yaw"] for s in splits]), device=self.device)
            self.length = torch.as_tensor(np.concatenate([data[f"{s}_length"] for s in splits]), device=self.device)
            self.areas = np.concatenate([data[f"{s}_area"] for s in splits])
            self.pool = torch.as_tensor(np.concatenate([np.full(len(data[f"{s}_count"]), k) for k, s in enumerate(splits)]),
                                        device=self.device)
        step = (self.way[:, 1:] - self.way[:, :-1]).norm(dim=-1)
        along = torch.cat((torch.zeros_like(step[:, :1]), step.cumsum(1)), dim=1)
        self.remaining = along[:, -1:] - along
        self.usable = torch.ones(len(self.length), dtype=torch.bool, device=self.device)
        for x0, y0, x1, y1 in spec.exclude:
            x, y = self.way.unbind(-1)
            self.usable &= ~((x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)).any(1)
        self.max_len = 3.0
        count = spec.count
        zeros = torch.zeros(count, device=self.device)
        self.route = torch.zeros_like(self.ids)
        self.index = self.route.clone()
        self.progress_index = self.route.clone()
        self.elapsed = self.route.clone()
        self.refresh = self.route.clone()
        self.age = self.route.clone()
        self.ticks = torch.full_like(self.ids, spec.step_ticks[0])
        self.control_dt = self.ticks * self.dt
        self.xy = torch.zeros((count, 2), device=self.device)
        self.last = torch.zeros_like(self.xy)
        self.target_yaw, self.waypoint_distance = zeros.clone(), zeros.clone()
        self.bias, self.bias_rate, self.limit = zeros.clone(), zeros.clone(), zeros.clone()
        self.potential, self.travel, self.peak = zeros.clone(), zeros.clone(), zeros.clone()
        self.contacts, self.near_steps, self.min_clearance = zeros.clone(), zeros.clone(), zeros.clone()
        self.success = torch.zeros_like(self.ids, dtype=torch.bool)
        self.collision, self.timeout = self.success.clone(), self.success.clone()
        self.touching, self.facing = self.success.clone(), self.success.clone()
        self.restarted = self.ids[:0]
        self.collision_kind = self.route.clone()
        self.depth_sim = DepthCorruptor(count, self.device, spec.depth_corruptions, self.control_dt)
        yy, xx = torch.meshgrid(torch.arange(HEIGHT, device=self.device) + 0.5,
                                torch.arange(WIDTH, device=self.device) + 0.5, indexing="ij")
        self.image_plane = torch.stack((xx, yy, torch.ones_like(xx)))
        parameters = yaml.safe_load(CALIBRATION.read_text())["parameters"]
        self.camera_range = torch.tensor([parameters[k] for k in CAMERA_KEYS], device=self.device)
        self.view = self.camera_range[:, 0].repeat(count, 1)
        self.footprint = self.measure_footprint()
        self.ray_angle = torch.arange(BUMPER_RAYS, device=self.device) * (2 * pi / BUMPER_RAYS)
        self.nominal_speed = self.drive.p["speed_scale"].clone()
        self.nominal_turn = self.drive.p["turn_scale"].clone()
        self.ready = True

    def scene_config(self, camera: CameraSpec):
        cfg = super().scene_config(camera)
        nominal = yaml.safe_load(CALIBRATION.read_text())["parameters"]
        pitch, hfov, height, forward = (nominal[k][0] for k in CAMERA_KEYS)
        cfg.depth_camera = MultiMeshRayCasterCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/Geometry/base_footprint/sensor_pan",
            mesh_prim_paths=[MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr=p, track_mesh_transforms=False)
                             for p in VISUAL],
            offset=MultiMeshRayCasterCameraCfg.OffsetCfg(
                pos=lens_offset(height, forward), rot=(0.0, -sin(radians(pitch) / 2), 0.0, cos(radians(pitch) / 2)),
                convention="world"),
            pattern_cfg=PinholeCameraPatternCfg(focal_length=18.0, horizontal_aperture=36.0 * tan(radians(hfov) / 2),
                                                width=WIDTH, height=HEIGHT),
            data_types=["distance_to_image_plane"],
            max_distance=20.0,
            depth_clipping_behavior="none",
        )
        cfg.bumper = MultiMeshRayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/Geometry/base_footprint",
            mesh_prim_paths=[
                COLLIDER,
                MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="/World/ClosureColliders", track_mesh_transforms=False),
            ],
            pattern_cfg=RingPatternCfg(),
            ray_alignment="world",
            max_distance=BUMPER_RANGE_M,
        )
        return cfg

    def measure_footprint(self) -> tuple[float, float, float, float]:
        """Car outline (x min, x max, y min, y max) around its origin, from the spawned asset's bounds."""
        import omni.usd
        from pxr import Usd, UsdGeom

        prim = omni.usd.get_context().get_stage().GetPrimAtPath("/World/envs/env_0/Robot")
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
        box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        x, y = self.robot.data.default_root_pose.torch[0, :2].tolist()
        lo, hi = box.GetMin(), box.GetMax()
        footprint = (lo[0] - x, hi[0] - x, lo[1] - y, hi[1] - y)
        print("DRIVER_FOOTPRINT", [round(v, 4) for v in footprint], flush=True)
        if not (0.15 < footprint[1] - footprint[0] < 0.4 and 0.1 < footprint[3] - footprint[2] < 0.3):
            raise RuntimeError(f"Implausible car footprint {footprint}")
        return footprint

    def body_reach(self, phi: torch.Tensor) -> torch.Tensor:
        """Distance from the car origin to its outline along the body-frame direction phi."""
        xmin, xmax, ymin, ymax = self.footprint
        c, s = phi.cos(), phi.sin()
        tx = torch.where(c > 0, xmax, -xmin) / c.abs().clamp_min(1e-6)
        ty = torch.where(s > 0, ymax, -ymin) / s.abs().clamp_min(1e-6)
        return torch.minimum(tx, ty)

    def on_free(self, xy: torch.Tensor) -> torch.Tensor:
        x0, y0, res = self.free_origin
        col = ((xy[..., 0] - x0) / res).round().long().clamp(0, self.free.shape[1] - 1)
        row = ((y0 - xy[..., 1]) / res).round().long().clamp(0, self.free.shape[0] - 1)
        return self.free[row, col]

    def set_conditions(self, response: torch.Tensor, stress: torch.Tensor) -> None:
        """Per car: scale speed and turn response, and put depth corruptions at their upper magnitudes."""
        self.drive.p["speed_scale"].copy_(self.nominal_speed * response)
        self.drive.p["turn_scale"].copy_(self.nominal_turn * response)
        self.depth_sim.stress.copy_(stress)

    def draw_routes(self, n: int) -> torch.Tensor:
        """Base pool within max_len; with home/short mixing, each car picks its pool first (0 base, 1 home, 2 short)."""
        ok = (self.length <= self.max_len) & self.usable
        kind = torch.zeros(n, dtype=torch.long, device=self.device)
        if self.pool.max() > 0:
            u = torch.rand(n, device=self.device)
            kind[u < self.spec.home_fraction] = 1
            kind[(u >= self.spec.home_fraction) & (u < self.spec.home_fraction + self.spec.short_fraction)] = 2
        routes = torch.empty(n, dtype=torch.long, device=self.device)
        for k in kind.unique().tolist():
            pick = kind == k
            pool = (ok & (self.pool == k)).nonzero().squeeze(1)
            if len(pool) == 0:
                pool = (self.pool == k).nonzero().squeeze(1)
            routes[pick] = pool[torch.randint(len(pool), (int(pick.sum()),), device=self.device)]
        return routes

    def reset(self, ids=None, routes=None) -> None:
        if not self.ready:
            super().reset(ids)
            return
        ids = self.ids if ids is None else ids
        if routes is None:
            routes = self.draw_routes(len(ids))
        super().reset(ids)
        self.route[ids] = routes
        for buffer in (self.index, self.progress_index, self.elapsed, self.refresh, self.age, self.collision_kind):
            buffer[ids] = 0
        self.xy[ids] = self.way[routes, 0]
        self.place(ids, self.xy[ids], self.yaw[routes])
        self.goal_xy[ids] = self.way[routes, self.counts[routes] - 1]
        self.potential[ids] = self.remaining[routes, 0]
        for buffer in (self.last, self.travel, self.peak, self.contacts, self.near_steps):
            buffer[ids] = 0
        self.min_clearance[ids] = BUMPER_RANGE_M
        for flag in (self.success, self.collision, self.timeout, self.touching):
            flag[ids] = False
        self.depth_sim.reset(ids)
        u = torch.rand((len(ids), 9), device=self.device)
        noisy = float(self.spec.noisy_command)
        self.bias[ids] = (u[:, 0] - 0.5) * 0.10 * noisy
        self.bias_rate[ids] = (u[:, 1] - 0.5) * 0.003 * noisy
        train = self.spec.split == "train"
        self.limit[ids] = (0.5 + 0.5 * u[:, 2]) if train else 1.0
        choices = torch.tensor(self.spec.step_ticks, device=self.device)
        self.ticks[ids] = choices[(u[:, 3] * len(choices)).long().clamp_max(len(choices) - 1)]
        self.control_dt[ids] = self.ticks[ids] * self.dt
        self.facing[ids] = (u[:, 4] < self.spec.facing_fraction) & train
        self.aim(ids, u[:, 5:9])

    def aim(self, ids: torch.Tensor, u: torch.Tensor) -> None:
        """Give each car its own lens: pitch, HFOV, height and forward offset (nominal unless random_camera)."""
        nominal, low, high = self.camera_range.unbind(1)
        view = torch.lerp(low.expand_as(u), high.expand_as(u), u) if self.spec.random_camera \
            else nominal.expand_as(u)
        self.view[ids] = view
        pitch, hfov, height, forward = view.unbind(1)
        focal = (WIDTH / 2) / torch.tan(torch.deg2rad(hfov) / 2)
        k = torch.zeros((len(ids), 3, 3), device=self.device)
        k[:, 0, 0], k[:, 1, 1], k[:, 0, 2], k[:, 1, 2], k[:, 2, 2] = focal, focal, WIDTH / 2, HEIGHT / 2, 1
        camera = self.depth_camera
        camera.set_intrinsic_matrices(k, env_ids=ids)
        half = torch.deg2rad(pitch) / 2
        zero = torch.zeros_like(half)
        # Isaac Lab 3.0 keeps each env's lens offset (world convention: x forward, z up) in these Warp buffers.
        wp.to_torch(camera._offset_pos_wp)[ids] = torch.stack(
            (forward - PAN_PIVOT[0], zero, height - PAN_PIVOT[2]), 1)
        wp.to_torch(camera._offset_quat_wp)[ids] = torch.stack((zero, -half.sin(), zero, half.cos()), 1)

    def advance(self, action: torch.Tensor | None, ticks: int | None = None) -> None:
        """Physics for one control step of every car (self.ticks each, or `ticks` for all); a car whose
        step has ended is held still, with its drive, packet queue and gyro paused, until the longest ends."""
        steps = self.ticks if ticks is None else torch.full_like(self.ticks, ticks)
        longest, shortest = int(steps.max()), int(steps.min())
        robot, drive = self.robot, self.drive
        start = robot.data.root_link_pose_w.torch[:, :2].clone()
        for i in range(longest):
            frozen = steps <= i if i >= shortest else None
            held = self.hold() if frozen is not None else None
            speed, omega = drive.tick(action)
            data = robot.data
            forward_w = quat_apply(data.root_link_pose_w.torch[:, 3:], self.forward)
            velocity = data.root_com_vel_w.torch.clone()
            velocity[:, :2] = forward_w[:, :2] * speed[:, None]
            velocity[:, 5] = omega
            # Wheel motion is visual; the calibrated root velocity is the drive input.
            wheel_rate = drive.pwm[:, self.wheel_sides] * (drive.p["speed_m_s_per_pwm"] / self.wheel_radius_m)[:, None]
            if frozen is not None:
                velocity = torch.where(frozen[:, None], 0.0, velocity)
                wheel_rate = torch.where(frozen[:, None], 0.0, wheel_rate)
            robot.write_root_velocity_to_sim_index(root_velocity=velocity, skip_forward=True)
            robot.set_joint_position_target_index(target=self.pan_target, joint_ids=self.pan_ids)
            robot.write_joint_velocity_to_sim_index(velocity=wheel_rate, joint_ids=self.wheel_ids)
            self.scene.write_data_to_sim()
            self.sim.step(render=self.camera is not None and i == longest - 1)
            self.scene.update(self.dt)
            self.sensors.integrate(data.root_link_pose_w.torch[:, 3:], data.root_com_vel_w.torch[:, 3:], self.dt)
            if frozen is not None:
                self.release(frozen, held)
        elapsed = steps * self.dt
        travel = (robot.data.root_link_pose_w.torch[:, :2] - start).norm(dim=1)
        translating = drive.pwm.mean(dim=1).abs() >= 55
        self.stall_seconds = torch.where(
            translating & (travel < 0.025 * elapsed), self.stall_seconds + elapsed, 0.0,
        )

    def hold(self) -> dict[str, torch.Tensor]:
        d = self.drive
        held = {k: getattr(d, k).clone() for k in
                ("queue", "queued", "pwm", "last_receive", "next_send", "speed", "omega", "coast_tau")}
        held["yaw"], held["rate"] = self.sensors.yaw_deg.clone(), self.sensors.rate_deg_s.clone()
        held["pose"] = self.robot.data.root_link_pose_w.torch.clone()
        return held

    def release(self, frozen: torch.Tensor, held: dict[str, torch.Tensor]) -> None:
        """Undo one tick for frozen cars. Their queued packets move one slot later (the ring index advanced)
        and their packet timestamps one tick later (the drive clock advanced)."""
        d, m = self.drive, frozen
        d.queue = torch.where(m[None, :, None], held["queue"].roll(1, 0), d.queue)
        d.queued = torch.where(m[None], held["queued"].roll(1, 0), d.queued)
        d.pwm = torch.where(m[:, None], held["pwm"], d.pwm)
        for k in ("speed", "omega", "coast_tau"):
            setattr(d, k, torch.where(m, held[k], getattr(d, k)))
        for k in ("last_receive", "next_send"):
            setattr(d, k, torch.where(m, held[k] + self.dt, getattr(d, k)))
        self.sensors.yaw_deg = torch.where(m, held["yaw"], self.sensors.yaw_deg)
        self.sensors.rate_deg_s = torch.where(m, held["rate"], self.sensors.rate_deg_s)
        ids = m.nonzero().squeeze(1)
        self.robot.write_root_pose_to_sim_index(root_pose=held["pose"][ids], env_ids=ids)
        self.robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros((len(ids), 6), device=self.device),
                                                    env_ids=ids)

    def command(self, xy: torch.Tensor, true_yaw: torch.Tensor, gyro_yaw: torch.Tensor) -> torch.Tensor:
        """Navigator: re-plans at 1-2 Hz from the true position; between updates the error follows the gyro.

        Target = farthest lookahead waypoint (0.44-2.4 m) still in line of sight of the path within
        10 cm, so it sometimes cuts corners through furniture. Noise: per-episode heading bias drifting
        up to 5 deg/min, 3.5 % of updates off by 20-40 deg, and in training 6 % deliberate shortcuts.
        """
        route, final = self.route, self.counts[self.route] - 1
        candidates = torch.minimum(self.index[:, None] + torch.arange(10, device=self.device), final[:, None])
        points = self.way[route[:, None], candidates]
        nearest = candidates.gather(1, (points - xy[:, None]).square().sum(-1).argmin(1)[:, None])[:, 0]
        hold = self.refresh > 0
        self.index = torch.where(hold, self.index, nearest)
        u = torch.rand((len(xy), 4), device=self.device)
        lookahead = torch.arange(1, 13, device=self.device)
        path_idx = torch.minimum(self.index[:, None] + lookahead, final[:, None])
        relative = self.way[route[:, None], path_idx] - xy[:, None]
        unit = torch.nn.functional.normalize(relative[:, 2], dim=1)
        cross_track = (relative[..., 0] * unit[:, 1, None] - relative[..., 1] * unit[:, 0, None]).abs()
        choice = ((cross_track < 0.10).cumprod(1).sum(1) - 1).clamp(2, 11)
        shortcut = (u[:, 2] < 0.06) & (self.spec.split == "train") & self.spec.noisy_command
        choice = torch.where(shortcut, 7, choice)
        delta = relative[self.ids, choice]
        distance = delta.norm(dim=1)
        bad = (u[:, 0] < 0.035) & (self.age > 0) & self.spec.noisy_command
        outlier = torch.where(bad, (u[:, 1] - 0.5).sign() * (0.349 + 0.349 * u[:, 3]), 0.0)
        estimate = torch.atan2(delta[:, 1], delta[:, 0]) - true_yaw + gyro_yaw + self.bias + outlier
        self.target_yaw = torch.where(hold, self.target_yaw, estimate)
        self.waypoint_distance = torch.where(hold, self.waypoint_distance, distance)
        ticks = ((0.50 + 0.50 * u[:, 1]) / self.control_dt).round().long()
        self.refresh = torch.where(hold, self.refresh, ticks) - 1
        self.bias = (self.bias + self.bias_rate * self.control_dt).clamp(-0.22, 0.22)
        level = (self.waypoint_distance >= 0.5).float() * 0.5 + (self.waypoint_distance > 2.0).float() * 0.5
        err = self.target_yaw - gyro_yaw
        return torch.stack((err.sin(), err.cos(), level, self.limit), dim=1)

    def true_depth(self) -> torch.Tensor:
        """Uncorrupted optical-axis depth of each car's own camera [N,120,160], metres, inf = no return."""
        z = self.depth_camera.data.output["distance_to_image_plane"].torch[..., 0]
        return z.nan_to_num(nan=torch.inf)

    def policy_obs(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns encoded corrupted depth [N,1,120,160] and aux [N,8] (layout in model.py)."""
        raw = super().observe()
        p = self.privileged()
        z = self.true_depth()
        data = self.depth_camera.data
        q = data.quat_w_world.torch
        eye = torch.eye(3, device=self.device).expand(len(z), 3, 3)
        basis_z = quat_apply(q[:, None].expand(-1, 3, -1), eye)[..., 2]
        k = data.intrinsic_matrices.torch
        rx = (self.image_plane[0] - k[:, 0, 2, None, None]) / k[:, 0, 0, None, None]
        ry = (self.image_plane[1] - k[:, 1, 2, None, None]) / k[:, 1, 1, None, None]
        # World-convention camera axes: x forward, y left, z up; image x right = -y, image y down = -z.
        cos_z = basis_z[:, 0, None, None] - basis_z[:, 1, None, None] * rx - basis_z[:, 2, None, None] * ry
        world_height = data.pos_w.torch[:, 2, None, None] + z * cos_z
        depth = self.depth_sim(z, world_height)
        # After a reset no frame has been taken at the new pose yet: show an all-far frame.
        depth = torch.where(self.age[:, None, None] == 0, 0, depth)
        command = self.command(p["pos_w"][:, :2], p["heading_rad"], torch.deg2rad(raw["yaw_deg"]))
        aux = torch.cat((raw["range_cm"][:, None] / 150, raw["gyro_rate_deg_s"][:, None] / 250,
                         self.last, command), dim=1)
        self.age.add_(1)
        return depth[:, None], aux

    def clearance(self, heading: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Gap between the car outline and the nearest obstacle (m): (bumper, nearest of all three).

        bumper: a horizontal 72-ray ring at 2.5 / 7 / 12 cm on the collider and closure walls (privileged);
        sonar: the cone reading from the front emitter; ahead: the lowest 40 rows of the true depth image."""
        data = self.bumper.data
        hits = data.ray_hits_w.torch
        reach = (hits[..., :2] - data.pos_w.torch[:, None, :2]).norm(dim=-1).nan_to_num(nan=torch.inf)
        phi = self.ray_angle.repeat(len(BUMPER_HEIGHTS))[None] - heading[:, None]
        bumper = (reach - self.body_reach(phi)).amin(1)
        front = self.footprint[1]
        sonar = self.sonar_cm().float() / 100 - (front - SONAR_X)
        ahead = self.true_depth()[:, GROUND_ROWS:].flatten(1).amin(1) - (front - self.view[:, 3])
        return bumper, torch.minimum(bumper, torch.minimum(sonar, ahead))

    def episode_step(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Send signed PWM/255, advance one control step; returns reward and done."""
        pwm = (action.clamp(-1, 1) * 255).round() / 255
        self.advance(pwm)
        p = self.privileged()
        xy = p["pos_w"][:, :2]
        distance = (xy - self.xy).norm(dim=1)
        speed = distance / self.control_dt
        self.travel.add_(distance)
        self.peak = self.peak.maximum(speed)
        self.elapsed.add_(1)
        bumper, nearest = self.clearance(p["heading_rad"])
        contact = bumper <= CONTACT_M
        self.contacts.add_((contact & ~self.touching).float())
        self.touching = contact
        self.near_steps.add_((nearest < self.spec.margin_m).float())
        self.min_clearance = self.min_clearance.minimum(bumper)
        kind = (p["tilted"].long() * COLLISION_TILT + p["stalled"].long() * COLLISION_STALL
                + p["off_map"].long() * COLLISION_OFF_MAP)
        self.collision = kind > 0
        self.collision_kind = kind
        self.success = p["reached"] & ~self.collision
        self.timeout = (self.elapsed * self.control_dt >= 8 + self.length[self.route] * 5.0) \
            & ~self.success & ~self.collision
        final = self.counts[self.route] - 1
        window = torch.minimum(self.progress_index[:, None] + torch.arange(8, device=self.device), final[:, None])
        near = (self.way[self.route[:, None], window] - xy[:, None]).square().sum(-1).argmin(1)
        self.progress_index = window.gather(1, near[:, None])[:, 0]
        ahead = torch.minimum(self.progress_index + 1, final)
        potential = self.remaining[self.route, ahead] + (self.way[self.route, ahead] - xy).norm(dim=1)
        # Per-time costs are scaled to the 70 ms step so 70 and 100 ms episodes pay the same per second.
        scale = self.control_dt / 0.07
        proximity = ((self.spec.margin_m - nearest) / self.spec.margin_m).clamp(0, 1)
        reward = 10.0 * (self.potential - potential) - 0.02 * scale
        reward -= 0.01 * ((pwm - self.last).abs().sum(dim=1) / (30 / 255))
        reward -= scale * (self.spec.proximity_weight * proximity * (0.25 + speed / 0.4)
                           + self.spec.contact_weight * contact.float() * (1.0 + speed / 0.2))
        reward += 10.0 * self.success.float() - self.spec.collision_penalty * self.collision.float()
        self.potential = potential
        self.xy.copy_(xy)
        self.last.copy_(pwm)
        done = self.success | self.collision | self.timeout
        self.restarted = self.ids[:0]
        starting = self.facing & (self.elapsed == 1) & ~done
        if starting.any():
            self.restarted = self.face(starting.nonzero().squeeze(1))
            reward[self.restarted] = 0
        return reward, done

    def face(self, ids: torch.Tensor) -> torch.Tensor:
        """Move freshly started cars to face an obstacle with 10-20 cm between it and the front of the car.

        Uses the bumper hits (2.5 and 7 cm rings) seen from the start pose: for each of the 72 directions,
        shift the car along it so the obstacle straight ahead is 10-20 cm off, keep it if every hit point
        stays >= 5 cm outside the outline and the centre is on the floor map, and pick one at random.
        Returns the cars that were moved (they restart there; the rest keep their normal start)."""
        self.facing[ids] = False
        data = self.bumper.data
        hits = data.ray_hits_w.torch[ids, : 2 * BUMPER_RAYS, :2]
        centre = data.pos_w.torch[ids, :2]
        unit = torch.stack((self.ray_angle.cos(), self.ray_angle.sin()), -1)
        rel = torch.where(torch.isfinite(hits).all(-1, keepdim=True), hits - centre[:, None],
                          unit.repeat(2, 1)[None] * 10.0)
        dist = rel.norm(dim=-1).view(len(ids), 2, BUMPER_RAYS).amin(1)
        xmin, xmax, ymin, ymax = self.footprint
        gap = 0.10 + 0.10 * torch.rand((len(ids), 1), device=self.device)
        shift = dist - xmax - gap
        q = rel[:, None] - (shift[..., None] * unit)[:, :, None]
        c, s = unit[None, :, None, 0], unit[None, :, None, 1]
        bx, by = q[..., 0] * c + q[..., 1] * s, -q[..., 0] * s + q[..., 1] * c
        ox, oy = torch.maximum(bx - xmax, xmin - bx), torch.maximum(by - ymax, ymin - by)
        outside = torch.where((ox < 0) & (oy < 0), torch.maximum(ox, oy),
                              torch.hypot(ox.clamp_min(0), oy.clamp_min(0))).amin(-1)
        moved = centre[:, None] + shift[..., None] * unit
        ok = (dist < BUMPER_RANGE_M) & (outside >= 0.05) & (shift.abs() <= 0.4) & self.on_free(moved)
        pick = torch.where(ok, torch.rand_like(dist), -1.0).argmax(1)
        chosen = ok.any(1)
        ids, pick = ids[chosen], pick[chosen]
        if not len(ids):
            return ids
        xy = moved[chosen.nonzero().squeeze(1), pick]
        self.place(ids, xy, self.ray_angle[pick])
        self.depth_sim.reset(ids)
        for buffer in (self.index, self.progress_index, self.elapsed, self.refresh, self.age):
            buffer[ids] = 0
        for buffer in (self.last, self.travel, self.peak, self.contacts, self.near_steps):
            buffer[ids] = 0
        self.touching[ids] = False
        self.min_clearance[ids] = BUMPER_RANGE_M
        self.xy[ids] = xy
        route = self.route[ids]
        self.potential[ids] = self.remaining[route, 0] + (self.way[route, 0] - xy).norm(dim=1)
        return ids

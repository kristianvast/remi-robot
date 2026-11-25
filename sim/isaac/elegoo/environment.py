"""Standalone vectorized Isaac Lab environment (GPU tensors in and out)."""

from dataclasses import dataclass, replace

import torch
import yaml
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils.math import quat_apply

from .drive import CALIBRATION, Drive
from .sensors import Sensors
from .world import CameraSpec, scene_config


@dataclass(frozen=True, slots=True)
class EnvSpec:
    count: int = 16
    camera: CameraSpec = CameraSpec()
    obstacles: bool = True
    randomize: bool = False
    decimation: int = 7


class ElegooEnv:
    """One call advances seven 10 ms PhysX ticks by default.

    Actions are [N,2] left/right normalized PWM. None means no new packet.
    Observations intentionally contain neither pose nor wheel joint state.
    """

    def __init__(self, sim: SimulationContext, spec: EnvSpec) -> None:
        self.sim, self.spec = sim, spec
        self.dt = sim.get_physics_dt()
        self.device = sim.device
        self.drive = Drive(spec.count, self.dt, self.device, randomize=spec.randomize)
        camera = spec.camera
        if spec.randomize and camera.outputs:
            # Tiled RTX shares one intrinsic projection across the camera batch.
            fields = ("camera_pitch_deg", "camera_hfov_deg", "camera_height_m", "camera_forward_m")
            pitch, hfov, height, forward = [self.drive.p[key][0].item() for key in fields]
            camera = replace(camera, pitch_deg=pitch, hfov_deg=hfov, height_m=height, forward_m=forward)
        self.scene: InteractiveScene = InteractiveScene(self.scene_config(camera))
        sim.reset()
        self.robot = self.scene["robot"]
        self.camera = self.scene["camera"] if spec.camera.outputs else None
        self.sensors = Sensors(self.drive.p, obstacles=spec.obstacles)
        self.ids = torch.arange(spec.count, device=self.device)
        self.pan_ids, _ = self.robot.find_joints("sensor_pan_joint")
        self.wheel_ids, wheel_names = self.robot.find_joints(".*_wheel_joint")
        self.wheel_sides = torch.tensor([0 if "_left_" in name else 1 for name in wheel_names],
                                        device=self.device)
        self.pan_target = torch.zeros((spec.count, 1), device=self.device)
        self.forward = torch.zeros((spec.count, 3), device=self.device)
        self.forward[:, 0] = 1
        self.wheel_radius_m = yaml.safe_load(CALIBRATION.read_text())["measured"]["wheel_diameter_m"] / 2
        self.reset()

    def scene_config(self, camera: CameraSpec) -> InteractiveSceneCfg:
        """Scene for this env; subclasses (nav.env.BuildingNavEnv) replace the flat course."""
        return scene_config(self.spec.count, camera, obstacles=self.spec.obstacles)

    def reset(self, ids: torch.Tensor | None = None) -> None:
        ids = self.ids if ids is None else ids
        self.scene.reset(ids)
        pose = self.robot.data.default_root_pose.torch[ids].clone()
        pose[:, :3] += self.scene.env_origins[ids]
        self.robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
        self.robot.write_root_velocity_to_sim_index(
            root_velocity=self.robot.data.default_root_vel.torch[ids], env_ids=ids,
        )
        self.robot.write_joint_position_to_sim_index(
            position=self.robot.data.default_joint_pos.torch[ids], env_ids=ids,
        )
        self.robot.write_joint_velocity_to_sim_index(
            velocity=self.robot.data.default_joint_vel.torch[ids], env_ids=ids,
        )
        self.drive.reset(ids)
        self.sensors.yaw_deg[ids] = 0
        self.sensors.rate_deg_s[ids] = 0
        self.pan_target[ids] = 0

    def set_pan(self, radians: torch.Tensor) -> None:
        self.pan_target.copy_(radians.reshape_as(self.pan_target).clamp(-torch.pi / 2, torch.pi / 2))

    def step(self, action: torch.Tensor | None) -> dict[str, torch.Tensor]:
        self.advance(action, self.spec.decimation)
        return self.observe()

    def advance(self, action: torch.Tensor | None, ticks: int) -> None:
        """Advance physics; capture only once at the end of the external step."""
        for i in range(ticks):
            speed, omega = self.drive.tick(action)
            data = self.robot.data
            forward_w = quat_apply(data.root_link_pose_w.torch[:, 3:], self.forward)
            velocity = data.root_com_vel_w.torch.clone()
            velocity[:, :2] = forward_w[:, :2] * speed[:, None]
            velocity[:, 5] = omega
            self.robot.write_root_velocity_to_sim_index(root_velocity=velocity, skip_forward=True)
            self.robot.set_joint_position_target_index(target=self.pan_target, joint_ids=self.pan_ids)
            # Wheel motion is visual; the calibrated root velocity is the drive input.
            wheel_rate = self.drive.pwm[:, self.wheel_sides] * (
                self.drive.p["speed_m_s_per_pwm"] / self.wheel_radius_m
            )[:, None]
            self.robot.write_joint_velocity_to_sim_index(velocity=wheel_rate, joint_ids=self.wheel_ids)
            self.scene.write_data_to_sim()
            self.sim.step(render=self.camera is not None and i == ticks - 1)
            self.scene.update(self.dt)
            self.sensors.integrate(data.root_link_pose_w.torch[:, 3:], data.root_com_vel_w.torch[:, 3:], self.dt)

    def observe(self) -> dict[str, torch.Tensor]:
        root = self.robot.data.root_link_pose_w.torch
        origin = root[:, :3] - self.scene.env_origins
        floor = self.sensors.floor_bits(origin, root[:, 3:])
        obs = {
            "range_cm": self.sensors.range_cm(origin, root[:, 3:], self.robot.data.joint_pos.torch[:, self.pan_ids[0]]),
            "yaw_deg": self.sensors.yaw_deg.clone(),
            "gyro_rate_deg_s": self.sensors.rate_deg_s.clone(),
            "floor_bits": floor,
            "on_ground": floor.all(dim=1),
            "last_action": self.drive.pwm / self.drive.limit,
        }
        if self.camera is not None:
            for name in self.spec.camera.outputs:
                image = self.camera.data.output[name].torch
                if name == "rgb" or name.startswith("simple_shading_"):
                    obs["rgb"] = image[..., :3]
                else:
                    obs[name] = image
        return obs

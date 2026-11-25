"""Scene geometry shared by PhysX, RTX, and the ultrasonic model."""

from dataclasses import dataclass
from math import cos, radians, sin, tan
from pathlib import Path

import isaaclab.sim as sim
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab_physx.renderers import IsaacRtxRendererCfg


@dataclass(frozen=True, slots=True)
class Box:
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    color: tuple[float, float, float]


BOXES = (
    Box((0.95, 0.10, 0.22), (0.24, 0.34, 0.44), (0.68, 0.16, 0.06)),
    Box((1.65, -0.55, 0.32), (0.35, 0.38, 0.64), (0.08, 0.40, 0.55)),
    Box((1.42, 0.67, 0.22), (0.40, 0.55, 0.44), (0.36, 0.52, 0.11)),
)
PAN_PIVOT = (0.0853, 0.0, 0.07905)
FLOOR_COLOR = (0.30, 0.26, 0.22)
ASSET = Path(__file__).with_name("asset") / "elegoo_v4.usdc"


@dataclass(frozen=True, slots=True)
class CameraSpec:
    outputs: tuple[str, ...] = ("rgb", "depth")
    width: int = 640
    height: int = 480
    pitch_deg: float = 20
    hfov_deg: float = 65
    height_m: float = 0.126
    forward_m: float = 0.0825
    far_clip_m: float = 6.0


def scene_config(count: int, camera: CameraSpec, *, obstacles: bool) -> InteractiveSceneCfg:
    cfg = InteractiveSceneCfg(num_envs=count, env_spacing=8.0, replicate_physics=True)
    no_friction = sim.RigidBodyMaterialCfg(static_friction=0.0, dynamic_friction=0.0,
                                           friction_combine_mode="multiply", restitution=0.0)
    cfg.floor = AssetBaseCfg(
        prim_path="/World/Floor",
        spawn=sim.CuboidCfg(
            size=(180.0, 180.0, 0.10), collision_props=sim.CollisionPropertiesCfg(),
            physics_material=no_friction,
            visual_material=sim.PreviewSurfaceCfg(diffuse_color=FLOOR_COLOR),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.05)),
    )
    cfg.light = AssetBaseCfg(prim_path="/World/Dome", spawn=sim.DomeLightCfg(intensity=2000.0))
    cfg.robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim.UsdFileCfg(
            usd_path=str(ASSET),
            rigid_props=sim.RigidBodyPropertiesCfg(linear_damping=0.0, angular_damping=0.0),
            articulation_props=sim.ArticulationRootPropertiesCfg(enabled_self_collisions=False),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        actuators={
            "pan": ImplicitActuatorCfg(
                joint_names_expr=["sensor_pan_joint"], stiffness=1.2, damping=0.07,
                effort_limit_sim=0.18, velocity_limit_sim=5.2,
            ),
            "wheels": ImplicitActuatorCfg(
                joint_names_expr=[".*_wheel_joint"], stiffness=0.0, damping=0.0,
            ),
        },
    )
    if obstacles:
        for i, box in enumerate(BOXES):
            setattr(cfg, f"box_{i}", AssetBaseCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Box_{i}",
                spawn=sim.CuboidCfg(
                    size=box.size, collision_props=sim.CollisionPropertiesCfg(),
                    visual_material=sim.PreviewSurfaceCfg(diffuse_color=box.color),
                ),
                init_state=AssetBaseCfg.InitialStateCfg(pos=box.center),
            ))
    if camera.outputs:
        pitch = radians(camera.pitch_deg) / 2
        cfg.camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/Geometry/base_footprint/sensor_pan/Lens",
            offset=CameraCfg.OffsetCfg(
                pos=(camera.forward_m - PAN_PIVOT[0], 0.0, camera.height_m - PAN_PIVOT[2]),
                rot=(0.0, -sin(pitch), 0.0, cos(pitch)), convention="world",
            ),
            spawn=sim.PinholeCameraCfg(
                focal_length=18.0, horizontal_aperture=36.0 * tan(radians(camera.hfov_deg) / 2),
                # Keep the shallow floor rays in view at the upward camera tilt.
                clipping_range=(0.02, camera.far_clip_m),
            ),
            width=camera.width, height=camera.height, data_types=list(camera.outputs),
            renderer_cfg=IsaacRtxRendererCfg(),
            background_color=(0.34, 0.43, 0.56),
        )
    return cfg

"""Replay a Booster K1 NPZ motion in Isaac Sim.

Usage:
    python scripts/replay_npz_k1.py --motion_file motions/k1/135_06_stageii.npz
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import pathlib

import numpy as np
import torch

from isaaclab.app import AppLauncher

DEFAULT_K1_ASSET_DIR = pathlib.Path("/home/liyunsong/RL_project/GMR/assets/booster_k1")
DEFAULT_K1_URDF = DEFAULT_K1_ASSET_DIR / "K1_22dof.urdf"

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay a Booster K1 NPZ motion.")
parser.add_argument("--motion_file", type=str, required=True, help="Local K1 motion NPZ file.")
parser.add_argument("--k1_urdf", type=str, default=str(DEFAULT_K1_URDF), help="Path to the K1 URDF.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


K1_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=str(pathlib.Path(args_cli.k1_urdf).expanduser()),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "all_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit_sim=100.0,
            velocity_limit_sim=100.0,
            stiffness=0.0,
            damping=0.0,
        ),
    },
)


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    robot: ArticulationCfg = K1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


class K1Motion:
    """Loads the NPZ written by csv_to_npz_k1.py."""

    def __init__(self, motion_file: str, device: str):
        data = np.load(motion_file)
        self.fps = float(np.asarray(data["fps"]).reshape(-1)[0])
        self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
        self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
        self.body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
        self.body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
        self.body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
        self.body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)
        self.time_step_total = self.joint_pos.shape[0]


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    robot: Articulation = scene["robot"]
    motion = K1Motion(args_cli.motion_file, sim.device)
    time_step = 0
    sim_dt = sim.get_physics_dt()

    print(f"[INFO]: Replaying K1 motion: {args_cli.motion_file}")
    print(f"[INFO]: Motion frames: {motion.time_step_total}, fps: {motion.fps}")
    print(f"[INFO]: Robot joints: {robot.data.joint_names}")
    print(f"[INFO]: Robot bodies: {robot.body_names}")

    while simulation_app.is_running():
        time_step = (time_step + 1) % motion.time_step_total

        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion.body_pos_w[time_step, 0] + scene.env_origins
        root_states[:, 3:7] = motion.body_quat_w[time_step, 0]
        root_states[:, 7:10] = motion.body_lin_vel_w[time_step, 0]
        root_states[:, 10:] = motion.body_ang_vel_w[time_step, 0]

        robot.write_root_state_to_sim(root_states)
        robot.write_joint_state_to_sim(
            motion.joint_pos[time_step].unsqueeze(0),
            motion.joint_vel[time_step].unsqueeze(0),
        )
        scene.write_data_to_sim()
        sim.render()
        scene.update(sim_dt)

        # pos_lookat = root_states[0, :3].cpu().numpy()
        # sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)


def main():
    """Main function."""
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 0.02
    sim = SimulationContext(sim_cfg)

    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()

"""Convert a Booster K1 retargeted CSV motion to the training NPZ format.

Expected CSV layout:
    root_pos_xyz(3), root_quat_xyzw(4), K1_joint_positions(22)
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import json
import pathlib
import xml.etree.ElementTree as ET

import numpy as np

from isaaclab.app import AppLauncher

DEFAULT_K1_ASSET_DIR = pathlib.Path("/home/liyunsong/RL_project/GMR/assets/booster_k1")
DEFAULT_K1_URDF = DEFAULT_K1_ASSET_DIR / "K1_22dof.urdf"
DEFAULT_K1_XML = DEFAULT_K1_ASSET_DIR / "K1_serial.xml"
DEFAULT_K1_IK_CONFIG = pathlib.Path(
    "/home/liyunsong/RL_project/GMR/general_motion_retargeting/ik_configs/smplx_to_k1.json"
)


def _joint_names_from_mjcf(xml_path: pathlib.Path) -> list[str]:
    """Read K1 motor joint order from the GMR MJCF file."""
    root = ET.parse(xml_path).getroot()
    return [motor.attrib["joint"] for motor in root.findall(".//motor")]


def _body_names_from_ik_config(ik_config_path: pathlib.Path) -> list[str]:
    """Read the robot body names used by the K1 IK config."""
    with ik_config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    names = [cfg["robot_root_name"]]
    for table_name in ("ik_match_table1", "ik_match_table2"):
        for body_name in cfg.get(table_name, {}):
            if body_name not in names:
                names.append(body_name)
    return names


# add argparse arguments
parser = argparse.ArgumentParser(description="Convert a Booster K1 retargeted CSV motion to NPZ.")
parser.add_argument("--input_file", type=str, required=True, help="The path to the input K1 motion CSV file.")
parser.add_argument("--input_fps", type=int, default=120, help="The FPS of the input motion.")
parser.add_argument(
    "--frame_range",
    nargs=2,
    type=int,
    metavar=("START", "END"),
    help=(
        "Frame range: START END, both inclusive. The frame index starts from 1. "
        "If not provided, all frames will be loaded."
    ),
)
parser.add_argument("--output_name", type=str, required=True, help="The local output NPZ path.")
parser.add_argument("--output_fps", type=int, default=50, help="The FPS of the output motion.")
parser.add_argument("--k1_urdf", type=str, default=str(DEFAULT_K1_URDF), help="Path to the K1 URDF loaded by IsaacLab.")
parser.add_argument(
    "--k1_xml",
    type=str,
    default=str(DEFAULT_K1_XML),
    help="Path to the GMR K1 MJCF file used to read the CSV joint order.",
)
parser.add_argument(
    "--ik_config",
    type=str,
    default=str(DEFAULT_K1_IK_CONFIG),
    help="Path to smplx_to_k1.json. Used to store K1 tracking body metadata in the NPZ.",
)
parser.add_argument("--upload_wandb", action="store_true", help="Also upload the generated NPZ to wandb.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

K1_JOINT_NAMES = _joint_names_from_mjcf(pathlib.Path(args_cli.k1_xml).expanduser())
K1_IK_BODY_NAMES = _body_names_from_ik_config(pathlib.Path(args_cli.ik_config).expanduser())

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul, quat_slerp


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


class MotionLoader:
    def __init__(
        self,
        motion_file: str,
        input_fps: int,
        output_fps: int,
        joint_names: list[str],
        device: torch.device,
        frame_range: tuple[int, int] | None,
    ):
        self.motion_file = motion_file
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.input_dt = 1.0 / self.input_fps
        self.output_dt = 1.0 / self.output_fps
        self.joint_names = joint_names
        self.current_idx = 0
        self.device = device
        self.frame_range = frame_range
        self._load_motion()
        self._interpolate_motion()
        self._compute_velocities()

    def _load_motion(self):
        """Loads the K1 motion from the CSV file."""
        if self.frame_range is None:
            motion = torch.from_numpy(np.loadtxt(self.motion_file, delimiter=","))
        else:
            motion = torch.from_numpy(
                np.loadtxt(
                    self.motion_file,
                    delimiter=",",
                    skiprows=self.frame_range[0] - 1,
                    max_rows=self.frame_range[1] - self.frame_range[0] + 1,
                )
            )
        if motion.ndim == 1:
            motion = motion.unsqueeze(0)

        motion = motion.to(torch.float32).to(self.device)
        expected_cols = 7 + len(self.joint_names)
        if motion.shape[1] != expected_cols:
            raise ValueError(
                f"K1 CSV has {motion.shape[1]} columns, but expected {expected_cols}: "
                f"3 root position + 4 root quaternion xyzw + {len(self.joint_names)} joints. "
                f"Joint order: {self.joint_names}"
            )

        self.motion_base_poss_input = motion[:, :3]
        self.motion_base_rots_input = motion[:, 3:7]
        self.motion_base_rots_input = self.motion_base_rots_input[:, [3, 0, 1, 2]]  # xyzw -> wxyz
        self.motion_dof_poss_input = motion[:, 7:]

        self.input_frames = motion.shape[0]
        self.duration = (self.input_frames - 1) * self.input_dt
        if self.input_frames < 3:
            raise ValueError("The input motion needs at least 3 frames to compute velocities.")
        print(f"Motion loaded ({self.motion_file}), duration: {self.duration} sec, frames: {self.input_frames}")
        print(f"K1 joint order ({len(self.joint_names)}): {self.joint_names}")

    def _interpolate_motion(self):
        """Interpolates the motion to the output FPS."""
        times = torch.arange(0, self.duration, self.output_dt, device=self.device, dtype=torch.float32)
        self.output_frames = times.shape[0]
        index_0, index_1, blend = self._compute_frame_blend(times)
        self.motion_base_poss = self._lerp(
            self.motion_base_poss_input[index_0],
            self.motion_base_poss_input[index_1],
            blend.unsqueeze(1),
        )
        self.motion_base_rots = self._slerp(
            self.motion_base_rots_input[index_0],
            self.motion_base_rots_input[index_1],
            blend,
        )
        self.motion_dof_poss = self._lerp(
            self.motion_dof_poss_input[index_0],
            self.motion_dof_poss_input[index_1],
            blend.unsqueeze(1),
        )
        print(
            f"Motion interpolated, input frames: {self.input_frames}, input fps: {self.input_fps}, output frames:"
            f" {self.output_frames}, output fps: {self.output_fps}"
        )

    def _lerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        """Linear interpolation between two tensors."""
        return a * (1 - blend) + b * blend

    def _slerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        """Spherical linear interpolation between two quaternions."""
        slerped_quats = torch.zeros_like(a)
        for i in range(a.shape[0]):
            slerped_quats[i] = quat_slerp(a[i], b[i], blend[i])
        return slerped_quats

    def _compute_frame_blend(self, times: torch.Tensor) -> torch.Tensor:
        """Computes the frame blend for the motion."""
        phase = times / self.duration
        index_0 = (phase * (self.input_frames - 1)).floor().long()
        index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1, device=self.device))
        blend = phase * (self.input_frames - 1) - index_0
        return index_0, index_1, blend

    def _compute_velocities(self):
        """Computes the velocities of the motion."""
        self.motion_base_lin_vels = torch.gradient(self.motion_base_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_dof_vels = torch.gradient(self.motion_dof_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_base_ang_vels = self._so3_derivative(self.motion_base_rots, self.output_dt)

    def _so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
        """Computes the derivative of a sequence of SO3 rotations."""
        q_prev, q_next = rotations[:-2], rotations[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))

        omega = axis_angle_from_quat(q_rel) / (2.0 * dt)
        omega = torch.cat([omega[:1], omega, omega[-1:]], dim=0)
        return omega

    def get_next_state(
        self,
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        bool,
    ]:
        """Gets the next state of the motion."""
        state = (
            self.motion_base_poss[self.current_idx : self.current_idx + 1],
            self.motion_base_rots[self.current_idx : self.current_idx + 1],
            self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
            self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
            self.motion_dof_poss[self.current_idx : self.current_idx + 1],
            self.motion_dof_vels[self.current_idx : self.current_idx + 1],
        )
        self.current_idx += 1
        reset_flag = False
        if self.current_idx >= self.output_frames:
            self.current_idx = 0
            reset_flag = True
        return state, reset_flag


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, joint_names: list[str]):
    """Runs the simulation loop."""
    motion = MotionLoader(
        motion_file=args_cli.input_file,
        input_fps=args_cli.input_fps,
        output_fps=args_cli.output_fps,
        joint_names=joint_names,
        device=sim.device,
        frame_range=args_cli.frame_range,
    )

    robot = scene["robot"]
    robot_joint_indexes = robot.find_joints(joint_names, preserve_order=True)[0]
    if len(robot_joint_indexes) != len(joint_names):
        raise RuntimeError(f"Could not find all K1 joints in the IsaacLab articulation: {joint_names}")

    missing_ik_bodies = [name for name in K1_IK_BODY_NAMES if name not in robot.body_names]
    if missing_ik_bodies:
        raise RuntimeError(f"IK config body names not found in K1 URDF: {missing_ik_bodies}")

    log = {
        "fps": np.asarray([args_cli.output_fps], dtype=np.float32),
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
        "joint_names": np.asarray(robot.data.joint_names),
        "body_names": np.asarray(robot.body_names),
        "ik_body_names": np.asarray(K1_IK_BODY_NAMES),
        "robot_root_name": np.asarray(["Trunk"]),
    }

    while simulation_app.is_running():
        (
            motion_base_pos,
            motion_base_rot,
            motion_base_lin_vel,
            motion_base_ang_vel,
            motion_dof_pos,
            motion_dof_vel,
        ), reset_flag = motion.get_next_state()

        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion_base_pos
        root_states[:, :2] += scene.env_origins[:, :2]
        root_states[:, 3:7] = motion_base_rot
        root_states[:, 7:10] = motion_base_lin_vel
        root_states[:, 10:] = motion_base_ang_vel
        robot.write_root_state_to_sim(root_states)

        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, robot_joint_indexes] = motion_dof_pos
        joint_vel[:, robot_joint_indexes] = motion_dof_vel
        robot.write_joint_state_to_sim(joint_pos, joint_vel)

        scene.write_data_to_sim()
        sim.render()
        scene.update(sim.get_physics_dt())

        pos_lookat = root_states[0, :3].cpu().numpy()
        sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)

        log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
        log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
        log["body_pos_w"].append(robot.data.body_pos_w[0, :].cpu().numpy().copy())
        log["body_quat_w"].append(robot.data.body_quat_w[0, :].cpu().numpy().copy())
        log["body_lin_vel_w"].append(robot.data.body_lin_vel_w[0, :].cpu().numpy().copy())
        log["body_ang_vel_w"].append(robot.data.body_ang_vel_w[0, :].cpu().numpy().copy())

        if reset_flag:
            for key in (
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
            ):
                log[key] = np.stack(log[key], axis=0)

            output_path = pathlib.Path(args_cli.output_name).expanduser()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(output_path, **log)
            print(f"[INFO]: Motion saved to: {output_path}")

            if args_cli.upload_wandb:
                import wandb

                collection = output_path.stem
                run = wandb.init(project="csv_to_npz", name=collection)
                print(f"[INFO]: Logging motion to wandb: {collection}")
                registry = "motions"
                logged_artifact = run.log_artifact(artifact_or_path=str(output_path), name=collection, type=registry)
                run.link_artifact(artifact=logged_artifact, target_path=f"wandb-registry-{registry}/{collection}")
                print(f"[INFO]: Motion saved to wandb registry: {registry}/{collection}")
            break


def main():
    """Main function."""
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)

    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[INFO]: Setup complete...")
    print(f"[INFO]: K1 IK config bodies: {K1_IK_BODY_NAMES}")
    run_simulator(sim, scene, joint_names=K1_JOINT_NAMES)


if __name__ == "__main__":
    main()
    simulation_app.close()

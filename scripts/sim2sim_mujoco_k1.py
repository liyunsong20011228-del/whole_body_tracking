"""Run a trained K1 ONNX policy in MuJoCo.

Example:
    python scripts/sim2sim_mujoco_k1.py \
        --onnx logs/rsl_rl/k1_flat/2026-05-29_10-54-12/exported/policy.onnx
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np


DEFAULT_K1_XML = "/home/liyunsong/RL_project/GMR/assets/booster_k1/K1_serial.xml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sim2sim a K1 tracking policy in MuJoCo.")
    parser.add_argument("--onnx", type=str, required=True, help="Path to exported policy.onnx.")
    parser.add_argument("--xml", type=str, default=DEFAULT_K1_XML, help="Path to K1 MuJoCo XML.")
    parser.add_argument("--duration", type=float, default=60.0, help="Viewer duration in seconds.")
    parser.add_argument("--policy_hz", type=float, default=50.0, help="Policy control frequency.")
    parser.add_argument("--kp_scale", type=float, default=1.0, help="Multiplier for metadata joint stiffness.")
    parser.add_argument("--kd_scale", type=float, default=1.0, help="Multiplier for metadata joint damping.")
    parser.add_argument("--action_scale", type=float, default=1.0, help="Multiplier for policy action scale.")
    parser.add_argument("--no_rate_limit", action="store_true", help="Run as fast as possible.")
    return parser.parse_args()


def _parse_csv_floats(value: str) -> np.ndarray:
    if not value:
        return np.asarray([], dtype=np.float32)
    return np.asarray([float(x) for x in value.split(",")], dtype=np.float32)


def _parse_csv_strings(value: str) -> list[str]:
    if not value:
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.asarray([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.asarray(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float32,
    )


def quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q_xyz = q[1:]
    t = 2.0 * np.cross(q_xyz, v)
    return v + q[0] * t + np.cross(q_xyz, t)


def quat_apply_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return quat_apply(quat_conj(q), v)


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    q = q / max(np.linalg.norm(q), 1e-8)
    w, x, y, z = q
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def subtract_frame(parent_pos: np.ndarray, parent_quat: np.ndarray, child_pos: np.ndarray, child_quat: np.ndarray):
    pos_b = quat_apply_inverse(parent_quat, child_pos - parent_pos)
    quat_b = quat_mul(quat_conj(parent_quat), child_quat)
    return pos_b, quat_b


def rot6d_from_quat(q: np.ndarray) -> np.ndarray:
    return quat_to_matrix(q)[:, :2].reshape(-1)


def get_joint_qpos(data, model, joint_names: list[str]) -> np.ndarray:
    values = []
    for name in joint_names:
        adr = model.joint(name).qposadr[0]
        values.append(data.qpos[adr])
    return np.asarray(values, dtype=np.float32)


def get_joint_qvel(data, model, joint_names: list[str]) -> np.ndarray:
    values = []
    for name in joint_names:
        adr = model.joint(name).dofadr[0]
        values.append(data.qvel[adr])
    return np.asarray(values, dtype=np.float32)


def set_joint_qpos(data, model, joint_names: list[str], joint_pos: np.ndarray):
    for i, name in enumerate(joint_names):
        adr = model.joint(name).qposadr[0]
        data.qpos[adr] = joint_pos[i]


def set_joint_qvel(data, model, joint_names: list[str], joint_vel: np.ndarray):
    for i, name in enumerate(joint_names):
        adr = model.joint(name).dofadr[0]
        data.qvel[adr] = joint_vel[i]


def read_body_state(data, model, body_name: str):
    body_id = model.body(body_name).id
    return data.xpos[body_id].copy(), data.xquat[body_id].copy()


def build_obs(
    data,
    model,
    joint_names: list[str],
    body_names: list[str],
    anchor_body_name: str,
    default_joint_pos: np.ndarray,
    last_action: np.ndarray,
    ref_joint_pos: np.ndarray,
    ref_joint_vel: np.ndarray,
    ref_body_pos: np.ndarray,
    ref_body_quat: np.ndarray,
) -> np.ndarray:
    robot_anchor_pos, robot_anchor_quat = read_body_state(data, model, anchor_body_name)
    anchor_idx = body_names.index(anchor_body_name)
    ref_anchor_pos = ref_body_pos[anchor_idx]
    ref_anchor_quat = ref_body_quat[anchor_idx]

    motion_anchor_pos_b, motion_anchor_quat_b = subtract_frame(
        robot_anchor_pos, robot_anchor_quat, ref_anchor_pos, ref_anchor_quat
    )
    motion_anchor_ori_b = rot6d_from_quat(motion_anchor_quat_b)

    root_lin_vel_w = data.qvel[0:3].copy()
    root_ang_vel_w = data.qvel[3:6].copy()
    base_lin_vel = quat_apply_inverse(robot_anchor_quat, root_lin_vel_w)
    base_ang_vel = quat_apply_inverse(robot_anchor_quat, root_ang_vel_w)

    joint_pos = get_joint_qpos(data, model, joint_names)
    joint_vel = get_joint_qvel(data, model, joint_names)

    return np.concatenate(
        [
            ref_joint_pos,
            ref_joint_vel,
            motion_anchor_pos_b,
            motion_anchor_ori_b,
            base_lin_vel,
            base_ang_vel,
            joint_pos - default_joint_pos,
            joint_vel,
            last_action,
        ]
    ).astype(np.float32)[None, :]


def main():
    args = parse_args()

    try:
        import mujoco
        import mujoco.viewer
        import onnxruntime as ort
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Missing dependency: {exc.name}. Install/use an environment with mujoco, onnxruntime, numpy."
        ) from exc

    onnx_path = str(Path(args.onnx).expanduser())
    xml_path = str(Path(args.xml).expanduser())

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    metadata = session.get_modelmeta().custom_metadata_map

    joint_names = _parse_csv_strings(metadata["joint_names"])
    body_names = _parse_csv_strings(metadata["body_names"])
    anchor_body_name = metadata["anchor_body_name"]
    default_joint_pos = _parse_csv_floats(metadata["default_joint_pos"])
    action_scale = _parse_csv_floats(metadata["action_scale"]) * args.action_scale
    kp = _parse_csv_floats(metadata["joint_stiffness"]) * args.kp_scale
    kd = _parse_csv_floats(metadata["joint_damping"]) * args.kd_scale

    if not (len(joint_names) == len(default_joint_pos) == len(action_scale) == len(kp) == len(kd)):
        raise RuntimeError("ONNX metadata joint arrays have inconsistent lengths.")

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # Ask the ONNX model for the first reference frame and initialize MuJoCo near the motion.
    zero_obs = np.zeros((1, session.get_inputs()[0].shape[1]), dtype=np.float32)
    outputs = session.run(None, {"obs": zero_obs, "time_step": np.asarray([[0]], dtype=np.float32)})
    ref_joint_pos = outputs[1][0].astype(np.float32)
    ref_joint_vel = outputs[2][0].astype(np.float32)
    ref_body_pos = outputs[3][0].astype(np.float32)
    ref_body_quat = outputs[4][0].astype(np.float32)

    anchor_idx = body_names.index(anchor_body_name)
    data.qpos[:3] = ref_body_pos[anchor_idx]
    data.qpos[3:7] = ref_body_quat[anchor_idx]
    set_joint_qpos(data, model, joint_names, ref_joint_pos)
    set_joint_qvel(data, model, joint_names, ref_joint_vel)
    mujoco.mj_forward(model, data)

    policy_dt = 1.0 / args.policy_hz
    substeps = max(1, round(policy_dt / model.opt.timestep))
    max_steps = int(args.duration * args.policy_hz)
    last_action = np.zeros(len(joint_names), dtype=np.float32)

    print(f"[INFO] ONNX: {onnx_path}")
    print(f"[INFO] XML : {xml_path}")
    print(f"[INFO] joints ({len(joint_names)}): {joint_names}")
    print(f"[INFO] body_names ({len(body_names)}): {body_names}")
    print(f"[INFO] policy_hz={args.policy_hz}, mujoco_dt={model.opt.timestep}, substeps={substeps}")
    print("[INFO] Close the MuJoCo viewer window to stop.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat = data.xpos[model.body(anchor_body_name).id]
        viewer.cam.distance = 2.0
        viewer.cam.elevation = -10
        viewer.cam.azimuth = 210

        for time_step in range(max_steps):
            if not viewer.is_running():
                break

            obs = build_obs(
                data,
                model,
                joint_names,
                body_names,
                anchor_body_name,
                default_joint_pos,
                last_action,
                ref_joint_pos,
                ref_joint_vel,
                ref_body_pos,
                ref_body_quat,
            )

            outputs = session.run(
                None,
                {"obs": obs, "time_step": np.asarray([[time_step]], dtype=np.float32)},
            )
            action = outputs[0][0].astype(np.float32)
            ref_joint_pos = outputs[1][0].astype(np.float32)
            ref_joint_vel = outputs[2][0].astype(np.float32)
            ref_body_pos = outputs[3][0].astype(np.float32)
            ref_body_quat = outputs[4][0].astype(np.float32)

            target_joint_pos = default_joint_pos + action * action_scale
            for _ in range(substeps):
                joint_pos = get_joint_qpos(data, model, joint_names)
                joint_vel = get_joint_qvel(data, model, joint_names)
                torque = kp * (target_joint_pos - joint_pos) - kd * joint_vel
                for i, name in enumerate(joint_names):
                    data.ctrl[model.actuator(name).id] = torque[i]
                mujoco.mj_step(model, data)

            last_action = action
            viewer.cam.lookat = data.xpos[model.body(anchor_body_name).id]
            viewer.sync()

            if not args.no_rate_limit:
                time.sleep(policy_dt)


if __name__ == "__main__":
    main()

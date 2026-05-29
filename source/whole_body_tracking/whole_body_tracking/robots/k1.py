import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

K1_URDF_PATH = "/home/liyunsong/RL_project/GMR/assets/booster_k1/K1_22dof.urdf"

LEG_NATURAL_FREQ = 4.0 * 2.0 * 3.1415926535
ARM_NATURAL_FREQ = 10.0 * 2.0 * 3.1415926535
HEAD_NATURAL_FREQ = 10.0 * 2.0 * 3.1415926535

LEG_DAMPING_RATIO = 1.5
KNEE_DAMPING_RATIO = 1.0
ARM_DAMPING_RATIO = 2.0
HEAD_DAMPING_RATIO = 2.0

K1_ARMATURE = {
    ".*_Hip_Pitch": 0.0478125,
    ".*_Hip_Roll": 0.0339552,
    ".*_Hip_Yaw": 0.0282528,
    ".*_Knee_Pitch": 0.095625,
    ".*_Ankle_Pitch": 0.0565056,
    ".*_Ankle_Roll": 0.0565056,
    ".*_Shoulder_Pitch": 0.001,
    ".*_Shoulder_Roll": 0.001,
    ".*_Elbow_Pitch": 0.001,
    ".*_Elbow_Yaw": 0.001,
    ".*Head.*": 0.001,
}

K1_EFFORT_LIMIT = {
    ".*_Hip_Pitch": 68.0,
    ".*_Hip_Roll": 76.0,
    ".*_Hip_Yaw": 38.3,
    ".*_Knee_Pitch": 112.0,
    ".*_Ankle_Pitch": 38.3,
    ".*_Ankle_Roll": 38.3,
    ".*_Shoulder_Pitch": 14.0,
    ".*_Shoulder_Roll": 14.0,
    ".*_Elbow_Pitch": 14.0,
    ".*_Elbow_Yaw": 14.0,
    ".*Head.*": 6.0,
}

K1_VELOCITY_LIMIT = {
    ".*_Hip_Pitch": 14.66,
    ".*_Hip_Roll": 12.57,
    ".*_Hip_Yaw": 17.59,
    ".*_Knee_Pitch": 12.57,
    ".*_Ankle_Pitch": 17.59,
    ".*_Ankle_Roll": 17.59,
    ".*_Shoulder_Pitch": 33.51,
    ".*_Shoulder_Roll": 33.51,
    ".*_Elbow_Pitch": 33.51,
    ".*_Elbow_Yaw": 33.51,
    ".*Head.*": 7.85,
}

K1_NATURAL_FREQ = {
    ".*_Hip_Pitch": LEG_NATURAL_FREQ,
    ".*_Hip_Roll": LEG_NATURAL_FREQ,
    ".*_Hip_Yaw": LEG_NATURAL_FREQ,
    ".*_Knee_Pitch": LEG_NATURAL_FREQ,
    ".*_Ankle_Pitch": LEG_NATURAL_FREQ,
    ".*_Ankle_Roll": LEG_NATURAL_FREQ,
    ".*_Shoulder_Pitch": ARM_NATURAL_FREQ,
    ".*_Shoulder_Roll": ARM_NATURAL_FREQ,
    ".*_Elbow_Pitch": ARM_NATURAL_FREQ,
    ".*_Elbow_Yaw": ARM_NATURAL_FREQ,
    ".*Head.*": HEAD_NATURAL_FREQ,
}

K1_DAMPING_RATIO = {
    ".*_Hip_Pitch": LEG_DAMPING_RATIO,
    ".*_Hip_Roll": LEG_DAMPING_RATIO,
    ".*_Hip_Yaw": LEG_DAMPING_RATIO,
    ".*_Knee_Pitch": KNEE_DAMPING_RATIO,
    ".*_Ankle_Pitch": LEG_DAMPING_RATIO,
    ".*_Ankle_Roll": LEG_DAMPING_RATIO,
    ".*_Shoulder_Pitch": ARM_DAMPING_RATIO,
    ".*_Shoulder_Roll": ARM_DAMPING_RATIO,
    ".*_Elbow_Pitch": ARM_DAMPING_RATIO,
    ".*_Elbow_Yaw": ARM_DAMPING_RATIO,
    ".*Head.*": HEAD_DAMPING_RATIO,
}

K1_STIFFNESS = {name: K1_ARMATURE[name] * K1_NATURAL_FREQ[name] ** 2 for name in K1_ARMATURE}
K1_DAMPING = {
    name: 2.0 * K1_DAMPING_RATIO[name] * K1_ARMATURE[name] * K1_NATURAL_FREQ[name] for name in K1_ARMATURE
}

K1_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=False,
        asset_path=K1_URDF_PATH,
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
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.57),
        joint_pos={
            "Left_Shoulder_Roll": -1.3,
            "Right_Shoulder_Roll": 1.3,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "joints": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_Hip_Pitch",
                ".*_Hip_Roll",
                ".*_Hip_Yaw",
                ".*_Knee_Pitch",
                ".*_Ankle_Pitch",
                ".*_Ankle_Roll",
                ".*_Shoulder_Pitch",
                ".*_Shoulder_Roll",
                ".*_Elbow_Pitch",
                ".*_Elbow_Yaw",
                ".*Head.*",
            ],
            effort_limit_sim=K1_EFFORT_LIMIT,
            velocity_limit_sim=K1_VELOCITY_LIMIT,
            stiffness=K1_STIFFNESS,
            damping=K1_DAMPING,
            armature=K1_ARMATURE,
        ),
    },
)

K1_ACTION_SCALE = {
    name: 0.25 * K1_EFFORT_LIMIT[name] / K1_STIFFNESS[name] for name in K1_ARMATURE if K1_STIFFNESS[name]
}

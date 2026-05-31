import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg

from whole_body_tracking.robots import actuator
from whole_body_tracking.robots.actuator import BoosterDelayedPDActuatorCfg


K1_URDF_PATH = "/home/liyunsong/RL_project/booster_assets/robots/K1/K1_22dof.urdf"


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
        "legs": BoosterDelayedPDActuatorCfg(
            min_delay=2,
            max_delay=8,
            joint_names_expr=[
                ".*_Hip_Pitch",
                ".*_Hip_Roll",
                ".*_Hip_Yaw",
                ".*_Knee_Pitch",
            ],
            booster_joint_cfgs={
                ".*_Hip_Pitch": actuator.BoosterJointE6408(natural_freq=4.0, damping_ratio=1.5),
                ".*_Hip_Roll": actuator.BoosterJointE4315(natural_freq=4.0, damping_ratio=1.5),
                ".*_Hip_Yaw": actuator.BoosterJointE4310(natural_freq=4.0, damping_ratio=1.5),
                ".*_Knee_Pitch": actuator.BoosterJointE6416(natural_freq=4.0, damping_ratio=1.0),
            },
        ),
        "feet": BoosterDelayedPDActuatorCfg(
            min_delay=2,
            max_delay=8,
            joint_names_expr=[
                ".*_Ankle_Pitch",
                ".*_Ankle_Roll",
            ],
            booster_joint_cfgs={
                ".*_Ankle_Pitch": actuator.BoosterK1AnkleParaWrapperCfg(
                    base_joint_cfg=actuator.BoosterJointE4310(),
                    serial_index=0,
                    natural_freq=4.0,
                    damping_ratio=1.5,
                ),
                ".*_Ankle_Roll": actuator.BoosterK1AnkleParaWrapperCfg(
                    base_joint_cfg=actuator.BoosterJointE4310(),
                    serial_index=1,
                    natural_freq=4.0,
                    damping_ratio=1.5,
                ),
            },
        ),
        "arms": BoosterDelayedPDActuatorCfg(
            min_delay=2,
            max_delay=8,
            joint_names_expr=[
                ".*_Shoulder_Pitch",
                ".*_Shoulder_Roll",
                ".*_Elbow_Pitch",
                ".*_Elbow_Yaw",
            ],
            booster_joint_cfgs=actuator.BoosterJointR14(),
        ),
        "head": BoosterDelayedPDActuatorCfg(
            min_delay=2,
            max_delay=8,
            joint_names_expr=[".*Head.*"],
            booster_joint_cfgs=actuator.BoosterJointHT4438(),
        ),
    },
)


K1_ACTION_SCALE = {}
for actuator_cfg in K1_CFG.actuators.values():
    effort_limit = actuator_cfg.effort_limit_sim
    stiffness = actuator_cfg.stiffness
    joint_names = actuator_cfg.joint_names_expr
    if not isinstance(effort_limit, dict):
        effort_limit = {name: effort_limit for name in joint_names}
    if not isinstance(stiffness, dict):
        stiffness = {name: stiffness for name in joint_names}
    for name in joint_names:
        if name in effort_limit and name in stiffness and stiffness[name]:
            K1_ACTION_SCALE[name] = 0.25 * effort_limit[name] / stiffness[name]

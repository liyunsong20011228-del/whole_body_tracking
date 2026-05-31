from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from whole_body_tracking.robots.k1 import K1_ACTION_SCALE, K1_CFG
from whole_body_tracking.tasks.tracking.config.k1.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg


K1_TRACKING_BODY_NAMES = [
    "Trunk",
    "Left_Hip_Yaw",
    "Left_Shank",
    "left_foot_link",
    "Right_Hip_Yaw",
    "Right_Shank",
    "right_foot_link",
    "Left_Arm_3",
    "left_hand_link",
    "Right_Arm_3",
    "right_hand_link",
    "Head_2",
]

K1_END_EFFECTOR_BODY_NAMES = [
    "left_foot_link",
    "right_foot_link",
    "left_hand_link",
    "right_hand_link",
]


@configclass
class K1FlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = K1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = K1_ACTION_SCALE
        self.commands.motion.anchor_body_name = "Trunk"
        self.commands.motion.body_names = K1_TRACKING_BODY_NAMES

        self.events.base_com.params["asset_cfg"] = SceneEntityCfg("robot", body_names="Trunk")
        self.terminations.ee_body_pos.params["body_names"] = K1_END_EFFECTOR_BODY_NAMES
        self.rewards.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces",
            body_names=[
                r"^(?!left_foot_link$)(?!right_foot_link$)(?!left_hand_link$)(?!right_hand_link$).+$",
            ],
        )


@configclass
class K1FlatWoStateEstimationEnvCfg(K1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None


@configclass
class K1FlatLowFreqEnvCfg(K1FlatWoStateEstimationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.decimation = round(self.decimation / LOW_FREQ_SCALE)
        self.rewards.action_rate_l2.weight *= LOW_FREQ_SCALE

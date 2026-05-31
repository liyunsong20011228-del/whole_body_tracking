from __future__ import annotations

from dataclasses import MISSING

import torch
from collections.abc import Sequence

from isaaclab.actuators import DelayedPDActuator, DelayedPDActuatorCfg, ImplicitActuator, ImplicitActuatorCfg
from isaaclab.utils import DelayBuffer, configclass
from isaaclab.utils.types import ArticulationActions


class DelayedImplicitActuator(ImplicitActuator):
    """Ideal PD actuator with delayed command application.

    This class extends the :class:`IdealPDActuator` class by adding a delay to the actuator commands. The delay
    is implemented using a circular buffer that stores the actuator commands for a certain number of physics steps.
    The most recent actuation value is pushed to the buffer at every physics step, but the final actuation value
    applied to the simulation is lagged by a certain number of physics steps.

    The amount of time lag is configurable and can be set to a random value between the minimum and maximum time
    lag bounds at every reset. The minimum and maximum time lag values are set in the configuration instance passed
    to the class.
    """

    cfg: DelayedImplicitActuatorCfg
    """The configuration for the actuator model."""

    def __init__(self, cfg: DelayedImplicitActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        # instantiate the delay buffers
        self.positions_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self.velocities_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self.efforts_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        # all of the envs
        self._ALL_INDICES = torch.arange(self._num_envs, dtype=torch.long, device=self._device)

    def reset(self, env_ids: Sequence[int]):
        super().reset(env_ids)
        # number of environments (since env_ids can be a slice)
        if env_ids is None or env_ids == slice(None):
            num_envs = self._num_envs
        else:
            num_envs = len(env_ids)
        # set a new random delay for environments in env_ids
        time_lags = torch.randint(
            low=self.cfg.min_delay,
            high=self.cfg.max_delay + 1,
            size=(num_envs,),
            dtype=torch.int,
            device=self._device,
        )
        # set delays
        self.positions_delay_buffer.set_time_lag(time_lags, env_ids)
        self.velocities_delay_buffer.set_time_lag(time_lags, env_ids)
        self.efforts_delay_buffer.set_time_lag(time_lags, env_ids)
        # reset buffers
        self.positions_delay_buffer.reset(env_ids)
        self.velocities_delay_buffer.reset(env_ids)
        self.efforts_delay_buffer.reset(env_ids)

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        # apply delay based on the delay the model for all the setpoints
        control_action.joint_positions = self.positions_delay_buffer.compute(control_action.joint_positions)
        control_action.joint_velocities = self.velocities_delay_buffer.compute(control_action.joint_velocities)
        control_action.joint_efforts = self.efforts_delay_buffer.compute(control_action.joint_efforts)
        # compte actuator model
        return super().compute(control_action, joint_pos, joint_vel)


@configclass
class DelayedImplicitActuatorCfg(ImplicitActuatorCfg):
    """Configuration for a delayed PD actuator."""

    class_type: type = DelayedImplicitActuator

    min_delay: int = 0
    """Minimum number of physics time-steps with which the actuator command may be delayed. Defaults to 0."""

    max_delay: int = 0
    """Maximum number of physics time-steps with which the actuator command may be delayed. Defaults to 0."""


class BoosterDelayedPDActuator(DelayedPDActuator):
    """Delayed PD actuator with speed-dependent torque clipping."""

    cfg: BoosterDelayedPDActuatorCfg

    def __init__(self, cfg: "BoosterDelayedPDActuatorCfg", *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.knee_point_velocity = self._parse_joint_parameter(cfg.knee_point_velocity, self.velocity_limit)
        self.knee_point_velocity = torch.clamp(self.knee_point_velocity, min=0.0)
        self.knee_point_velocity = torch.minimum(self.knee_point_velocity, self.velocity_limit)
        self._joint_vel = torch.zeros_like(self.computed_effort)
        self._denom = (self.velocity_limit - self.knee_point_velocity).clamp(min=1e-6)

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        self._joint_vel[:] = joint_vel
        return super().compute(control_action, joint_pos, joint_vel)

    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        joint_vel_abs = self._joint_vel.abs()
        v_max = self.velocity_limit
        tau_max = self.effort_limit

        non_positive_vmax = v_max <= 0.0
        non_finite_vmax = ~torch.isfinite(v_max)

        tau_linear = tau_max * (v_max - joint_vel_abs) / self._denom
        max_effort = tau_linear.clamp(min=0.0).clamp(max=tau_max)
        max_effort = torch.where(non_finite_vmax, tau_max, max_effort)
        max_effort = torch.where(non_positive_vmax, torch.zeros_like(max_effort), max_effort)
        return torch.clip(effort, min=-max_effort, max=max_effort)


@configclass
class BoosterJointCfg:
    """Configuration for Booster joint models."""

    joint_model_name: str = MISSING

    effort_limit: float = MISSING
    velocity_limit: float = MISSING
    knee_point_velocity: float = MISSING
    armature: float = MISSING

    stiffness: float | None = None
    damping: float | None = None

    natural_freq: float = 10.0
    damping_ratio: float = 2.0

    def __post_init__(self):
        if self.stiffness is None:
            self.stiffness = self.armature * (2.0 * 3.1415926535 * self.natural_freq) ** 2
        if self.damping is None:
            self.damping = 2.0 * self.damping_ratio * self.armature * (2.0 * 3.1415926535 * self.natural_freq)


@configclass
class BoosterDelayedActuatorCfg(DelayedPDActuatorCfg):
    """Configuration for Booster delayed actuator models."""

    class_type: type = MISSING

    knee_point_velocity: dict[str, float] | float | None = None
    stiffness: dict[str, float] | float | None = None
    damping: dict[str, float] | float | None = None
    booster_joint_cfgs: dict[str, BoosterJointCfg] | BoosterJointCfg | None = None

    def __post_init__(self):
        if self.booster_joint_cfgs is None:
            return

        if isinstance(self.booster_joint_cfgs, BoosterJointCfg):
            self.effort_limit_sim = self.booster_joint_cfgs.effort_limit
            self.velocity_limit_sim = self.booster_joint_cfgs.velocity_limit
            self.knee_point_velocity = self.booster_joint_cfgs.knee_point_velocity
            self.armature = self.booster_joint_cfgs.armature
            if self.stiffness is None:
                self.stiffness = self.booster_joint_cfgs.stiffness
            if self.damping is None:
                self.damping = self.booster_joint_cfgs.damping
            return

        self.effort_limit_sim = {
            joint_name: joint_cfg.effort_limit for joint_name, joint_cfg in self.booster_joint_cfgs.items()
        }
        self.velocity_limit_sim = {
            joint_name: joint_cfg.velocity_limit for joint_name, joint_cfg in self.booster_joint_cfgs.items()
        }
        self.armature = {joint_name: joint_cfg.armature for joint_name, joint_cfg in self.booster_joint_cfgs.items()}
        self.knee_point_velocity = {
            joint_name: joint_cfg.knee_point_velocity for joint_name, joint_cfg in self.booster_joint_cfgs.items()
        }
        if self.stiffness is None:
            self.stiffness = {
                joint_name: joint_cfg.stiffness for joint_name, joint_cfg in self.booster_joint_cfgs.items()
            }
        if self.damping is None:
            self.damping = {joint_name: joint_cfg.damping for joint_name, joint_cfg in self.booster_joint_cfgs.items()}


@configclass
class BoosterDelayedPDActuatorCfg(BoosterDelayedActuatorCfg):
    """Configuration for Booster delayed PD actuators."""

    class_type: type = BoosterDelayedPDActuator


@configclass
class ParallelJointWrapperCfg(BoosterJointCfg):
    """Map a base serial joint model to one joint in a parallel mechanism."""

    joint_model_name: str = "ParallelJointWrapper"

    effort_ratio: tuple[float, float] = MISSING
    velocity_ratio: tuple[float, float] = MISSING
    armature_ratio: tuple[float, float] = MISSING
    knee_point_velocity_ratio: tuple[float, float] = (1.0, 1.0)

    base_joint_cfg: BoosterJointCfg = MISSING
    serial_index: int = MISSING

    def __post_init__(self):
        self.effort_limit = self.effort_ratio[self.serial_index] * self.base_joint_cfg.effort_limit
        self.velocity_limit = self.velocity_ratio[self.serial_index] * self.base_joint_cfg.velocity_limit
        self.knee_point_velocity = (
            self.knee_point_velocity_ratio[self.serial_index] * self.base_joint_cfg.knee_point_velocity
        )
        self.armature = self.armature_ratio[self.serial_index] * self.base_joint_cfg.armature
        self.joint_model_name = f"{self.joint_model_name}({self.base_joint_cfg.joint_model_name})[{self.serial_index}]"
        super().__post_init__()


@configclass
class BoosterK1AnkleParaWrapperCfg(ParallelJointWrapperCfg):
    """Parallel-joint wrapper for K1 ankle joints."""

    joint_model_name: str = "BoosterK1AnkleParaWrapper"
    effort_ratio: tuple[float, float] = (1.0, 1.0)
    velocity_ratio: tuple[float, float] = (1.0, 1.0)
    armature_ratio: tuple[float, float] = (2.0, 2.0)


@configclass
class BoosterJointE6408(BoosterJointCfg):
    """K1 hip pitch motor model."""

    joint_model_name: str = "E6408"
    effort_limit: float = 68.0
    velocity_limit: float = 14.66
    knee_point_velocity: float = 1.88
    armature: float = 0.0478125


@configclass
class BoosterJointE4315(BoosterJointCfg):
    """K1 hip roll motor model."""

    joint_model_name: str = "E4315"
    effort_limit: float = 76.0
    velocity_limit: float = 12.57
    knee_point_velocity: float = 2.62
    armature: float = 0.0339552


@configclass
class BoosterJointE4310(BoosterJointCfg):
    """K1 hip yaw and ankle base motor model."""

    joint_model_name: str = "E4310"
    effort_limit: float = 38.3
    velocity_limit: float = 17.59
    knee_point_velocity: float = 7.85
    armature: float = 0.0282528


@configclass
class BoosterJointE6416(BoosterJointCfg):
    """K1 knee motor model."""

    joint_model_name: str = "E6416"
    effort_limit: float = 112.0
    velocity_limit: float = 12.57
    knee_point_velocity: float = 2.09
    armature: float = 0.095625


@configclass
class BoosterJointR14(BoosterJointCfg):
    """K1 arm motor model."""

    joint_model_name: str = "R14"
    effort_limit: float = 14.0
    velocity_limit: float = 33.51
    knee_point_velocity: float = 5.24
    armature: float = 0.001


@configclass
class BoosterJointHT4438(BoosterJointCfg):
    """K1 head motor model."""

    joint_model_name: str = "HT4438"
    effort_limit: float = 6.0
    velocity_limit: float = 7.85
    knee_point_velocity: float = 10.47
    armature: float = 0.001

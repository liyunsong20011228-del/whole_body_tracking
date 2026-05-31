# Booster K1 格斗动作模仿学习经验文档

这份文档总结本仓库里 Booster K1 格斗动作模仿学习的完整流程：从 GMR/CSV 动作数据，到 IsaacLab 可训练的 NPZ，再到 RSL-RL 训练、播放、导出 ONNX，以及后续 sim2sim 检查。重点不是只记录命令，而是把这次适配 K1 时踩过的坑、容易混淆的问题和关键判断依据放在一起，方便以后复现和交接。

## 1. 项目目标

目标是在 `whole_body_tracking` 框架中，让 Booster K1 机器人学习并复现格斗/全身动作。整体思路是：

1. 先把重定向后的 K1 动作转成训练需要的 `motion.npz` 格式。
2. 在 IsaacLab 中注册 K1 tracking 任务。
3. 用 RSL-RL 训练策略跟踪参考动作。
4. 用 `play` 脚本回放和导出策略。
5. 导出 TorchScript/ONNX 策略；当前 `booster_deploy` 主路径使用 TorchScript `.pt`，ONNX 主要用于 metadata 检查和其它部署链路。

这套流程和 G1/humanoid 的原始流程很像，但 K1 有自己的关节命名、URDF/MJCF 资产、脚踝并联机构、执行器参数和部署约束，所以不能简单复制 G1 配置。

## 2. 关键代码位置

- K1 机器人资产和执行器配置：`source/whole_body_tracking/whole_body_tracking/robots/k1.py`
- Booster 执行器模型：`source/whole_body_tracking/whole_body_tracking/robots/actuator.py`
- K1 环境配置：`source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/k1/flat_env_cfg.py`
- K1 gym 任务注册：`source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/k1/__init__.py`
- K1 PPO 配置：`source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/k1/agents/rsl_rl_ppo_cfg.py`
- CSV 转 NPZ：`scripts/csv_to_npz_k1.py`
- NPZ 回放检查：`scripts/replay_npz_k1.py`
- 训练：`scripts/rsl_rl/train.py`
- 播放/导出：`scripts/rsl_rl/play.py`
- K1 专用导出脚本：`scripts/rsl_rl/play_k1.py`
- ONNX 导出和 metadata：`source/whole_body_tracking/whole_body_tracking/utils/exporter.py`
- MuJoCo sim2sim/部署工程：`~/RL_project/booster_deploy`

当前仓库里已有的样例动作：

- `motions/k1/135_06_stageii.csv`
- `motions/k1/135_06_stageii.npz`

## 3. 全流程概览

### 3.1 动作重定向输出

K1 动作数据来自 GMR 重定向流程。这里假设已经得到 CSV：

```bash
motions/k1/135_06_stageii.csv
```

CSV 期望格式是：

```text
root_pos_xyz(3), root_quat_xyzw(4), K1_joint_positions(22)
```

这里第一个坑是四元数格式。GMR/CSV 里是 `xyzw`，IsaacLab 里通常使用 `wxyz`。`csv_to_npz_k1.py` 里专门做了转换：

```python
self.motion_base_rots_input = self.motion_base_rots_input[:, [3, 0, 1, 2]]
```

如果这个地方忘了转，动作回放时根节点姿态会明显不对，轻则方向错，重则机器人一开始就翻。

### 3.2 CSV 转 NPZ

转换脚本：

```bash
python scripts/csv_to_npz_k1.py \
    --input_file motions/k1/135_06_stageii.csv \
    --output_name motions/k1/135_06_stageii.npz \
    --input_fps 120 \
    --output_fps 50
```

常用可选项：

```bash
--frame_range START END
```

`frame_range` 是 1-based，并且包含起止帧。这个点容易和 Python 的 0-based、左闭右开切片混淆。

转换脚本做了几件关键事情：

1. 从 GMR 的 K1 MJCF 读取 motor joint 顺序。
2. 从 `smplx_to_k1.json` 读取 IK 里使用的 body 名称。
3. 读取 CSV 中的 root pose 和 22 个关节角。
4. 把输入 FPS 插值到训练输出 FPS，默认输出 50 Hz。
5. 通过 IsaacLab 加载 K1 URDF，写入 root/joint state，再从仿真 articulation 里记录：
   - `joint_pos`
   - `joint_vel`
   - `body_pos_w`
   - `body_quat_w`
   - `body_lin_vel_w`
   - `body_ang_vel_w`
   - `joint_names`
   - `body_names`
   - `ik_body_names`
   - `robot_root_name`

这里的设计很重要：不要只保存 CSV 里的关节角。训练 reward 需要 body-level 的位置、姿态、线速度和角速度，所以必须通过 IsaacLab articulation 生成和真实 URDF body 对齐的数据。

### 3.3 NPZ 回放检查

训练前先回放 NPZ：

```bash
python scripts/replay_npz_k1.py \
    --motion_file motions/k1/135_06_stageii.npz
```

检查点：

1. 机器人是否从合理高度开始。
2. 根节点方向是否正确。
3. 脚、手、头这些关键 body 是否跟参考动作一致。
4. 动作是否存在明显抖动、穿地或关节顺序错乱。
5. `joint_names` 和 `body_names` 是否符合 K1 URDF。

如果这里都不对，不要直接训练。训练只会把数据问题放大。

## 4. K1 机器人配置

K1 机器人在 `robots/k1.py` 中定义，核心是：

```python
K1_URDF_PATH = "/home/liyunsong/RL_project/booster_assets/robots/K1/K1_22dof.urdf"
```

这里有一个工程坑：当前路径是本机绝对路径。换机器或给别人复现时，需要改成对方机器上的 K1 URDF 路径，或者后续把它改成环境变量/配置项。

### 4.1 初始状态

当前 K1 初始状态：

```python
pos=(0.0, 0.0, 0.57)
joint_pos={
    "Left_Shoulder_Roll": -1.3,
    "Right_Shoulder_Roll": 1.3,
}
```

肩膀初始值不是随便设的。K1 手臂如果完全默认，可能与身体姿态、动作初始帧或碰撞状态不匹配。给肩 roll 一个更接近自然下垂/展开的初始角，可以减少 reset 初期的不稳定。

### 4.2 URDF 导入参数

关键设置：

```python
fix_base=False
replace_cylinders_with_capsules=False
activate_contact_sensors=True
enabled_self_collisions=True
solver_position_iteration_count=8
solver_velocity_iteration_count=4
```

其中 `fix_base=False` 是全身运动必须的；`activate_contact_sensors=True` 是为了 contact reward/termination；自碰撞开启后更接近真实，但也会让一些激烈格斗动作更容易暴露穿模和初始姿态问题。

### 4.3 执行器从隐式 PD 改成 Booster 延迟 PD

最早 K1 配置可以用普通 `ImplicitActuatorCfg`，但为了更贴近 Booster K1，当前改成了 `BoosterDelayedPDActuatorCfg`：

- 腿：hip pitch/roll/yaw、knee pitch
- 脚：ankle pitch/roll
- 手臂：shoulder pitch/roll、elbow pitch/yaw
- 头：Head 相关关节

这里要区分两个层次：

- `ImplicitActuatorCfg` 是 IsaacLab 里的理想/隐式 PD actuator。它主要给仿真关节设置 `stiffness`、`damping`、`effort_limit`、`velocity_limit`、`armature` 等参数，可以理解成“仿真里比较理想的 PD 驱动”。
- `BoosterDelayedPDActuatorCfg` 是 Booster 专用的延迟 PD actuator。它仍然接收 policy 输出的位置目标，但底层会模拟控制延迟、按 Booster 电机型号填参数，并根据当前关节速度做力矩裁剪，因此比普通 `ImplicitActuatorCfg` 更接近真实 Booster 执行器。

每类关节使用对应电机型号：

- `E6408`：hip pitch
- `E4315`：hip roll
- `E4310`：hip yaw 和 ankle 基础电机
- `E6416`：knee
- `R14`：arm
- `HT4438`：head

这次一个关键改动是把 K1 脚踝作为并联关节包装：

```python
BoosterK1AnkleParaWrapperCfg(...)
```

它会根据基础电机参数推导 ankle pitch/roll 的 effort、velocity、armature 等参数。K1 的脚踝不能只当普通单电机关节看，否则仿真能力和部署侧会不一致。

这里的“并联机构参数等效”可以这样理解：普通关节通常是“一个电机直接控制一个自由度”，例如一个膝盖电机主要控制膝盖 pitch。但 K1 脚踝不是这么简单，ankle pitch/roll 背后有更复杂的机械耦合，可以理解成多个电机/连杆共同形成脚踝的两个自由度。真实结构很复杂，训练时不一定把完整机械传动都精确建出来，于是就在仿真关节上放一组“等效参数”。

所谓等效参数，就是用一组近似的 `effort_limit`、`velocity_limit`、`armature`、`stiffness`、`damping`，让仿真里的 ankle pitch/roll 表现得尽量像真实 K1 脚踝。`BoosterK1AnkleParaWrapperCfg` 做的事就是：拿基础电机 `E4310` 的参数，再按 K1 脚踝并联机构关系，换算成 ankle pitch/roll 这两个仿真关节上使用的等效执行器参数。

这个近似很重要。如果把 K1 脚踝当成普通单电机关节，policy 可能会以为脚踝能输出某种力矩、达到某种速度、拥有某种惯量和响应速度；但真实 K1 脚踝由于并联机构，实际表现并不完全一样。这样训练时看着能站住，到 MuJoCo 或真机上可能落脚响应不对，最后出现不稳甚至摔倒。

所以“换成 `BoosterDelayedPDActuatorCfg`”和“把 K1 脚踝作为并联关节包装”不是同一个概念，而是父子关系：

```text
BoosterDelayedPDActuatorCfg
├─ 负责 actuator 类型：延迟 PD + 速度相关力矩裁剪
├─ 负责接收每个关节的 Booster 电机参数
└─ 其中脚踝关节的参数由 BoosterK1AnkleParaWrapperCfg 生成
```

一句话总结：`ImplicitActuatorCfg -> BoosterDelayedPDActuatorCfg` 是把整个 K1 的关节驱动从“理想仿真 PD”换成“更像 Booster 真实电机的延迟 PD”；脚踝并联关节包装只是这个新 actuator 配置里针对 K1 ankle 的特殊参数建模。

### 4.4 延迟和速度相关力矩裁剪

`actuator.py` 中实现了 `BoosterDelayedPDActuator`，它做了两件事：

1. 使用 IsaacLab 的 delayed PD actuator 机制，引入控制延迟。
2. 根据关节速度做力矩裁剪，速度越接近最大速度，可输出力矩越小。

这比固定 effort limit 更贴近真实电机能力。格斗动作里有很多快速摆腿、摆臂，如果只看静态最大力矩，仿真策略可能学出真实电机跟不上的动作。

当前延迟配置：

```python
min_delay=2
max_delay=8
```

仿真步长 `dt=0.005`，所以延迟大约是 10 ms 到 40 ms。这个范围会增加 sim2real 鲁棒性，但也会让训练变难。

### 4.5 Action scale

K1 的动作尺度不是手写常数，而是从每组 actuator 的 effort/stiffness 推出来：

```python
K1_ACTION_SCALE[name] = 0.25 * effort_limit[name] / stiffness[name]
```

直觉上，policy 输出 action 后走 joint-position target。`effort / stiffness` 约等于在最大力矩下可容忍的位置偏差，再乘一个 0.25 做保守缩放。这个尺度太大，动作会猛、容易撞关节限位；太小，跟踪不了大幅格斗动作。

## 5. K1 Tracking 环境配置

K1 环境在 `config/k1/flat_env_cfg.py`。

### 5.1 跟踪 body 选择

当前 K1 跟踪的 body：

```python
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
```

末端 body：

```python
K1_END_EFFECTOR_BODY_NAMES = [
    "left_foot_link",
    "right_foot_link",
    "left_hand_link",
    "right_hand_link",
]
```

这里的选择很关键。全身动作模仿不是每个 body 都必须跟踪，通常选 anchor、腿部关键段、脚、手、头。选太少会丢动作细节；选太多会让 reward 变得难优化，而且任何 body 名称不匹配都会直接报错。

### 5.2 Anchor body

K1 使用：

```python
self.commands.motion.anchor_body_name = "Trunk"
```

anchor 是 motion 对齐的核心。训练中的参考 body 会根据机器人当前 anchor 的水平位置/yaw 做相对对齐。选错 anchor，reward 里相对 body 位置和姿态都会变得奇怪。

### 5.3 无状态估计版本

当前默认注册的 `Tracking-Flat-K1-v0` 已经改成使用：

```python
K1FlatWoStateEstimationEnvCfg
```

该配置删除了 policy observation 中的：

```python
motion_anchor_pos_b = None
base_lin_vel = None
```

这里的“无状态估计”不是说机器人完全没有状态，而是说 actor 不依赖那些需要额外状态估计器才能稳定得到的量。它仍然会使用关节角、关节速度、root/base 角速度、姿态相关信息、last action、参考动作命令等更容易在部署侧获得的观测。

这两个被删掉的量分别是：

- `base_lin_vel`：机器人 base/root 在自身坐标系下的线速度。仿真里很好拿，因为模拟器知道真实速度；真机上通常要靠 IMU、编码器、接触、滤波器或状态估计器估出来，会有延迟、噪声、漂移，脚打滑或快速格斗动作时更难准。
- `motion_anchor_pos_b`：参考动作 anchor，比如 `Trunk`，相对当前机器人 anchor 的位置误差。这个量依赖当前机器人 root/anchor 在世界系的位置。MuJoCo 里可以算，真机上如果没有 mocap、视觉定位或高质量里程计，就很难稳定得到。

所以删掉它们不是因为它们“没用”，而是因为它们“仿真里有用，但部署侧不够可靠”。如果训练时 actor 看到了这些量，部署侧却只能给一个有坐标系误差、延迟或噪声的版本，policy 就会把错误状态当真，做出错误补偿。格斗动作幅度大、速度快，一点错误补偿就可能导致摔倒。

部署侧并不是绝对拿不到这两个量。`base_lin_vel` 可以估，`motion_anchor_pos_b` 在 MuJoCo 里也能算；问题是它们很难保证和 IsaacLab 训练时同语义、同坐标系、同延迟、同滤波。为了 sim2sim/sim2real 更稳，actor 宁愿少看一点，也要看得更真实。

这个改动也和 `booster_deploy` 的 beyond_mimic observation 对齐。`booster_deploy` 里实际保留的是：

```text
command
motion_anchor_ori_b
root_ang_vel_b
joint_pos
joint_vel
last_action
```

而线性状态被注释掉：

```python
# motion_anchor_pos_b
# root_lin_vel_b
```

所以训练侧删掉 `motion_anchor_pos_b` 和 `base_lin_vel`，本质上是为了让 IsaacLab actor observation 和 `booster_deploy` 的实际部署输入保持一致。可以理解为参考了部署侧/官方框架的输入约定，而不是随便删的。critic 仍然可以保留 privileged observation，actor 则尽量贴近部署可观测量。

这里是这次很重要的疑惑点：  
“训练时能不能把 base 线速度、anchor 位置都喂给 policy？”  
答案是仿真里当然能，但如果目标是导出到 Booster deploy 或 MuJoCo/实机，policy observation 必须和部署侧能构造出的 observation 对齐。否则 play 时看起来好，部署时就缺输入。

### 5.4 Low frequency 版本

注册了：

```python
Tracking-Flat-K1-Low-Freq-v0
```

它继承自无状态估计版本，并调整：

```python
self.decimation = round(self.decimation / LOW_FREQ_SCALE)
self.rewards.action_rate_l2.weight *= LOW_FREQ_SCALE
```

PPO 配置里也同步调整：

```python
LOW_FREQ_SCALE = 0.5
self.num_steps_per_env = round(self.num_steps_per_env * LOW_FREQ_SCALE)
self.algorithm.gamma = self.algorithm.gamma ** (1 / LOW_FREQ_SCALE)
self.algorithm.lam = self.algorithm.lam ** (1 / LOW_FREQ_SCALE)
```

这里的要点是：只改控制频率不够，rollout 长度、折扣因子和 GAE lambda 也要按时间尺度修正，否则同样的物理时间会对应不同的 RL horizon。

## 6. Reward、Termination 和随机化

基础环境在 `tracking_env_cfg.py`。

### 6.1 Reward

主要 reward 项：

- `motion_global_anchor_pos`
- `motion_global_anchor_ori`
- `motion_body_pos`
- `motion_body_ori`
- `motion_body_lin_vel`
- `motion_body_ang_vel`
- `action_rate_l2`
- `joint_limit`
- `undesired_contacts`

这套 reward 既看全局 anchor，也看相对 body，还看速度。格斗动作速度变化大，只看位置会导致动作“摆到了但节奏不对”；加入速度项能约束动作时序。

### 6.2 Termination

主要 termination：

- `anchor_pos`
- `anchor_ori`
- `ee_body_pos`

K1 环境里把 `ee_body_pos` 的 body 名称换成 K1 的脚和手：

```python
left_foot_link
right_foot_link
left_hand_link
right_hand_link
```

这个坑很常见：从 G1 复制环境后，如果 termination 还在找 `left_ankle_roll_link` 之类的 G1 名称，要么报错，要么 termination 逻辑完全不对。

### 6.3 不期望接触

K1 中允许脚和手接触，其它 body 接触会被惩罚：

```python
r"^(?!left_foot_link$)(?!right_foot_link$)(?!left_hand_link$)(?!right_hand_link$).+$"
```

格斗动作里手可能触地或打击动作接近地面，所以把手也放进允许接触集合是合理的。否则 policy 会为了躲接触牺牲动作模仿。

### 6.4 随机化

基础随机化包括：

- 摩擦随机化：静摩擦、动摩擦、restitution
- 默认关节位置随机扰动
- base COM 随机化
- interval push

K1 配置里把 base COM 的 body 改成：

```python
SceneEntityCfg("robot", body_names="Trunk")
```

这里也是从 humanoid/G1 迁移到 K1 的典型坑。原始配置里可能是 `torso_link`，但 K1 的根 body 叫 `Trunk`。

## 7. 任务注册

K1 注册在 `config/k1/__init__.py`：

```python
Tracking-Flat-K1-v0
Tracking-Flat-K1-Wo-State-Estimation-v0
Tracking-Flat-K1-Low-Freq-v0
```

当前默认的 `Tracking-Flat-K1-v0` 指向无状态估计版本，这样训练、play 和导出时更接近部署输入。

使用任务名前要保证：

```python
import whole_body_tracking.tasks
```

训练和 play 脚本里已经有这行导入，用来触发包扫描和 gym 注册。

## 8. 训练流程

训练脚本使用 wandb registry 中的 motion artifact：

```bash
python scripts/rsl_rl/train.py \
    --task Tracking-Flat-K1-v0 \
    --registry_name <wandb-registry-path>:latest \
    --num_envs 4096 \
    --max_iterations 30000
```

如果 registry_name 不写 alias，脚本会自动补 `:latest`。

训练脚本做的关键事情：

1. 解析 task 和 PPO 配置。
2. 从 wandb registry 下载 motion artifact。
3. 设置 `env_cfg.commands.motion.motion_file`。
4. 创建 IsaacLab gym env。
5. 使用 `MotionOnPolicyRunner` 训练。
6. 保存 env/agent 配置和 git 状态。

K1 PPO 默认：

```python
num_steps_per_env = 24
max_iterations = 30000
save_interval = 500
experiment_name = "k1_flat"
empirical_normalization = True
actor_hidden_dims = [512, 256, 128]
critic_hidden_dims = [512, 256, 128]
```

经验上，全身模仿任务对 normalization 很敏感，`empirical_normalization=True` 是必要配置。动作数据分布、速度项和 body pose 项尺度差别很大，不做 normalization 会让 PPO 学得很不稳定。

## 9. 播放和导出

普通播放：

```bash
python scripts/rsl_rl/play.py \
    --task Tracking-Flat-K1-v0 \
    --num_envs 1 \
    --load_run <run-dir> \
    --checkpoint model_XXXXX.pt \
    --motion_file motions/k1/135_06_stageii.npz
```

从 wandb run 播放：

```bash
python scripts/rsl_rl/play.py \
    --task Tracking-Flat-K1-v0 \
    --num_envs 1 \
    --wandb_path <entity/project/run>
```

`play.py` 已经支持 `--motion_file` 覆盖 motion 文件。这里解决的一个实际问题是：有时候 checkpoint 来自某个 run，但你想临时换一个本地 motion 做测试。如果不加这个参数，就只能依赖 run 的 artifact。

K1 专用导出脚本：

```bash
python scripts/rsl_rl/play_k1.py \
    --task Tracking-Flat-K1-v0 \
    --num_envs 1 \
    --load_run <run-dir> \
    --checkpoint model_XXXXX.pt \
    --motion_file motions/k1/135_06_stageii.npz \
    --jit_filename policy.pt
```

`play_k1.py` 的重点是导出：

- TorchScript：`policy.pt`
- ONNX：`policy.onnx`

并给 ONNX 附加 metadata。

实际接入 `booster_deploy` 时，当前 beyond_mimic 任务用的是 TorchScript `.pt` 和 motion `.npz`，不是本仓库里单独的 MuJoCo ONNX 脚本。也就是说：

- 训练仓库负责训练、回放、导出 `policy.pt`。
- `booster_deploy` 负责把 `policy.pt`、部署用 motion `.npz`、K1 机器人配置和 MuJoCo/实机 controller 接起来。
- ONNX 仍然适合做调试、metadata 检查或其它部署链路，但当前 K1 fight sim2sim 的主路径是 `booster_deploy`。

## 10. ONNX 导出设计

`exporter.py` 里的 `_OnnxMotionPolicyExporter` 不只是导出 policy action，还把参考 motion 也作为 ONNX 输出：

```text
actions
joint_pos
joint_vel
body_pos_w
body_quat_w
body_lin_vel_w
body_ang_vel_w
```

输入是：

```text
obs
time_step
```

这样部署/sim2sim 侧可以用同一个 ONNX：

1. 给定当前 observation 和 time_step。
2. 得到 policy action。
3. 同时得到该 time_step 的参考动作帧。
4. 用参考动作帧构造下一步 observation 或做可视化/调试。

metadata 包括：

- `joint_names`
- `joint_stiffness`
- `joint_damping`
- `default_joint_pos`
- `command_names`
- `observation_names`
- `observation_history_lengths`
- `action_scale`
- `anchor_body_name`
- `body_names`
- `run_path`

这里有一个非常关键的部署坑：ONNX 只有网络权重是不够的。部署侧还需要知道关节顺序、action scale、默认关节角、PD 参数、observation 顺序和 tracking body 名称。metadata 就是为了解决“策略文件能跑，但输入输出语义对不上”的问题。

## 11. MuJoCo sim2sim

这部分不在本仓库里跑，而是在另一个工程：

```bash
cd ~/RL_project/booster_deploy
```

`booster_deploy` 是 Booster 的轻量部署框架，统一支持：

- MuJoCo sim2sim
- Webots/internal sim2sim
- 真机 sim2real

K1 格斗动作模仿对应的是 `tasks/beyond_mimic`。相关文件：

- `tasks/beyond_mimic/beyond_mimic.py`
- `tasks/beyond_mimic/__init__.py`
- `booster_deploy/robots/booster.py`
- `booster_deploy/controllers/mujoco_controller.py`
- `scripts/deploy.py`

### 11.1 任务注册和运行方式

`tasks/beyond_mimic/__init__.py` 里注册了几个 K1 任务：

```python
register_task("k1_mj2", K1MJ2ControllerCfg())
register_task("k1_fight", K1FightControllerCfg())
register_task("k1_135_06_stageii", K113506StageIIControllerCfg())
```

查看可用任务：

```bash
cd ~/RL_project/booster_deploy
python scripts/deploy.py --list
```

运行 K1 fight 的 MuJoCo sim2sim：

```bash
cd ~/RL_project/booster_deploy
python scripts/deploy.py --task k1_fight --mujoco
```

运行 `135_06_stageii` 这个动作：

```bash
cd ~/RL_project/booster_deploy
python scripts/deploy.py --task k1_135_06_stageii --mujoco
```

如果要跑 GPU 上的策略推理，可以加：

```bash
python scripts/deploy.py --task k1_fight --mujoco --device cuda
```

### 11.2 模型和动作文件放置

`booster_deploy` 的 beyond_mimic 任务默认从任务目录下读取模型和动作：

```text
~/RL_project/booster_deploy/tasks/beyond_mimic/
├─ models/
│  ├─ k1_fight_001.pt
│  └─ 135_06_stageii.pt
└─ motions/
   ├─ k1_fight_final_deploy.npz
   └─ 135_06_stageii.npz
```

例如 `K1FightControllerCfg` 中：

```python
self.policy.motion_path = "motions/k1_fight_final_deploy.npz"
self.policy.checkpoint_path = "models/k1_fight_001.pt"
```

所以从本仓库训练完后，需要把导出的 TorchScript 策略和部署用 motion 放到 `booster_deploy/tasks/beyond_mimic/` 对应目录，再在 `__init__.py` 里确认 task 指向正确文件。

### 11.3 beyond_mimic policy 怎么构造输入

`BeyondMimicPolicy` 会加载：

```python
torch.jit.load(f"{self.task_path}/{self.cfg.checkpoint_path}")
MotionLoader(motion_file=f"{self.task_path}/{self.cfg.motion_path}", ...)
```

它的 observation 和当前无状态估计训练版本一致，主要包括：

- 参考命令：`cmd_dof_pos` 和 `cmd_dof_vel`
- 参考 root/anchor 姿态：`motion_anchor_ori_b`
- 当前 root 角速度：`root_ang_vel_b`
- 当前关节位置偏差：`joint_pos`
- 当前关节速度：`joint_vel`
- 上一次 action：`last_action`

代码里故意没有把线性状态放进 actor 输入：

```python
# motion_anchor_pos_b    # linear states
# root_lin_vel_b         # linear states
```

这和训练里的 `K1FlatWoStateEstimationEnvCfg` 对应。也就是说，训练时移除 `motion_anchor_pos_b` 和 `base_lin_vel`，是为了和 deploy 侧实际构造的 observation 对齐。

### 11.4 action scale 和 PD 控制

deploy 侧不是直接把 policy action 当关节目标，而是先按 K1 参数缩放：

```python
self.action_scale = 0.25 * self.robot.effort_limit / self.robot.joint_stiffness
target = action * action_scale + default_joint_pos
```

MuJoCo controller 再用 PD 算力矩，并按 effort limit 裁剪：

```python
self.mj_data.ctrl = np.clip(
    kp * (dof_targets - dof_pos) - kd * dof_vel,
    -ctrl_limit,
    ctrl_limit,
)
```

这里要注意一个细节：训练仓库里的 `K1_ACTION_SCALE` 也是 `0.25 * effort / stiffness`，deploy 侧也按同样公式重算。两边的 `effort_limit` 和 `joint_stiffness` 如果不一致，policy 输出同样的 action，实际目标关节角就会不一致。

### 11.5 IsaacLab 和 MuJoCo 配置对齐

sim2sim 最重要的是让 IsaacLab 训练环境和 `booster_deploy` 的 MuJoCo controller 在语义上对齐，而不是只看“都能跑起来”。

#### 11.5.1 控制频率和仿真步长

训练侧 K1 默认：

```python
self.sim.dt = 0.005
self.decimation = 4
```

所以 policy 频率是：

```text
1 / (0.005 * 4) = 50 Hz
```

`booster_deploy` 默认：

```python
policy_dt = 0.02
mujoco.decimation = 10
mujoco.physics_dt = policy_dt / decimation
```

所以 MuJoCo 是：

```text
policy_dt = 0.02 s = 50 Hz
physics_dt = 0.002 s
```

这意味着两边 policy 控制频率都是 50 Hz，但 physics dt 不同。这个是可以接受的，关键是 policy 每 0.02 秒更新一次，motion NPZ 也应该按 50 FPS 输出。

对齐检查：

- `csv_to_npz_k1.py --output_fps 50`
- IsaacLab `sim.dt * decimation = 0.02`
- `booster_deploy` `policy_dt = 0.02`
- MuJoCo `physics_dt * decimation = 0.02`

#### 11.5.2 joint 顺序对齐

`booster_deploy/robots/booster.py` 里同时维护了两套顺序：

- `joint_names`：真实机器人/MuJoCo controller 使用的顺序。
- `sim_joint_names`：IsaacSim/IsaacLab 使用的顺序。

`BeyondMimicPolicy` 里会用映射处理：

```python
real2sim_map = self.robot.data.real2sim_joint_indexes
sim2real_map = self.robot.data.sim2real_joint_indexes
```

输入 observation 里关节状态要转到 sim 顺序，输出目标再转回 real/MuJoCo 顺序。这个映射是 K1 部署里最容易出错的地方之一。

对齐检查：

- 本仓库 NPZ 中的 `joint_names` 应该对应 IsaacLab articulation 顺序。
- `booster_deploy` 的 `sim_joint_names` 应该与训练侧顺序一致。
- `joint_names` 应该与 MuJoCo MJCF/真机 SDK 顺序一致。
- 如果左右腿、手臂动作错位，第一优先检查这个映射。

#### 11.5.3 body 名称和 anchor 对齐

训练侧 K1 使用：

```python
anchor_body_name = "Trunk"
```

deploy 侧 `BeyondMimicPolicyCfg` 也默认：

```python
anchor_body_name = "Trunk"
```

`booster_deploy/robots/booster.py` 里也维护了：

- `body_names`
- `sim_body_names`

motion loader 会按 `sim_body_names` 读取 IsaacLab 生成的 NPZ，再按部署侧需要的 body 名称做追踪。这里如果 `Trunk`、`left_foot_link`、`right_foot_link`、`left_hand_link`、`right_hand_link` 等名字对不上，参考 ghost 和实际机器人会明显错位。

#### 11.5.4 root pose、四元数和速度坐标系

IsaacLab 和 MuJoCo 都按 root qpos：

```text
root_pos_xyz + root_quat_wxyz + joint_pos
```

所以进入 NPZ 后，四元数必须已经是 `wxyz`。deploy 侧的 `MujocoController` 读取：

```python
base_pos_w = qpos[:3]
base_quat = qpos[3:7]
base_lin_vel_b = qvel[:3]
base_ang_vel_b = qvel[3:6]
```

训练侧 policy 当前不使用 root linear velocity，但使用 root angular velocity。要重点确认角速度都是 base/body frame 下的语义。如果出现上身姿态跟随慢、转身方向异常，要检查 `root_ang_vel_b` 和 IsaacLab 的 `base_ang_vel` 是否同义。

#### 11.5.5 default joint pose、stiffness、damping、effort 对齐

训练侧默认关节角在 `robots/k1.py`：

```python
"Left_Shoulder_Roll": -1.3
"Right_Shoulder_Roll": 1.3
```

deploy 侧 `K1_CFG.default_joint_pos` 也有对应肩膀默认角。这个必须一致，因为 policy 输出的是相对 default pose 的 action target。

对齐检查：

- `default_joint_pos` 一致。
- `joint_stiffness` 一致或明确知道 deploy 侧做了调整。
- `joint_damping` 一致或明确知道 deploy 侧做了调整。
- `effort_limit` 一致或明确知道 deploy 侧为了安全做了限幅。
- action scale 使用同一个公式。

当前 `k1_fight` task 里为了安全和效果，手臂/头/腿的 stiffness、damping、effort limit 有单独覆盖；`k1_135_06_stageii` 更接近训练侧由电机参数推导出的数值。调试时要明确自己跑的是哪个 task。

#### 11.5.6 初始高度和初始姿态对齐

训练侧 K1 初始高度：

```python
pos=(0.0, 0.0, 0.57)
```

deploy MuJoCo 配置：

```python
mujoco = MujocoControllerCfg(init_pos=[0.0, 0.0, 0.57])
```

这两个要一致。初始高度差一点，接触状态就会不同，尤其格斗动作有快速抬腿、落脚、手部接触时，MuJoCo 里可能一开始就出现不必要的冲击。

#### 11.5.7 MuJoCo 资产路径

`booster_deploy/robots/booster.py` 中 K1 MJCF：

```python
mjcf_path="{BOOSTER_ASSETS_DIR}/robots/K1/K1_22dof.xml"
```

`MujocoController` 会用 `booster_assets.BOOSTER_ASSETS_DIR` 展开这个占位符。要确保安装/配置的 `booster_assets` 版本和训练侧使用的 K1 URDF 是同一版机器人模型，否则 body/joint 名称也许能对上，但惯量、碰撞、关节限位和电机能力可能已经不一致。

#### 11.5.8 reference ghost

deploy 侧 MuJoCo controller 支持参考 ghost：

```python
visualize_reference_ghost=True
```

`BeyondMimicPolicy` 每一步会把参考动作写给 controller：

```python
self.controller.set_reference_qpos(ref_qpos)
```

这个 ghost 非常适合排查：

- 参考动作本身是否正确。
- policy 跟踪误差主要出在 root、腿、手还是头。
- 是动作数据错了，还是控制/动力学跟不上。

### 11.6 sim2sim 主要排查什么

这个 sim2sim 环节主要用来发现 IsaacLab 和 MuJoCo/部署侧的差异，比如：

- joint 名称和顺序是否一致。
- root quaternion 顺序是否一致。
- observation 顺序是否一致。
- policy Hz 是否一致。
- PD stiffness/damping/action scale 是否一致。
- anchor/body 名称在 MJCF 中是否存在。
- motion FPS 是否等于 policy FPS。
- deploy 侧是否用了和训练侧一致的无状态估计 observation。

经验上，如果 IsaacLab play 正常、MuJoCo sim2sim 异常，优先按这个顺序排查：joint 映射、observation 维度和顺序、action scale、PD 参数、motion FPS、root quaternion。

### 11.7 一次摔倒问题的复盘：从手写 sim2sim 到 booster_deploy

之前用自己写的 `sim2sim_mujoco_k1` 跑 MuJoCo 时出现过摔倒，后来主要做了三类改动：

1. 训练和部署都删掉了 `motion_anchor_pos_b`、`base_lin_vel` 这两个 policy observation。
2. K1 训练侧从普通 `ImplicitActuatorCfg` 换成了 `BoosterDelayedPDActuatorCfg`。
3. sim2sim 不再走手写 MuJoCo 脚本，而是走官方/统一的 `~/RL_project/booster_deploy`。

从现在代码和旧脚本对比看，摔倒大概率不是单一 bug，而是几类不一致叠加导致的。

#### 11.7.1 observation 不一致或线性状态难以对齐

旧的手写 MuJoCo sim2sim 里，observation 是手动拼出来的：

```python
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
)
```

这里包含两个后来删掉的线性状态：

- `motion_anchor_pos_b`
- `base_lin_vel`

这两个量在 IsaacLab 里很好拿，但在 MuJoCo/实机部署侧非常容易出现语义差异：

- root position 的参考系是否完全一致。
- MuJoCo `qvel` 的线速度/角速度到底按哪个 frame 理解。
- anchor body 和 root/free joint 的位置是否完全等价。
- motion 开始帧和当前 MuJoCo 初始位姿有没有做一致的对齐。
- 手写脚本里的 yaw 对齐、frame transform 和训练环境是否完全一致。

如果这些线性量有一点偏差，policy 会把偏差当成真实状态误差，然后给出比较激进的修正动作。格斗动作本来就快，修正一激进就容易把机器人打翻。

当前训练配置里专门定义了无状态估计版本：

```python
self.observations.policy.motion_anchor_pos_b = None
self.observations.policy.base_lin_vel = None
```

`booster_deploy` 的 `BeyondMimicPolicy` 也对应地把这两个量注释掉：

```python
# motion_anchor_pos_b    # linear states
# self.robot.data.root_lin_vel_b           # linear states
```

保留下来的 actor 输入主要是：

- 参考关节位置/速度。
- anchor/root 姿态误差。
- root 角速度。
- 当前关节位置/速度。
- last action。

这套 observation 对 MuJoCo 和真机都更容易稳定构造，减少了“训练能用、部署侧很难精确复现”的线性状态依赖。

这件事的经验是：全身模仿里，不是 observation 越多越好。只要某个 observation 在部署侧有坐标系、延迟、估计误差或初始对齐问题，它对 actor 就可能是毒药。可以给 critic 更多 privileged 信息，但 actor 最好只吃部署侧能稳定构造的量。

#### 11.7.2 执行器动力学从理想 PD 变成更接近真实 K1

旧配置用 `ImplicitActuatorCfg` 时，训练侧更接近理想 PD。MuJoCo 侧再用另一个 PD controller 去追 target，二者之间会有明显动力学差异：

- 训练侧没有显式控制延迟，部署/MuJoCo 有控制周期和执行延迟。
- 训练侧如果只按固定 effort limit，不能体现电机高速时力矩下降。
- K1 不同关节电机能力差别很大，手臂、膝盖、髋关节、脚踝不能用一套泛化参数糊过去。
- K1 脚踝还有并联机构等效参数问题。

后来换成 `BoosterDelayedPDActuatorCfg` 后，训练侧开始显式建模：

- `min_delay=2, max_delay=8` 的命令延迟。
- Booster 电机型号对应的 effort、velocity、armature。
- 速度相关力矩裁剪。
- K1 ankle 的 `BoosterK1AnkleParaWrapperCfg`。

这会让训练时的策略更早感受到“真实执行器没那么理想”。它可能在 IsaacLab 里看起来没有理想 PD 那么猛，但迁移到 MuJoCo/部署侧时会稳很多。

这件事的经验是：sim2sim 摔倒时，不能只查 observation。全身动作模仿对执行器模型非常敏感，特别是格斗动作这种高速度、大加速度动作。如果训练侧 actuator 太理想，policy 很容易学出 MuJoCo 或真机跟不上的 target。

#### 11.7.3 手写 sim2sim 太容易漏掉隐藏约定

旧手写脚本需要自己维护所有部署约定：

- 从 ONNX metadata 读 joint names、body names、default pose、action scale、PD 参数。
- 手动构造 observation 顺序。
- 手动处理 root quaternion、body frame transform。
- 手动设置 MuJoCo 初始 qpos/qvel。
- 手动做 PD 控制和 effort clip。
- 手动保证 policy Hz、MuJoCo timestep、motion FPS 一致。

这些每一项看起来都不难，但只要有一项和训练环境差一点，就可能摔。

`booster_deploy` 解决的是工程一致性问题。它把 K1 的部署约定集中到了：

- `booster_deploy/robots/booster.py`
- `tasks/beyond_mimic/beyond_mimic.py`
- `tasks/beyond_mimic/__init__.py`
- `booster_deploy/controllers/mujoco_controller.py`

里面明确区分了：

- `joint_names`：真机/MuJoCo controller 顺序。
- `sim_joint_names`：IsaacLab/IsaacSim 顺序。
- `body_names` 和 `sim_body_names`。
- `real2sim_joint_indexes` 和 `sim2real_joint_indexes`。
- `policy_dt=0.02` 和 MuJoCo `decimation=10`。
- `default_joint_pos`、`joint_stiffness`、`joint_damping`、`effort_limit`。

`BeyondMimicPolicy` 还做了 motion 的初始对齐和 root yaw 对齐：

```python
align_to_first_frame=True
self.init_root_yaw_quat_w_inv = quat_inv(yaw_quat(self.robot.data.root_quat_w))
```

这比在一个临时 sim2sim 脚本里手写所有细节更可靠。后来 MuJoCo 不再摔，一个重要原因就是这些隐含约定被统一到了 `booster_deploy` 里，而不是散落在一次性的调试脚本里。

#### 11.7.4 这次问题可以总结成三条经验

第一，训练 actor 的 observation 必须和部署侧 observation 完全一致。  
如果部署侧很难稳定拿到 `motion_anchor_pos_b`、`base_lin_vel`，就不要让 actor 依赖它们。

第二，训练侧 actuator 不能太理想。  
`BoosterDelayedPDActuatorCfg` 引入延迟、速度相关力矩裁剪和 K1 电机参数后，策略更容易迁移到 MuJoCo/真机。

第三，sim2sim 最好走统一部署框架。  
手写脚本适合快速验证 ONNX，但不适合作为最终判断标准。真正要排查 sim2real/sim2sim，应该尽量用和实机同一套 joint order、default pose、PD 参数、action scale 和 observation 构造逻辑，也就是现在的 `booster_deploy`。

所以这次从“手写 MuJoCo sim2sim 摔倒”到“`booster_deploy` MuJoCo 正常”的关键，不是某个单点魔法改动，而是把训练侧和部署侧的 observation、执行器动力学、控制器约定统一了。

## 12. 这次踩过/容易踩的坑

### 12.1 关节顺序不能靠猜

CSV 的 22 个关节必须和 K1 MJCF motor 顺序一致。`csv_to_npz_k1.py` 从 `K1_serial.xml` 的 motor 标签读取 joint order，这比手写列表可靠。

错误表现：

- 左右腿动作错位。
- 膝盖/脚踝角度不合理。
- replay 时身体姿态看似有动作，但四肢明显乱。

处理方法：

- 打印 `K1 joint order`。
- 对照 `robot.data.joint_names`。
- replay NPZ，先肉眼检查再训练。

### 12.2 四元数 `xyzw` 和 `wxyz`

CSV 输入是 `xyzw`，IsaacLab 用 `wxyz`。如果忘记转换，root 姿态会错。

处理方法：

- CSV 转 NPZ 阶段统一转换。
- 后续 NPZ、IsaacLab、ONNX 内部都按 `wxyz` 理解。
- MuJoCo 里 `data.qpos[3:7]` 也按 `wxyz` 写入。

### 12.3 K1 body 名称和 G1/humanoid 不一样

从其它机器人迁移配置时，最容易遗漏：

- anchor body
- COM randomization body
- end-effector body
- undesired contact allowlist
- tracking body list

K1 里根 body 是 `Trunk`，脚是 `left_foot_link/right_foot_link`，手是 `left_hand_link/right_hand_link`。不能继续用 G1 的 `torso_link`、`left_ankle_roll_link` 等名字。

### 12.4 训练输入和部署输入要一致

仿真里能拿到 base linear velocity、motion anchor relative position，但部署侧未必可靠。所以当前默认任务使用无状态估计版本，把这些 policy observation 移除。

疑惑点可以总结成一句话：  
“训练时多给信息会不会更好？”  
短期会更好，长期可能更糟。只要最终要部署，actor observation 就应尽量使用部署侧能稳定构造的信息；extra 信息可以留给 critic。

### 12.5 Action scale 不是越大越好

格斗动作幅度大，很容易想把 action scale 调大。但 K1 是 position target policy，action scale 过大时，PD 目标跳变会引起：

- action rate 惩罚变大
- 关节限位惩罚变大
- 接触不稳定
- sim2sim 中 PD 扭矩饱和

当前用 `0.25 * effort / stiffness` 是保守做法。

### 12.6 真实执行器不是理想 PD

K1 当前引入：

- command delay
- speed-dependent torque clipping
- motor-specific effort/velocity/armature
- ankle parallel wrapper

这些都会降低策略在仿真里的“理想表现”，但提升仿真和真实/部署侧的一致性。

### 12.7 低频控制需要连 PPO 参数一起改

如果只改 `decimation`，不改 `gamma`、`lam`、`num_steps_per_env`，训练的时间尺度就变了。Low-freq 版本里同步做了这些处理。

### 12.8 Motion artifact 和本地 motion 的优先级

训练脚本默认从 wandb registry 下载 motion。play/export 时又可能从 wandb run 使用当时的 artifact。为了方便调试，`play.py`/`play_k1.py` 增加了 `--motion_file` 覆盖。

经验：复现实验时优先用 run 对应的 artifact；排查动作问题时可以用 `--motion_file` 强制指定本地 NPZ。

### 12.9 ONNX metadata 和 deploy 配置都很重要

如果只拿到一个策略文件，部署侧只能看到一串 action，无法可靠知道：

- action 对应哪个关节。
- action scale 是多少。
- 默认关节角是多少。
- observation 顺序是什么。
- PD 参数是什么。

ONNX metadata 是一种解决方式；当前 `booster_deploy` 的 beyond_mimic 路径则把这些信息放在 `booster_deploy/robots/booster.py` 和 `tasks/beyond_mimic/__init__.py` 里。无论用哪种方式，本质都是同一个要求：策略权重、关节顺序、默认关节角、PD 参数、action scale、observation 顺序必须成套保存和成套使用。

### 12.10 硬编码路径影响复现

当前若干默认路径是本机绝对路径，例如：

- `/home/liyunsong/RL_project/booster_assets/robots/K1/K1_22dof.urdf`
- `/home/liyunsong/RL_project/GMR/assets/booster_k1`
- `/home/liyunsong/RL_project/GMR/general_motion_retargeting/ik_configs/smplx_to_k1.json`

这在本机开发最快，但交给别人时会成为第一道门槛。后续建议改成环境变量或命令行参数，并在 README 中说明。

## 13. 推荐复现顺序

1. 确认 K1 URDF、MJCF、IK config 路径可用。
2. 准备 GMR 输出 CSV。
3. 用 `csv_to_npz_k1.py` 转成 NPZ。
4. 用 `replay_npz_k1.py` 检查动作质量。
5. 上传 NPZ 到 wandb registry，或准备本地 motion path。
6. 用 `Tracking-Flat-K1-v0` 训练。
7. 用 `play.py` 或 `play_k1.py` 回放 checkpoint。
8. 导出 `policy.onnx` 和 `policy.pt`。
9. 如果走 ONNX 链路，检查 ONNX metadata；如果走 `booster_deploy`，把 `.pt` 和 `.npz` 放到 `tasks/beyond_mimic/` 并检查 task 配置。
10. 在 `~/RL_project/booster_deploy` 里用 MuJoCo sim2sim 做二次验证。

## 14. 常用命令模板

CSV 转 NPZ：

```bash
python scripts/csv_to_npz_k1.py \
    --input_file motions/k1/135_06_stageii.csv \
    --output_name motions/k1/135_06_stageii.npz \
    --input_fps 120 \
    --output_fps 50
```

回放 NPZ：

```bash
python scripts/replay_npz_k1.py \
    --motion_file motions/k1/135_06_stageii.npz
```

训练：

```bash
python scripts/rsl_rl/train.py \
    --task Tracking-Flat-K1-v0 \
    --registry_name <wandb-registry-path>:latest \
    --num_envs 4096 \
    --max_iterations 30000
```

本地 checkpoint 播放：

```bash
python scripts/rsl_rl/play.py \
    --task Tracking-Flat-K1-v0 \
    --num_envs 1 \
    --load_run <run-dir> \
    --checkpoint model_XXXXX.pt \
    --motion_file motions/k1/135_06_stageii.npz
```

导出 K1 策略：

```bash
python scripts/rsl_rl/play_k1.py \
    --task Tracking-Flat-K1-v0 \
    --num_envs 1 \
    --load_run <run-dir> \
    --checkpoint model_XXXXX.pt \
    --motion_file motions/k1/135_06_stageii.npz \
    --jit_filename policy.pt
```

MuJoCo sim2sim：

```bash
cd ~/RL_project/booster_deploy
python scripts/deploy.py --task k1_fight --mujoco
```

运行 `135_06_stageii`：

```bash
cd ~/RL_project/booster_deploy
python scripts/deploy.py --task k1_135_06_stageii --mujoco
```

## 15. 后续可以继续优化的点

1. 把 K1 URDF/MJCF/IK config 的绝对路径改成环境变量或统一配置。
2. 把本仓库导出的 `policy.pt`、motion `.npz` 和 `booster_deploy` 的 task 配置形成固定同步流程，避免手动改错文件名。
3. 给 `csv_to_npz_k1.py` 增加更明确的数据校验，比如 root 高度范围、四元数 norm、关节限位检查。
4. 给 K1 写一份最小 smoke test：能注册 task、能加载 NPZ、能 reset env。
5. 对比有/无状态估计版本的训练曲线，确认无状态估计版本的性能损失是否可接受。
6. 根据真实 Booster K1 控制频率和延迟，进一步校准 `decimation`、`min_delay/max_delay` 和 PD 参数。
7. 对格斗动作单独做 motion curriculum，例如先短片段、慢速、低幅度，再逐步放开。

## 16. 一句话总结

这次 K1 格斗动作模仿学习的核心经验是：动作数据、机器人资产、执行器模型、observation 设计和部署配置必须是一条闭环。只要其中一个环节的关节顺序、body 名称、四元数格式、控制频率或 action scale 对不上，训练表面上可能还能跑，但回放、sim2sim 或部署一定会出问题。先把 NPZ replay、策略导出和 `booster_deploy` MuJoCo 对齐这几个检查点做扎实，再谈调 reward 和 PPO，会少走很多弯路。

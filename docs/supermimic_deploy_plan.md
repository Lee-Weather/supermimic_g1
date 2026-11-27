# 把 `deploy/G1_deploy` 改造成 SuperMimic 部署方案

本文是**方案设计**，不是已实现代码。基于对 `deploy/G1_deploy`（Unitree `unitree_rl_lab` + BeyondMimic 部署链路）
与 `supermimic` 策略实际签名的逐项比对，给出改造点、数据契约、分阶段计划与风险。

## 0. 结论先行

- **可复用约 90%**：DDS 通信、FSM（1000 Hz）、ONNX Runtime、`param`/日志、安全退出、离线 MuJoCo 验证框架全部照用。
- **只需重写 3 处**：① 观测构造（gencoord `rep_state` 68 维）；② 目标构造（motion 640 维窗口 / velocity 3 维）；
  ③ 动作解算（`q_def + 0.5·a`）。
- **两个利好**（从 `models/g1_walk_velocity_30k_20261126/actor.pt` 直接读出）：
  1. `decoder.0.weight = (2048, 100) = 32(latent) + 68(rep_state)` → **`history_length = 1`，不需要历史环形缓冲**；
  2. 我们的 policy 关节顺序 **等于 mjlab XML 顺序、等于 SDK 电机顺序**，`sdk_indices` 为**恒等映射**（BeyondMimic 需要重排）。
- **两个部署入口**：`velocity`（摇杆，无参考动作，**优先做**）与 `motion`（回放参考片段，需 `reference.bin`）。

## 1. 现有 BeyondMimic 链路（把它当成模板）

| 层 | 文件 | 职责 | 我们要改吗 |
|---|---|---|---|
| 观测/估计/限幅 | `deploy/include/beyondmimic/core.h` | `Snapshot`、FK、`LegOdometry`、`observe()`、`targets()` | **改** observe/targets，**复用** Snapshot/LegOdometry |
| 包加载+推理 | `deploy/include/beyondmimic/package.h` | `load_config()`（yaml 校验）、`Runner`（ONNX 1→1） | **改** 校验字段，Runner 基本复用 |
| 策略状态 | `deploy/robots/g1_29dof/src/State_BeyondMimic.cpp` | 20 ms 独立线程、参考帧推进、安全退出 | **改** 观测/目标/参考源 |
| 状态机 | `deploy/include/FSM/CtrlFSM.h`、`BaseState.h` | 1 ms 节拍、按键转状态 | 复用 |
| 入口/配置 | `deploy/robots/g1_29dof/main.cpp`、`config/config.yaml`、`param.h` | 参数、`policy_dir`、FSM 表 | 复用 + 新增 FSM 项 |
| 导出/验证 | `tools/export_beyondmimic.py`、`tools/verify_beyondmimic.py`、`deploy/tools/beyondmimic_offline.cpp` | ONNX 打包、64 组对拍、MuJoCo 闭环 | **照抄结构，改数据契约** |

**硬约束（模板已经定死，我们必须遵守）**

- ONNX：**单输入单输出**，输入名必须叫 `obs`，输出 `actions`，形状 `[1,O] → [1,J]`，`float32`；
  `package.h::Runner` 会逐项校验名字/形状/dtype，不匹配直接抛异常。
- 归一化**内嵌进图**（C++ 端输入原始值，不能再归一化一次）。
- 策略周期 **20 ms**；FSM 与 DDS 发送 1000 Hz；推理在独立线程，`targets_` 双缓冲 + 100 ms 超时 → 阻尼。
- 部署端会做关节限位、变化率、力矩包络三重限幅——**这会改变策略行为**，需在训练/验证时预期。

## 2. SuperMimic 策略的真实签名（从权重读出，不是猜）

```
encoders.velocity.0.weight = (2048, 3)      # velocity goal = [vx_b, vy_b, wz_b]
encoders.motion.0.weight   = (2048, 640)    # motion goal = 10 帧 × (29 q + 29 dq) + 10 帧 × 6 (rot6d)
decoder.0.weight           = (2048, 100)    # = latent 32 + rep_state 68  → history_length = 1
MLP: Linear → LayerNorm → GELU（hidden 2048/…），ONNX 完全支持
```

- `rep_state(68) = [z(1), g_b(3), v_b(3), w_b(3), qj(29), vj(29)]`
- 归一化：`obs_normalizer`(68) + `goal_normalizers["motion"]`(640) / `["velocity"]`(3)，导出时冻结内嵌
- 动作：`q_des = q_default + 0.5 · a`（`tracking_env_cfg.py: scale=0.5, use_default_offset=True`）
- 频率 50 Hz（20 ms），回合上限 10 s（500 步），平地、无 DR
- `slot_mix`：`{"motion":0.5,"velocity":0.5}` 训练 → 推理时用**各自任务的 ONNX**（图里只有一条槽位路径）

### 2.1 关节映射：恒等

`datasets/walk_only/npz/*` 的 `joint_names`、`mjlab .../unitree_g1/xmls/g1.xml` 的 joint 定义顺序、
G1 29DoF 的 SDK 电机顺序三者**完全一致**：

```
0-5   left  hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
6-11  right hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
12-14 waist yaw, roll, pitch
15-21 left  shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw
22-28 right 同上
```

→ `sdk_indices = [0,1,…,28]`。**这是与 BeyondMimic 最大的便利差异**（它需要交错重排）。

## 3. 差异对照表

| 项 | BeyondMimic（现成） | SuperMimic（要改） |
|---|---|---|
| actor 输入 | 单向量 160 | `goal`(640 或 3) + `rep_state`(68) 两段 |
| 历史 | 无（观测内含 last_action） | 无（H=1），但有显式 `goal` 段 |
| 参考动作 | `reference.bin`，每帧 `2J+7B`=65 floats（29 q + 29 dq + 7 体 ×7） | 每帧 `2J+3+4`=**65 floats**（29 q + 29 dq + root_pos 3 + root_quat 4），但**不进 obs**，只喂 goal 窗口 |
| anchor | `torso_link`（body 7） | **root（pelvis）** |
| goal 前瞻 | 无（逐帧跟随） | **10 帧 × stride 5 = 0.9 s 前瞻** |
| 动作解算 | `q_def + 0.25·effort/kp · a` | **`q_def + 0.5·a`** |
| 用 `q - default` 做观测 | 是 | **否**：gencoord 用**原始 qj**（最易写错） |
| 归一化 | `(obs-mean)/(std+0.01)` 内嵌 | 同，但 **obs 与 goal 两套**，且按任务选 |
| 状态数 | 1 个（BeyondMimic） | **2 个**（Velocity / Motion），或 1 个类 + `mode` 字段 |

## 4. 改造方案（文件级）

```
deploy/include/supermimic/core.h          # 新增：Snapshot/GenCoordObs/Motion/LegOdometry/Goal/targets
deploy/include/supermimic/package.h       # 新增：load_config + Runner（支持两个 onnx）
deploy/robots/g1_29dof/include/State_SuperMimic.h
deploy/robots/g1_29dof/src/State_SuperMimic.cpp   # 新增：20ms 线程 + velocity/motion 两模式
deploy/tools/supermimic_offline.cpp       # 新增：离线回放（给 MuJoCo 验证用）
deploy/robots/g1_29dof/config/supermimic_velocity.yaml   # 新增：FSM 配置
deploy/robots/g1_29dof/config/policy/supermimic/velocity_v1/
    policy_velocity.onnx  deploy.yaml  manifest.json      # 由导出器生成
tools/export_supermimic.py                # 新增：torch → ONNX + deploy.yaml + manifest
tools/verify_supermimic.py                # 新增：ONNX 对拍 + C++ obs 对拍 + MuJoCo 闭环
```

### 4.1 `deploy.yaml`（新格式 `supermimic_g1_gencoord_v1`）

在 BeyondMimic 版基础上：

- 保留：`joint_names`、`sdk_indices`、`default_joint_pos`、`kp`、`kd`、`effort_limit`、`joint_lower/upper`、
  `kinematics.{torso,left_foot,right_foot}`、`normalizer.epsilon`
- 新增：`action_scale_mode: fixed`、`action_scale: 0.5`、`obs_dim: 68`、`goal_dim: {motion: 640, velocity: 3}`、
  `goal_frames: {count: 10, stride: 5}`、`leg_odometry: true`、`vx_clip: [-0.1, 1.0]`、`vy_clip: [-0.1, 0.1]`、
  `wz_clip: [-1.5, 1.5]`
- 删除：`observation_dim: 160`、`anchor_body_name: torso_link`（我们是 root）、`action_scale` 的 effort/kp 公式

### 4.2 观测构造（C++，每 20 ms）

```cpp
// rep_state(68) —— 注意 qj 用原始值、不减 default
z    = odom.position().z();                       // 骨盆高度
g_b  = R(q).transpose() * [0,0,-1];
v_b  = odom.velocity_b();                         // 腿部里程计（打滑即失效）
w_b  = gyro;                                      // 体坐标系角速度，直接用 IMU
qj   = q_sdk;                                     // 原始关节角
vj   = dq_sdk;
obs  = [z, g_b(3), v_b(3), w_b(3), qj(29), vj(29)];   // 共 68
```

### 4.3 目标构造

- **velocity 模式（无参考）**：`goal = clamp(joystick[vx, vy, wz])`，按上表钳位；摇杆死区 ≥0.05。
- **motion 模式**：维护浮点参考帧 `t`，取 `idx = clamp(t + k*5, clip_end)` for k=0..9；
  `content = concat(ref.q[idx], ref.dq[idx])`（580 维）；
  `rel_ori = rot6d(R_root_nowᵀ · R_root_ref[idx])`（60 维）；`goal = [content, rel_ori]`（640 维）。
  `rel_pos` **只用于世界模型，不进 actor**。

### 4.4 动作解算

```cpp
q_des[i] = default_q[i] + 0.5f * action[i];        // 训练时的 action scale
clip(q_des, lower+0.01, upper-0.01);
rate_limit(q_des, last_q_des, 8.0f * dt);          // 8 rad/s
torque_envelope(q_des, q, dq, kp, kd, effort);      // 与 BeyondMimic 相同
```

## 5. 分阶段计划与验收

| 阶段 | 内容 | 验收 | 估计 |
|---|---|---|---|
| **0 导出** | `tools/export_supermimic.py` 张出 `velocity_v1`（ONNX + yaml + manifest） | Python 端 64 组随机采样对拍，`max_abs_err ≤ 2e-4`；ONNX 元数据含 checkpoint sha256 | 0.5 天 |
| **1 C++ 对拍** | `supermimic/core.h` + `package.h` + `supermimic_offline.cpp` | 用仿真录制的 `(obs, action)` 序列逐帧比对，`≤1e-6`；**不需要机器人** | 1 天 |
| **2 velocity 闭环** | `State_SuperMimic`（velocity 模式）+ FSM 配置 | MuJoCo（mjlab `g1.xml`）里摇杆给小速度，能站住并跟速 ±20% | 0.5 天 |
| **3 motion 闭环** | motion 模式 + `reference.bin` 生成 | MuJoCo 里回放 3 条片段（直行/转向/起步-停止）不摔 | 1 天 |
| **4 真机吊装** | 吊装 + FixStand → SuperMimic_Velocity → 小速度 | 0.2→0.5→0.8 m/s 逐级；任一异常按 `LT+B` 回 Passive | 1–2 天 |

编译：`tools/build_g1_deploy.sh` 原样可用；aarch64 需要 ONNX Runtime ARM64（脚本会自动下载）。

## 6. 风险清单（含我们已有实测证据）

| # | 风险 | 证据 | 对策 |
|---|---|---|---|
| 1 | **`v_b` 估计**（最大风险） | 策略速度任务是体坐标系速度闭环；真机只能用腿部里程计 | 训练时给 `v_b` 加噪声/延迟 DR；真机上辅以 EKF；先用小速度试 |
| 2 | **后退发散** | 实测指令 −0.3 → 实际 **0.83 m/s**（5 s 走 4.06 m） | 部署端 `vx_clip` 下限设 **−0.1**；或直接禁用后退 |
| 3 | **侧移不对称** | 左 0.15 / 右 0.10（指令 ±0.2） | `vy_clip` 设 ±0.1 或 0；真机不依赖侧移 |
| 4 | H=1 无历史 | `decoder.0` in=100 | 输出加一阶低通/斜率限制（≤8 rad/s），首帧从当前姿态热启动 |
| 5 | goal 时序错帧 | 640 维 = 10 帧 × 0.1 s 间隔、0.9 s 前瞻 | 帧索引公式写死在 core.h，用离线对拍锁死 |
| 6 | 观测忘了减/多减 default | 我们是**不减** default（与 BeyondMimic 相反） | 单元测试里放一条固定向量做 golden |
| 7 | 无地形/DR 训练 | 训练 cfg 平地 | 只在平整地面用；上坡/地毯先别试 |
| 8 | 部署限幅改变行为 | BeyondMimic 文档明确警告 | 限幅参数写进 deploy.yaml，训练期就按同参数 clip |

## 7. 现在就能做的第一步

`tools/export_supermimic.py` 是纯 Python，不需要机器人也不需要 C++：

```bash
python tools/export_supermimic.py \
  --repo E:/G1/G1_supermimic \
  --model-dir models/g1_walk_velocity_30k_20261126 \
  --task velocity \
  --output deploy/G1_deploy/deploy/robots/g1_29dof/config/policy/supermimic/velocity_v1
```

它要做的事：加载 `world_model.pt`/`actor.pt` → 重建 `MultiEncoderAE` → 冻结两个 normalizer →
把 `goal` 与 `rep_state` 拼成单输入 `obs`（`[1, 68+640]` 或 `[1, 68+3]`）→ 导出 ONNX →
64 组对拍 → 写 `deploy.yaml`（含恒等 `sdk_indices`、`kp/kd/effort/limits`、`kinematics`）与 `manifest.json`。

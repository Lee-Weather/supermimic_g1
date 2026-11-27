# 把 SuperMimic 接到 `G1_deploy`：实现状态

配套文档：`docs/supermimic_deploy_plan.md`（改造方案与风险清单）。本文件只记录
**已经做完、验证到什么程度、还缺什么**。

## 1. 已完成

### 1.1 权重导出（无需 torch）

`tools/export_supermimic.py` + `tools/supermimic_ckpt.py`

- 直接从 `torch.save` 的 zip 里用 numpy 还原 66 个张量（本机 Python 3.11 没有 torch）；
- 按 `MultiEncoderAE.forward` 重建计算图并写成 ONNX：
  `goal → 归一化 → encoder(Linear/LayerNorm/GELU ×4) → tanh → latent(32)`，
  `[latent, rep_state_norm] → decoder → action(29)`；
- 两个 `EmpiricalNormalization` 折进图内，公式 **`(x - mean) / (std + 0.01)`**
  （从 `rsl_rl 5.5.2` 源码确认，`eps` 默认 `1e-2`）；
- 接口：单输入 `obs` `[1,71]`（`[vx,vy,wz]` + 68 维 `rep_state`）→ 单输出 `actions` `[1,29]`，
  与 `beyondmimic::Runner` 的校验逻辑完全一致，所以 C++ 侧只换常量不改架构。

**验证**：64 组随机采样，`max_abs_err = 3.5e-06`（numpy 参考实现 vs onnxruntime）。
产物 `models/deploy_supermimic/v1/`（与行走控制器 `config/policy/velocity/v0` 同构）：
`exported/policy.onnx`（23 MB）、`params/deploy.yaml`、`params/manifest.json`。
真机上的同一份：`config/policy/supermimic/v1/`。

`deploy.yaml` 里的机器人参数不是抄的，是按 `g1_constants.py` 的公式重算的
（`tools/g1_robot_constants.py`）：`kp = ARMATURE·(2π·10)²`、`kd = 2·2·ARMATURE·2π·10`，
再按 `_.*_elbow_joint` 等正则分配到 4 种电机；算出来与 BeyondMimic 的部署包
**逐位一致**（40.18 / 99.10 / 28.50 / 14.25 / 16.78，2.558 / 6.309 / 1.814 / 0.907 / 1.068）。

### 1.2 部署端代码

`deploy/supermimic_patch/`（`deploy/G1_deploy` 是独立 git 仓库，那里也已提交 `7e53e2d`）：

| 文件 | 说明 |
|---|---|
| `include/supermimic/core.h` | `rep_state` 68 维观测、速度指令钳位、`q_des = default + scale·a`、关节上下限 + 速率限幅、倾角健康检查 |
| `include/supermimic/package.h` | `deploy.yaml` 校验 + ONNX `Runner`（输入名/输出名/形状/dtype 全校验）+ 速度指令 clip |
| `robots/g1_29dof/src/State_SuperMimic.cpp` | 20 ms 策略线程、1 kHz 发布、100 ms 输出看门狗、超时/倾角 → 阻尼兜底、`REGISTER_FSM` |
| `robots/g1_29dof/config/supermimic_velocity.yaml` | `Passive` / `FixStand` / `SuperMimicVelocity` 三段 FSM |
| `robots/g1_29dof/config/velocity_cmd.yaml` | 速度指令：**摇杆回中时**才用（台架试） |
| `tools/supermimic_offline.cpp` | 无硬件对拍：CSV 快照 → 动作/目标 |

**复用而非重写**的部分：`beyondmimic::Snapshot/Config/Chain/forward_kinematics/LegOdometry`
（同一台机器人、同一套腿链），所以 `core.h` 只做「观测拼装 + 动作映射」，
里程计仍是 BeyondMimic 那套（IMU + 编码器，假设至少一只脚不打滑）。

### 1.3 训练/部署一致性

- **关节顺序**：policy 顺序 = mjlab `g1.xml` joint 顺序 = G1 SDK 电机顺序 →
  `sdk_indices = [0..28]` 恒等（BeyondMimic 需要置换，我们不需要）。
- **action**：`q_des = default_joint_pos + 0.5·a`（`tracking_env_cfg.py` 的
  `scale=0.5, use_default_offset=True`，`default_joint_pos` = `KNEES_BENT_KEYFRAME`）。
- **`qj` 不减 default**（与 BeyondMimic 相反），已用归一化统计验证：
  `obs_normalizer._mean` 的 68 维重排后 `z≈0.772`、`g_b≈(0,0,-0.927)`、`v_b≈(0.308,0,0)`、
  `qj` 膝 0.342 / 踝 -0.122、`vj≈0` —— 与「直立行走」完全吻合，说明布局
  `[z, g_b, v_b, w_b, qj, vj]` 正确（`history_length=1`，没有历史环）。

## 2. 验证到哪一步

| 项目 | 状态 |
|---|---|
| ONNX vs numpy 参考实现 | ✅ `3.5e-06` |
| 观测布局 / 归一化公式 | ✅ 用 checkpoint 统计量反证 |
| C++ 编译 | ❌ 未编译（本机无 g++/Eigen/yaml-cpp/onnxruntime） |
| C++ 与 Python 对拍 | ❌ 未跑（需要先编译 + 录一段快照 CSV） |
| 闭环回放（MuJoCo） | ⚠️ 云上 `--deploy-onnx` 正在跑；本机手搭场景不稳定，见下 |

### 本机手搭场景为什么不能用

本机用 `mujoco.MjSpec.from_file(g1.xml)` + 自建地板 + 自建位置伺服跑闭环时，
策略会前倾摔倒。已逐个排除并修正了：

1. **自由关节角速度**：MuJoCo 的 `qvel[3:6]` 已经是 **body 系**，再乘 `Rᵀ` 会让 `w_b` 错
   （已用 `mj_objectVelocity(flg=1)` 验证并改正）；
2. **`dof_armature`**：不设反射转子惯量时，1 kHz 位置伺服会把手腕/脚踝打成
   数百 rad/s 的数值爆炸（`Nan, Inf or huge value in QACC`）；
3. **出生高度**：弯腿姿势按参考高度直接落会插进地板，被接触力弹飞；
   改成用足底几何前向解算求高度；
4. **积分器/求解器**：改成 mjlab 的 `implicitfast` + newton + 100/50 迭代。

修完仍然不稳，说明还有**接触/关节参数**层面的差异（mjlab 的 `FULL_COLLISION`、
`condim`、`priority`、`solref` 等），继续猜成本太高。正确做法是
**把 mjlab 编译后的 MJCF 拿出来**做本地对拍 —— 为此已加
`replay_headless.py --dump-model`：一次云端回放就能导出 `mjlab_compiled.xml`
和解析后的 `mjlab_sim_options.txt`，之后本机 100% 复刻训练仿真。

## 3. 还缺什么（真机之前必须补）

1. **编译**：`cd G1_deploy && bash tools/build_g1_deploy.sh`（或按 README 手工 cmake），
   然后把 `supermimic_offline` 跑一遍，用同一段快照 CSV 与 Python 对拍。
2. ~~摇杆接入~~ **已接**：`vx=+ly, vy=-lx, wz=-rx`，与 `velocity_commands` 观测逐字一致。
3. **里程计漂移**：速度模式下 `v_b` 全靠 `LegOdometry`，训练时用的是**真值速度**；
   这是 sim2real 最大的未验证项，先低速、短时、吊带。
4. **进态姿态**：`enter()` 只检查倾角，没有检查「当前关节是否接近训练分布」；
   建议先 `FixStand` 站稳再切，否则第一帧观测就是分布外。
5. **安全**：`damping()` 只是 kp=0/kd=3，撑不住；真机首测必须吊带。
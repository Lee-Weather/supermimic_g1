# SuperMimic 真机操作手册（G1 EDU U2 · 29 DoF）

适用：把 `models/deploy_velocity_v1/` 这个部署包 + `deploy/supermimic_patch/` 这套代码
装到 G1 上，用遥控器切进 `SuperMimicVelocity`，摇杆/指令文件给速度，让它按速度走。

**当前状态（务必先看）**：ONNX 与 torch 策略数值一致（3.5e-06），部署观测链路已在训练用
MuJoCo 场景里闭环验证过（30 s 视频，六段速度均能站稳/加速/减速）。但
`deploy/supermimic_patch/` 这套 C++ **从未在 Linux 上编译过**，也**没有上过真机**。
第一次一定是「吊装 + 低速 + 有人盯着 `LT + B`」。

---

## 0. 把代码和包弄到机器人上

G1 机载电脑拉的是 `weilai-robot/G1_deploy`，我们提交在他们仓库的是本地 commit，
**没往他们远端推**。所以两条路，选一条：

**A. 从我们的公开仓库拷（推荐）**

```bash
cd ~
git clone https://github.com/Lee-Weather/supermimic_g1.git sm   # 只需拷文件，不用编译它
cp -r ~/sm/deploy/supermimic_patch/include/supermimic          ~/G1_deploy/deploy/include/
cp    ~/sm/deploy/supermimic_patch/robots/g1_29dof/include/State_SuperMimic.h \
      ~/G1_deploy/deploy/robots/g1_29dof/include/
cp    ~/sm/deploy/supermimic_patch/robots/g1_29dof/src/State_SuperMimic.cpp \
      ~/G1_deploy/deploy/robots/g1_29dof/src/
cp    ~/sm/deploy/supermimic_patch/robots/g1_29dof/config/supermimic_velocity.yaml \
      ~/sm/deploy/supermimic_patch/robots/g1_29dof/config/velocity_cmd.yaml \
      ~/G1_deploy/deploy/robots/g1_29dof/config/
cp    ~/sm/deploy/supermimic_patch/tools/supermimic_offline.cpp ~/G1_deploy/deploy/tools/
mkdir -p ~/G1_deploy/deploy/robots/g1_29dof/config/policy/supermimic/velocity_v1
cp -r ~/sm/models/deploy_supermimic/v1 \
      ~/G1_deploy/deploy/robots/g1_29dof/config/policy/supermimic/
```

部署包结构（与行走控制器 `config/policy/velocity/v0` 同构）：

```text
config/policy/supermimic/v1/
  exported/policy.onnx        # 单输入 obs[1,71] -> 单输出 actions[1,29]
  params/deploy.yaml          # 机器人参数、entry_pose、clip、运动学链
  params/manifest.json        # sha256 与对拍误差
```

**B. 用 u 盘 / scp**：把你手上 `deploy/supermimic_patch/` 和 `models/deploy_velocity_v1/`
按上表结构拷过去，效果一样。

然后补 `deploy/robots/g1_29dof/CMakeLists.txt` 两行（`State_SuperMimic.cpp` 会被
`file(GLOB_RECURSE src/*.cpp)` 自动收编，不用加）：

```cmake
  add_executable(supermimic_offline ../../tools/supermimic_offline.cpp)
  target_link_libraries(supermimic_offline PRIVATE beyondmimic_runtime)
```

## 1. 编译（第一次必做）

```bash
cd ~/G1_deploy
bash tools/build_g1_deploy.sh            # SDK 在 /opt 时加 UNITREE_SDK_PREFIX=/opt/unitree_robotics
ls -l deploy/robots/g1_29dof/build/g1_ctrl deploy/robots/g1_29dof/build/supermimic_offline
./deploy/robots/g1_29dof/build/g1_ctrl --help
ldd ./deploy/robots/g1_29dof/build/g1_ctrl | grep -i 'not found' || echo "依赖 OK"
```

编译报错集中在三处（我按 BeyondMimic 的同名 API 写的，但没编译过）：
`Ort::Session` 构造、`lowstate->mutex_`、`param::proj_dir`。报错把行号发我。

**不动机器人先自检**（纯本地，不发 DDS 指令）：

```bash
cd ~/G1_deploy
sha256sum deploy/robots/g1_29dof/config/policy/supermimic/velocity_v1/policy_velocity.onnx
cat      deploy/robots/g1_29dof/config/policy/supermimic/velocity_v1/manifest.json   # 比 sha256
# 离线对拍（可选，需要一份 CSV 快照；格式见 tools/supermimic_offline.cpp 头注释）
./deploy/robots/g1_29dof/build/supermimic_offline \
   deploy/robots/g1_29dof/config/policy/supermimic/velocity_v1 snapshots.csv | head
```

## 2. 机器人准备

1. **吊装**（吊带吃力，脚能踩地但摔不倒）。
2. 遥控器 `L2 + R2` 进开发/调试模式；确认机身内置运动控制**不再发包**。
3. 找内部网卡（`192.168.123.x`）：

```bash
ip -br -4 addr
G1_NET=<你看到的网卡名>      # 常见 eth0，但不一定
```

## 3. 启动

```bash
cd ~/G1_deploy/deploy/robots/g1_29dof
./build/g1_ctrl --network "$G1_NET" --domain 0 \
  --config config/supermimic_velocity.yaml --log
```

正常应该看到：

```text
Connected to robot.
SuperMimic loaded: obs 71 (goal 3 + rep 68), 29 actions, clip vx[-0.10,1.00] vy[-0.10,0.10] wz[-1.50,1.50], command file .../config/velocity_cmd.yaml
FSM: Start Passive
```

看到最后一行就是**阻尼态**，还没跑策略。策略会读 `config/velocity_cmd.yaml`，
**初始值是 `vx: 0`**，所以进态后先站在原地。

## 3.5 「站姿」是哪来的（重要）

`FixStand` 不是硬编码的姿势，它读 `config/supermimic_velocity.yaml` 里的：

```yaml
FixStand:
  ts: [0, 3]      # 两个时间节点（秒）
  qs:             # 两行 29 维关节角
    - []          # 第 0 行只是占位：State_FixStand 进态时用「实测姿态」覆盖它
    - [ -0.092, 0.009, 0.039, 0.342, -0.122, ... ]   # 第 1 行 = 目标站姿
```

**第 1 行才是站姿**，它是 `tools/gen_fsm_config.py` 从 actor 权重里算出来的：
`obs_normalizer._mean` 的 `qj` 段（第 10–38 维）= **训练数据里关节角的均值**，也就是
策略训练分布的**中心**（膝 0.342、踝 -0.122、肘 0.971、肩 roll ±0.2/∓0.546）。

为什么不沿用 BeyondMimic 的那一行：`dance2_subject3_30s.yaml` 里的 `qs[1]` 是**那段舞蹈
参考的第 0 帧**（右髋 roll -0.344、手腕上抬），对速度策略来说是分布外姿势，一进态就会
先"摔一下再找平衡"。

为什么不直接用 `default_joint_pos`（KNEES_BENT，膝 0.669）：那是 `q_des = default + 0.5a`
的零点，蹲得比数据深（膝 0.669 vs 数据均值 0.342，约 0.85σ），进态瞬间策略要先把腿伸直。

进态那一刻还会**校验**：`State_SuperMimic::enter()` 逐关节比对当前姿态与部署包里的
`entry_pose`，任一处偏差 > `max_start_error`（0.6 rad）就拒绝启动并打日志，提示先回
`FixStand`。所以流程必须是 `FixStand` 站稳 → 再 `L2 + →`。

改了姿态想重新生成（会自动同步到 `deploy/supermimic_patch/` 和 `G1_deploy/`）：

```bash
python3 tools/gen_fsm_config.py --repo ~/sm     # 或在有仓库的机器上跑完再拷过去
```

## 4. 遥控器操作

| 操作 | 结果 |
|---|---|
| `L2 + ↑` | `FixStand`：3 s 过渡到**训练数据均值站姿**并保持（PD 定姿，还不是策略） |
| 让双脚**平踩地面**，等 2–3 s | 腿式里程计需要支撑脚接地；悬空/半悬空会让 `v_b` 估计失真 |
| `L2` 按住 2 s，再按 `→` | `SuperMimicVelocity`：策略接管，50 Hz；此时 `vx=0`，应该站住 |
| `L2 + B` | 任何时刻退回 `Passive` 阻尼 |
| 终端 `Ctrl+C` | 程序请求阻尼后退出 |

自动退出条件（代码里）：倾角 > **0.85 rad**，或策略输出超过 **100 ms** 没更新。

## 5. 给速度：摇杆（和行走控制器同一套映射）

`policy/supermimic/v1` 的目录结构和 `policy/velocity/v0` 完全一致
（`exported/policy.onnx` + `params/deploy.yaml`），指令也走同一套摇杆映射：

| 摇杆 | 含义 | 代码 |
|---|---|---|
| 左摇杆 上下 `ly` | 前后速度 `vx` | `vx = +ly` |
| 左摇杆 左右 `lx` | 侧向速度 `vy` | `vy = -lx` |
| 右摇杆 左右 `rx` | 转向 `wz` | `wz = -rx` |

（和 `isaaclab/envs/mdp/observations/observations.h` 的 `velocity_commands` 逐字一致，
包括符号；随后由部署包 clip 成 `vx[-0.1,1.0]`、`vy[-0.1,0.1]`、`wz[-1.5,1.5]`。）

**推杆顺序**：先只推 `ly` 一点点（约 20%，≈0.2 m/s），站稳了再加；每档 10–15 s。

**摇杆回中时**，才读 `config/velocity_cmd.yaml`（台架测试/没有手柄时用）：

```bash
printf 'vx: 0.2\nvy: 0.0\nwz: 0.0\n' > config/velocity_cmd.yaml   # 10 Hz 内生效
```

**不要**推负 `vx`：仿真里 `vx=-0.3` 会反向发散（0.83 m/s）。包里把 `vx` 卡在 `[-0.1,1.0]`
是兜底，不是许可。

`vy`、`wz` 第一轮先保持中立。想测侧移/转向，单独一轮，`vy ±0.1`、`wz ±0.5` 起，
手放 `L2 + B`。

## 6. 这一轮要盯什么、记什么

- 终端里有没有 `SuperMimic stopping: ... (clipped N targets)`，`clipped` 增长说明关节目标
  在撞限位或撞速率限幅（8 rad/s）。
- **实际速度**只能靠肉眼/卷尺量：走 5 s，看走了几米 → 除以 5。跟指令比。
- 脚有没有打滑、膝/踝有没有明显抖动（仿真里没有伺服延迟，真机会抖）。
- 站住时会不会缓慢前移（里程计漂移的直接症状）。

## 7. 已知风险 / 下一轮要补的

| 风险 | 说明 |
|---|---|
| C++ 没编译过 | 第 1 步就可能报错，别在机器人上首次编译时试动作 |
| `v_b` 来源 | 训练用**真值速度**，真机用 `LegOdometry`（IMU+编码器，假设脚不打滑）。最大 sim2real 差距 |
| 摇杆未接 | 现在走 `velocity_cmd.yaml`；接摇杆要改 `read_command()`（`lowstate->joystick` 的模拟量接口待确认） |
| 进态姿态 | `enter()` 只查倾角，不查关节是否在训练分布内；一定先 `FixStand` 站稳再切 |
| 阻尼撑不住 | `damping()` 是 kp=0/kd=3，只能卸力，不能扶住；首测必须吊装 |
| 0.8 m/s 欠速 | 策略特性，不是接线错误；真机别指望精确跟 0.8 |

## 8. 出问题怎么报

把这几样发我，基本能定位：

```bash
# 1) 启动后到退出前的完整终端输出（含 SuperMimic loaded / stopping 那几行）
# 2) 策略是否真的在读指令文件
grep -n command_file ~/G1_deploy/deploy/robots/g1_29dof/config/supermimic_velocity.yaml
cat ~/G1_deploy/deploy/robots/g1_29dof/config/velocity_cmd.yaml
# 3) 部署包完整性
cat ~/G1_deploy/deploy/robots/g1_29dof/config/policy/supermimic/velocity_v1/manifest.json
```
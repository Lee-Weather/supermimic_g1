# SuperMimic × `G1_deploy` 补丁包

这里是我们为 `deploy/G1_deploy`（Unitree `unitree_rl_lab` + BeyondMimic 部署链路）新增的全部文件。
`deploy/G1_deploy` 是**独立 git 仓库**（有自己的 `.git`），所以本仓库不会跟踪它里面的内容，
本目录是**可版本化的镜像**，也是给别人复现的依据。

## 文件落位

| 本目录 | 目标位置（`G1_deploy/`） | 作用 |
|---|---|---|
| `include/supermimic/core.h` | `deploy/include/supermimic/core.h` | 观测 68 维 `rep_state`、目标钳位、动作映射与限幅 |
| `include/supermimic/package.h` | `deploy/include/supermimic/package.h` | `deploy.yaml` 校验 + ONNX Runner（`obs[1,71] → actions[1,29]`） |
| `robots/g1_29dof/include/State_SuperMimic.h` | `deploy/robots/g1_29dof/include/` | 状态声明 + `REGISTER_FSM` |
| `robots/g1_29dof/src/State_SuperMimic.cpp` | `deploy/robots/g1_29dof/src/` | 20 ms 策略线程、1 kHz 发布、100 ms 看门狗、阻尼兜底 |
| `robots/g1_29dof/config/supermimic_velocity.yaml` | `deploy/robots/g1_29dof/config/` | 任务级 FSM（`--config` 传入） |
| `robots/g1_29dof/config/velocity_cmd.yaml` | `deploy/robots/g1_29dof/config/` | 速度指令文件（10 Hz 轮询） |
| `tools/supermimic_offline.cpp` | `deploy/tools/` | 无硬件对拍（CSV 快照 → 动作/目标） |

另外要**改一处已存在的文件**：`deploy/robots/g1_29dof/CMakeLists.txt` 增加

```cmake
  add_executable(supermimic_offline ../../tools/supermimic_offline.cpp)
  target_link_libraries(supermimic_offline PRIVATE beyondmimic_runtime)
```

`State_SuperMimic.cpp` 无需改 CMake —— `g1_29dof/CMakeLists.txt` 用
`file(GLOB_RECURSE ... src/*.cpp CONFIGURE_DEPENDS)` 自动收编。

## 部署包（权重产物，不在本目录）

`models/deploy_velocity_v1/` 是导出器生成的部署包：

- `policy_velocity.onnx`：单输入 `obs` `[1,71]`（`[vx,vy,wz]` + `rep_state`）→ 单输出 `actions` `[1,29]`，
  两个经验归一化已折进图里（`(x-mean)/(std+0.01)`）
- `deploy.yaml`：`format: supermimic_g1_gencoord_v1`，含恒等 `sdk_indices`、训练用 `kp/kd/effort/limits`、
  腿部运动学链（供 `LegOdometry`）、速度指令钳位
- `manifest.json`：checkpoint / onnx 的 sha256 与 ONNX 对拍误差（3.5e-06）

要在 `G1_deploy` 里用，复制到
`deploy/robots/g1_29dof/config/policy/supermimic/velocity_v1/`，
然后 `./g1_ctrl --config config/supermimic_velocity.yaml`。

## 状态

- ONNX 与 torch 策略的数值一致性已验证（64 组随机采样，`max_abs_err = 3.5e-06`）。
- `core.h` / `package.h` / `State_SuperMimic.cpp` **尚未在 Linux 上编译**（本机没有
  g++/Eigen/yaml-cpp/onnxruntime），需要在机载电脑或 Linux 容器里跑一次
  `bash tools/build_g1_deploy.sh` 才能确认编译期无错。
- 速度指令目前来自 `velocity_cmd.yaml`；接摇杆需要在 `read_command()` 里换成
  `lowstate->joystick` 的模拟量（`unitree_joystick.hpp` 的数值接口待确认）。
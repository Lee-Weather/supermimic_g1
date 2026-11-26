# 云端训练可行性评估（flux-cli / Gradmotion）

评估对象：用当前 `E:\G1\G1_supermimic` 的代码 + `datasets/walk_only`（63 条 / 37.2 min / 30 MB）
在 Flux 云平台跑 `SuperMimic-GenCoord-G1`。所有平台信息来自只读查询（`flux task resource list` /
`flux task image official` / `flux task image versions` / `flux task list`），**未创建任何云端资源**。

## 一、结论

**可行，但不是开箱即用**：需要一个约 1 小时的环境搭建 + 标定任务；按当前预算只够**一次约 12 h 的正式训练**。
主要不确定性不在算力，而在**镜像自带的 NVIDIA 驱动版本能否跑 mjlab 的 CUDA 栈**。

## 二、平台现状（实测）

| 项 | 值 |
|---|---|
| 算力 SKU | `ESKU000001` = 1×4090D 24 GB / 20 vCPU / 89 GB RAM，**5.76 单位/h**（唯一可选 RL 资源） |
| 官方镜像 | 12 个。最新是 IsaacSim 5.0/IsaacLab 2.2（Python 3.11.13、torch 2.7.0、Ubuntu 22.04）；另有裸机 `Ubuntu 24.04`（`BJX00000119` / 版本 `V000158`）、`ubuntu:22.04`（`BJX00000016`） |
| 账号池额度 | 8 个账号，**6 个已耗尽**（300/400 单位），剩 2 个各 50 → **100 单位 ≈ 17.4 GPU-小时** |
| 今日已消耗 | `TASK_20261124_156`（10:52–14:57，4.1 h）+ `TASK_20261124_226`（14:15–15:49，1.6 h）≈ **5.6 GPU-h ≈ 32 单位**，真实余额需以平台为准 |
| 代码/数据入口 | `codeUrl`（git 仓库，平台 clone）+ `mainCodeUri`（从仓库目录名起）+ `startScript`（**必须以 `gm-run` 开头**） |

## 三、三个硬约束 vs 平台能力

| 约束 | 要求 | 平台现状 | 判定 |
|---|---|---|---|
| Python | mjlab 要求 **3.13**（`>=3.13,<3.14`） | 镜像最高 3.11.13 | ⚠️ 需在 pod 内 `uv python install 3.13` + `uv sync`（用户态、免 sudo） |
| CUDA / 驱动 | mujoco-warp + warp-lang + torch `cu126/cu130`；GPU 需 Turing+；驱动 **≥525（推荐 ≥550）** | 4090D 满足算力架构；**镜像驱动的具体版本未公开** | ⚠️ 最大风险，必须实测 `nvidia-smi` |
| 显存 | SuperMimic 默认 `num_envs=4096` + BPTT 32 + CUDA graphs | 24 GB（本地 A2000 只有 12 GB，且慢 4–6×） | ✅ 但建议 1024 起，标定后上调 |

补充：数据只有 **30 MB**，随 git 仓库走最省事（无需推送 4 GB 原始素材）；`walk_only` 已验证 0 problems。

## 四、准备清单（我可代做，需逐项确认）

1. **代码/数据入库**：把仓库推到平台能访问的 git（含 `supermimic/`、`mjlab_research/`、`datasets/walk_only/packed`），
   `mainCodeUri` 指向 `supermimic/src/supermimic/scripts/train.py`。
2. **`cloud/bootstrap.sh`**：装 uv → `uv python install 3.13` → `uv sync` → 打印 `nvidia-smi`、`torch.cuda.is_available()`、
   显存峰值 → 跑 200 iter 标定 → 把 it/s 写进日志。
3. **冒烟任务（1 h，约 6 单位）**：只验证 4 件事——`import mjlab` 成功、CUDA graph 捕获成功、
   显存峰值 < 22 GB、得到 it/s 基准。
4. **正式任务（10–12 h，约 58–70 单位）**：`num_envs` 用标定值，`--checkpoint-interval 1000`，
   每 10 min 拉一次日志（`flux task logs --follow`）。

## 五、成本估算

| 阶段 | 时长 | 单位 | 说明 |
|---|---|---|---|
| 环境搭建 + 标定 | 1 h | 5.8 | 一次性；含 uv 装 Python、首次 `torch.compile` 编译 |
| 冒烟 200 iter | 含上 | 0 | 验证显存与吞吐 |
| 正式训练 | 12 h | 69.1 | 若达 20 it/s @1024 envs → 约 86 万 iter |
| **合计** | **13 h** | **≈75** | 预留约 25 单位（4.3 h）应对失败重跑 |

作为参照：本地 A2000（12 GB）能跑但慢 4–6×，适合调参迭代；云端适合**最终一次长训练**。

## 六、风险与对策

| 风险 | 概率 | 对策 |
|---|---|---|
| 镜像驱动 < 525，torch/warp 起不来 | 中 | 换 `ubuntu:22.04`；或 `uv sync --torch-backend cu126`；最坏本地编译 CPU 版只做 import 验证 |
| pod 内 pip 源不可达 / 很慢 | 中 | 先跑 5 min 连通性测试；备选把 wheel 缓存挂进仓库（torch + warp ≈ 3 GB，不推荐） |
| 无 sudo 导致系统库缺失（libGL 等） | 低 | mjlab 训练 headless（MuJoCo Warp 不走 GL），渲染另行处理 |
| 单任务最长时长未知（文档未给） | 中 | 训练脚本必须可 checkpoint 续训（平台 `trainType=2`） |
| 预算被今日任务消耗 | 高 | 开跑前确认平台余额，再决定 `num_envs` 与时长 |
| 4096 envs 显存溢出 | 中 | 从 1024 起，标定后按 `nvidia-smi` 峰值上调 |

## 七、建议

先做**冒烟任务（1 h）**而不是直接长训练：这一步只花约 6 单位，却能一次性排除驱动、Python 3.13、显存三项风险。
标定拿到 it/s 后，再决定是 1024 还是 2048 envs、跑 8 h 还是 12 h。

需要你确认三件事：① 代码推到哪个仓库（GitHub 私有 / 内部 GitLab）；② 平台当前真实余额；
③ 是否授权我创建这个 1 h 的冒烟任务（会消耗约 6 单位）。

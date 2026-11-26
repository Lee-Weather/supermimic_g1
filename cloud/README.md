# 云端训练（Gradmotion / flux-cli）

本目录是给 Flux 云平台用的入口脚本。平台创建任务时通过 `codeUrl` clone 一个 git 仓库，
所以**代码 + 数据集必须都在这个仓库里**（数据集只有 29.4 MB，随仓库走最简单）。

## 仓库内容（已按此结构准备）

| 路径 | 是否入库 | 说明 |
|---|---|---|
| `supermimic/` | ✅ | 训练代码（`src/supermimic`），依赖 mjlab |
| `mjlab_research/` | ✅ `src/` `scripts/` `pyproject.toml` `uv.lock` | 路径依赖（`../mjlab_research`），必须与 supermimic 同级 |
| `datasets/walk_only/npz/` | ✅ 29.4 MB | 63 条 / 37.2 min 行走数据集，训练默认入口 |
| `datasets/walk_only/SHA256SUMS` | ✅ | 数据校验；容器内可 `sha256sum -c` |
| `datasets/g1_ase_45/npz/` | ✅ 4.1 MB | 原始 UMR/ACCAD 子集（转向/后退/起步） |
| `datasets/tools/`、`*.md`、`manifest.json` | ✅ | 转换/校验/审计脚本与报告 |
| `datasets/*/packed/` | ❌ | 派生物（约 32 MB），bootstrap 会在容器内用 `pack_motions.py` 生成 |
| `datasets/source/`、`datasets/lafan1_g1/`、`datasets/walk_gapfill/` | ❌ | junction 指向 `E:\G1\datasets\...`，训练不需要 |
| `czy/` | ❌ | 本地工具与预览，**内含 `flux-cli/api_key.json`，绝不入库** |

## 平台侧参数

| 字段 | 值 |
|---|---|
| 算力 `goodsId` | `ESKU000001`（1×4090D 24 GB / 20 vCPU / 89 GB，5.76 单位/h） |
| 镜像 `imageId` / `imageVersion` | `BJX00000119` / `V000158`（Ubuntu 24.04，裸机）<br>备选 `BJX00000016`（ubuntu 22.04）——**不要用 Isaac Gym 镜像**，Python 3.8 太旧 |
| `mainCodeUri` | `supermimic/src/supermimic/scripts/train.py` |
| `startScript` | `gm-run cloud/start_train.sh --num-envs 1024 --iters 120000`（**必须以 `gm-run` 开头**） |
| `gpuNum` | 1 |
| `trainType` | 1（新训练）；断点续训用 2 |

> `codeUrl` 必须是平台可访问的 git 地址（GitHub 私有仓库 / 内部 GitLab）。仓库根目录名即 clone 后的目录名，
> `mainCodeUri` 与 `startScript` 里的路径都以它为起点。

## 上云前要做的 4 步（第 1 次必须全做）

1. **推送仓库**（含 29.4 MB 数据）——见仓库根目录的推送说明。
2. **bootstrap**：`bash cloud/bootstrap.sh`。它会装 uv、Python 3.13、torch(cu126)、mjlab+supermimic
   （含 `mujoco-warp`），并做 **20 iter / 64 envs 冒烟**。任何一步失败立即退出，不浪费 GPU 时间。
3. **看三个数**：`nvidia-smi` 的驱动版本、`torch.cuda.is_available()`、冒烟的显存峰值。
   驱动 < 525 时 `torch cu126` 起不来——换 `ubuntu:22.04` 镜像或 `bash cloud/bootstrap.sh --cu128`。
4. **标定吞吐**：冒烟日志里的 `it/s`，用它算正式训练时长（目标 10–12 GPU 小时）。

## 正式训练

```bash
# 任务 startScript 里：
gm-run cloud/start_train.sh --num-envs 1024 --iters 120000
```

- `--runner.save-interval 2000` 写 checkpoint（默认），断点续训用 `trainType=2` + `checkPointFilePath`；
- `wandb` 默认 `WANDB_MODE=offline`（镜像里没有你的 wandb 凭据，避免启动即失败）；
- 显存 24 GB：`num_envs=4096`（代码默认）有 OOM 风险，从 1024 起，按冒烟峰值上调。

## 下载产物

```bash
flux task model list --task-id "TASK_xxx" --page 1 --limit 20     # 看有哪些 checkpoint
flux task model download --task-id "TASK_xxx" --checkpoint 2000   # 下载到本地
```

## 已知限制

- 平台镜像最高 Python 3.11，所以**每次新 pod 都要重跑 bootstrap**（约 6–10 min，含 torch 下载）；
  想省这一步就用「个人镜像」把装好的环境打成镜像（需平台支持构建）。
- 单任务最长运行时长平台文档未给出；长训练请依赖 checkpoint 续训，不要假设能一口气跑完。
- 数据只有 37 min，若后续扩到几小时，建议改用平台存储（`personalDataPath`）而不是塞进 git。
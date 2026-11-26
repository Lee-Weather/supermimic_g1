# G1 行走跟踪模型（walk-only，约 10 万步）

SuperMimic 在 MuJoCo Warp 里训练的 Unitree G1 29-DoF 行走跟踪策略与世界模型。

## 来源

| 项 | 值 |
|---|---|
| 平台任务 | `TASK_20261126_328`（项目 `PRO_20261126_004`） |
| 训练时间 | 2026-11-26 13:46:33 → 17:02:15（11,742 s，`train_exit_code=0`） |
| 数据 | `datasets/walk_only`，63 条片段 / 37.2 min / 111,692 帧（50 Hz） |
| 配置 | 1024 envs、`max-iterations 100000`（策略步）、`save-interval 2000`、`WANDB_MODE=offline` |
| 最终 checkpoint | 平台 `policUrl` 时间戳 2026-11-26 **16:58:59 / 16:59:01** |
| 权重环境 | Python 3.13.18、torch 2.14.1+cu126、warp-lang **1.17.0**、mujoco 4.0.1 |

> 注意：`task model list` 里 `createTime` 显示 14:13，是因为平台按文件名唯一、原地覆盖同一行；
> 真实版本时间在 `policUrl` 的文件名里（`world_model_20261126165859A054.pt`）。

## 文件

| 文件 | 大小 | SHA256 |
|---|---|---|
| `world_model.pt` | 33.37 MB | `5D7EC0FBDA09EBB339CA40621FD61E139B4CD0B28CAA5C5EE0D59C65EDF46909` |
| `actor.pt` | 37.55 MB | `F6A5C9E7E741AD464D84BC98B16993B7C58064EECB54FD6C7B5D6C081D4D5CDE` |
| `critic.pt` | 0.89 MB | `8AB6D2ABC11B89786E7FC746FA85183609739A2E607F94174E677FEA144BB6D1` |

- `world_model.pt`：状态表示（`representation`）+ 可微分动力学（`dynamics`），回放时产生"预测幽灵"；
- `actor.pt`：跟踪策略，输入 `history`（本体观测历史）+ `goal_motion`（参考目标），输出 29 维关节目标；
- `critic.pt`：价值网络，仅训练期使用，回放/部署不需要。

## 最终训练指标（末次 log block，第 9999/10000 块）

| 指标 | 值 |
|---|---|
| Mean reward | **44.86** |
| `error_joint_pos` | **0.3769 rad**（全程最好） |
| `error_anchor_pos` | 0.1764 m |
| `Episode_Termination/time_out` | 1.33 |
| `Episode_Termination/motion_end` | 0.89 |
| `Episode_Termination/anchor_pos` / `anchor_ori` | **0 / 0**（无摔倒终止） |

训练曲线：`metrics_TASK_20261126_328.csv`（每 10 分钟采样）。

## 这是一份什么模型

- ✅ **参考动作跟踪**：平地、直行 0.36–1.07 m/s、转向、后退、站立、起步/停止（63 条片段覆盖）；
- ✅ **可微分世界模型**：`(state, action) → next state`，可用于规划、模型式训练、预测误差分析；
- ❌ 不是速度指令策略（`slot_mix={"motion":1.0,"velocity":0.0}`，velocity 槽位未训练）；
- ❌ 不做地形、外力扰动、域随机化；每回合上限 10 s（500 控制步 @50 Hz）；
- ⚠️ `error_joint_pos ≈ 0.38 rad` 表示"跟得住"而非"高保真复现"。

## 如何加载

`supermimic/scripts/play.py` 目前要求 `wandb_path` 从 wandb artifact 下载权重；用本地文件需加
`--model-dir` 分支（`torch.load(model_dir / "world_model.pt")` / `actor.pt`）。
回放环境要求同版权重栈（py3.13 + torch 2.14 + **warp 1.17**；驱动 12.2/12.6 均可）。
## 版本管理说明

`.gitignore` 里 `*.pt` 被排除，因此本目录只有 `README.md` 会进 git，三个权重文件**仅存在于本地磁盘**。
如果需要在其他机器上复现，两个办法：① 从平台 `flux task model list` 用 `policUrlDown` 重新下载
（签名 URL 有效期有限）；② 手动把 `.pt` 传到对象存储/网盘。

# G1 行走 + 速度指令模型（第三阶段，motion 0.5 / velocity 0.5）

SuperMimic 多任务训练：同一个 actor 的输入槽位一半给**动作跟踪**（motion），一半给**速度指令**（velocity），
因此这一个权重同时具备两种能力。

## 来源

| 项 | 值 |
|---|---|
| 平台任务 | `TASK_20261126_508`（项目 `PRO_20261126_066`，账号 `limxfada3b6m5o6p@euploading.com`） |
| 训练时间 | 2026-11-26 17:47:28 → 19:46:47（7,159 s = 1.99 h，status=5 正常结束） |
| 规模 | **4096 envs × 30000 步** = 1.23 亿 env-step，20,447 steps/s |
| 槽位 | `slot_mix = {"motion": 0.5, "velocity": 0.5}`（`SM_MIX_VELOCITY=0.5`） |
| 速度范围 | **conservative**：`vx∈[-0.3,1.1]`、`vy∈[-0.2,0.2]`、`wz∈[-2,2]`，噪声 `(0.25,0.1,0.4)` |
| 数据 | `datasets/walk_only`（63 条 / 37.2 min / 111,692 帧 @50 Hz） |
| checkpoint | 每 500 步（服务端保留 60 个 `critic_*.pt` 版本 + 最终 `actor/world_model`） |

## 最终指标（第 2999/3000 log 块 = 第 30000 步）

| 指标 | 值 | 对比 100k 步纯 motion 版 |
|---|---|---|
| Mean reward | **44.19** | 44.86 |
| Mean episode length | **497.18 / 500**（基本跑满 10 s） | — |
| `error_joint_pos` | **0.5325 rad** | 0.3769 |
| `error_anchor_pos` | **0.1642 m** | 0.1764 |
| `error_anchor_rot` | 0.0789 | — |
| `error_body_pos` | **0.0336** | — |
| `error_joint_vel` | 2.947 | — |
| `Episode_Termination/time_out` | 6.3 | 1.33 |
| `Episode_Termination/motion_end` | 2.9 | 0.89 |

结论：**回合几乎不摔了**（episode length 497/500，无 anchor 终止），motion 跟踪精度略逊于纯 motion 版
（0.53 vs 0.38 rad）——这正是把一半槽位分给速度指令的代价，属于预期。

## 验证：velocity 槽**确实**在训练

用 `datasets/tools/compare_ckpt.py`（不依赖 torch，直接解析 torch zip 的 pickle + CRC32）对比
第 0 步 actor 与最终 actor：

```
tensors: 66  common: 66  changed: 66  identical: 0
[encoders.velocity] 18 tensors, 18 changed
[encoders.motion]   18 tensors, 18 changed
```

`encoders.velocity.*` 全部 18 个张量都变了 → 速度指令编码器真实接收了梯度，第三阶段生效。

## 文件

| 文件 | 大小 |
|---|---|
| `world_model.pt` | 33.37 MB |
| `actor.pt` | 37.55 MB |
| `critic.pt` | 0.89 MB |

（`.gitignore` 排除 `*.pt`，权重只在本地；服务端可用 `flux task model list --task-id TASK_20261126_508`
+ `policUrlDown` 重新下载，最终版本时间戳 2026-11-26 19:45:1x。）

## 怎么用（推理端）

- 回放/跟踪：`play.py --initial-task motion`，给参考动作；
- 速度指令：`play.py --initial-task velocity`，给体坐标系 `[vx, vy, wz]`：
  - ✅ 可靠区：**前进 0.3–1.0 m/s**、`|wz| ≤ 1.5 rad/s`；
  - ⚠️ 名义到 ±1.1 m/s / ±2 rad/s，但训练范围就是保守档，超出即不可信；
  - ❌ 侧移 `vy` 数据几乎无覆盖，别指望。

两者共用同一 actor（`viewers.py::set_slot_mix` 在推理期切换槽位），`--initial-task` 决定走哪条路。

## 已知限制

- 中间 `actor_*.pt` 快照未保留（平台按内容去重，只有 `critic_*` 小文件齐全；固定名行保留最新一版）；
- 每回合上限 10 s（500 控制步 @50 Hz），长距离行走需切段；
- 平地、无地形/外力扰动/域随机化，上真机需 DR 与部署适配；
- 速度指令范围保守是刻意的：宽档（±1.5 m/s、±3.14 rad/s）会让 27% 指令落在数据支撑不到的区域。
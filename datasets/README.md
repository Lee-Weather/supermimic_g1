# SuperMimic 训练数据集（已转换就绪）

本目录把散落在多处的 G1 动作数据统一成 **mjlab `MotionLoader` 可直接读取的 NPZ**，供
`supermimic` 的 `--motion-file` 使用。所有 clip 都经过同一套结构与物理校验，校验结果见
[manifest.json](manifest.json)（由 `tools/build_manifest.py` 生成，是唯一权威计数来源）。

## 目录结构

```
datasets/
├── README.md                      # 本文件
├── manifest.json                  # 机器可读清单 + 校验结果（每次重跑 build_manifest 覆盖）
├── tools/                         # 全部转换/校验/渲染脚本，见文末"工具清单"
├── walk_only/                     # ★ 行走训练集：63 条 / 37.2 min（推荐入口）
│   ├── npz/
│   └── packed/
├── walk_gapfill/                  # 新补的直行：CMU + HDM05 经 UMR（30 条 / 33.1 min）
│   ├── npz/
│   ├── packed/
│   └── batch_report.json          # 逐条耗时与接地指标
├── g1_ase_45/                     # UMR/ACCAD 原始集（含转向/后退/起步）
│   ├── npz/                       # 45 个 clip
│   ├── packed/
│   └── smoke/                     # 3 个 clip 的最小冒烟集（stand / walk / run）
├── lafan1_g1/                     # LAFAN1 经 GMR 重定向（接地不合格，勿用于训练）
│   ├── npz/
│   └── packed/
└── source/                        # 原始素材（训练不需要，可删）
    ├── lafan1_bvh/                # LAFAN1 BVH 77 条
    ├── p0_src/                    # P0 选中的 CMU/HDM05 SMPL-X 序列
    ├── p0_qpos/                   # UMR 输出的 qpos（36 维）
    └── hdm05_probe/               # 早期单条验证用的探针
```

## 数据来源

| 数据集 | 来源 | 处理方式 |
|---|---|---|
| `walk_only` | `g1_ase_45` 的 walk 类 + `walk_gapfill` 全部 | 直接合并（符号意义上），**训练用这一个** |
| `walk_gapfill` | `E:\F1\F1_ase_h\czy\npz\{CMU,HDM05}.tar.bz2`（SMPL-X `stageii.npz`） | UMR（`czy/g1/umr_g1_config.json`）逐帧重定向 → 50 Hz 重采样 → mjlab 模型 FK + rolling 接地 |
| `g1_ase_45` | `E:\G1\G1_ase_h\G1_ase\ase\data\motions\g1`（由 `czy/g1_wbt/scripts/umr_g1_to_tracking.py` 从 UMR/ACCAD 生成） | 已是 50 Hz、G1 关节顺序、wxyz 四元数；只做字段重映射 `body_*_w[:, pelvis] → root_*_w` |
| `lafan1_g1` | `lafan1.zip`（Ubisoft LAFAN1，77 条 BVH，30 fps） | GMR（`bvh_lafan1_to_g1.json`）逐帧 IK → qpos → 50 Hz 重采样 → MuJoCo FK → 六字段 |

`lafan1_g1` 与原仓库默认的 `supermimic/wandb-registry-motions/lafan_run1_subject5` 同源（LAFAN1），
因此无需 wandb 权限即可在本地复现默认数据配置。

## 规模（最终，见 [manifest.json](manifest.json)）

| 数据集 | clip 数 | 帧数 | 时长 | 磁盘 |
|---|---|---|---|---|
| **`walk_only`** ★ | **63** | **111 692** | **37.2 min** | 30 MB（packed 32 MB） |
| ↳ `walk_gapfill` | 30 | 99 194 | 33.1 min | 26 MB（packed 28 MB） |
| ↳ `g1_ase_45` 的 walk 类 | 33 | 12 498 | 4.2 min | 3.3 MB |
| `g1_ase_45`（全部） | 45 | 14 738 | 4.91 min | 3.9 MB（packed 4.0 MB） |
| `lafan1_g1`（勿用于训练） | 71（源 77，剔除 6） | 763 762 | 254.6 min | 190.3 MB（packed 206.9 MB） |

`walk_only` 的能力构成：直行 **33.2 min**、转向 2.6 min、后退 0.75 min、起步/停止 0.44 min、站立 0.25 min。
直行来自 11 个受试者来源（CMU 7 组 + HDM05 4 人），速度覆盖 **0.36–1.07 m/s**（v50），
最长连续片段 **183 s**。新增 30 条的接地实测：脚底中位 **0.40 cm**（最大 0.80），
触地率中位 **96%**（最低 75%，23/30 条 ≥90%），**0 条告警**。

LAFAN1 剔除的 6 条是 `obstacles1_subject1/2/5`、`obstacles4_subject2/3`、`obstacles5_subject3`
——越过障碍类动作要求骨盆升到 1.9 m 左右（人体 1.75 m 映射到 1.32 m 的 G1 上 IK 会"飞起来"），
按根高 0.05–1.8 m 的物理边界被拒，属预期行为。若要放宽，改
`tools/ase_to_mjlab_npz.py::ROOT_Z_RANGE` 后重跑对应 BVH 即可。

各 LAFAN1 片段时长 1.8–5 min，覆盖 walk / run / sprint / turn / dance / fight / jumps / push /
fallAndGetUp / ground / multipleActions / aiming；但接地不合格，**只作素材保留**。

质量抽检（两批数据一起统计）：

| 指标 | 结果 |
|---|---|
| 超出 G1 MJCF 关节限位（29 关节逐帧） | **0.000%**（`g1_ase_45` 与 `lafan1_g1` 都是 0，最大越界 0.00 rad） |
| 四元数模长偏差 | ≤ 2e-3（阈值内） |
| 根高（骨盆）范围 | ASE 0.73–0.82 m（站立态）；LAFAN1 0.35–1.08 m |
| NaN / Inf | 无 |

## 格式契约（与 mjlab 对齐）

每个 `.npz` 含 7 个数组（`fps` 为附加信息，`MotionLoader` 忽略）：

| 字段 | 形状 | 说明 |
|---|---|---|
| `joint_pos` | `(T, 29)` | float32，关节角，顺序见下 |
| `joint_vel` | `(T, 29)` | float32，关节角速度 |
| `root_pos_w` | `(T, 3)` | 根（`pelvis`）世界位置，首帧 XY 归零，脚底贴地 |
| `root_quat_w` | `(T, 4)` | **wxyz** 四元数（与 mjlab NPZ 约定一致，非 CSV 的 xyzw） |
| `root_lin_vel_w` | `(T, 3)` | 世界系线速度 |
| `root_ang_vel_w` | `(T, 3)` | 世界系角速度 |
| `joint_names` | `(29,)` | 必须等于 G1 MJCF 关节顺序 |

关节顺序（= `g1.xml` 文档顺序 = `mjlab/scripts/csv_to_npz.py::JOINT_NAMES`）：
左腿 6 → 右腿 6 → 腰 3（yaw/roll/pitch）→ 左臂 7 → 右臂 7。转换器会校验，不一致时按名重排。

## 用法

```bash
# 单批 clip（glob / 逗号列表 / 单文件都可以）
uv run src/supermimic/scripts/train.py SuperMimic-GenCoord-G1 \
  --num-envs 512 \
  --motion-file "datasets/g1_ase_45/npz/*.npz"

# 或 packed 目录（上万 clip 时启动更快，本目录同样可用）
uv run src/supermimic/scripts/train.py SuperMimic-GenCoord-G1 \
  --num-envs 512 \
  --motion-file "datasets/g1_ase_45/packed"
```

传入 `--motion-file` 时 [train.py](../supermimic/src/supermimic/scripts/train.py) 会自动清空
`registry_name`，不需要 wandb 凭据。

## 校验与重建

```bash
cd datasets

# 校验（结构 + 物理边界：NaN / 四元数模长 / 关节跳变 / 根高度 / 速度 / fps）
python tools/ase_to_mjlab_npz.py --input g1_ase_45/npz --output g1_ase_45/npz --verify
python tools/ase_to_mjlab_npz.py --input lafan1_g1/npz  --output lafan1_g1/npz  --verify

# 重新生成 manifest（同时复验所有 clip 与 packed 目录）
python tools/build_manifest.py --root .

# 重新打包
python ../mjlab_research/src/mjlab/scripts/pack_motions.py \
  --input "g1_ase_45/npz/*.npz" --output g1_ase_45/packed
```

校验阈值（`tools/ase_to_mjlab_npz.py`）：`|q| ≤ 3.2 rad`、单帧关节跳变 `≤ 1.5 rad`、
根速 `≤ 12 m/s`、角速 `≤ 80 rad/s`、根高 `0.05–1.8 m`、四元数模长误差 `≤ 2e-3`、`fps == 50`。
这些是有意放宽的"单位/轴向错误探测器"，不是动作风格审查。

## 接地处理（重要）

接地必须用 **mjlab 自己的 G1 模型**来量，否则会得到错误的根高。两批数据最初各自按
UMR / GMR 的 MJCF 对齐，放到 mjlab 模型上量出来会差几厘米，处理过程：

1. `tools/reground_to_mjlab_model.py --apply`：逐条按 mjlab 模型的最低脚底重新对齐，
   平移了 **82 / 116** 条（`lafan1_g1` 中位 1.13 cm、最大 6.40 cm；`g1_ase_45` 12 条 0.1–1.69 cm）。
2. `tools/apply_rolling_ground.py --apply`：对 `g1_ase_45` 里 **6 条** 系统性悬空片段做
   rolling 接地（±0.3 s 局部最低脚底），脚底 p50 从 3.0–5.0 cm 降到 **0.37–0.46 cm**、
   触地率 11–32% → **93–99%**，无穿地（min ≥ −0.00），根高起伏 std 基本不变。

结果（`tools/grounding_report.py`，阈值：中位脚底 >2 cm 或触地率 <10%）：
`g1_ase_45` 的 walk-only **33 条只剩 1 条**告警，且是 `Female1_A14_stand_to_skip`
（起跳动作，真实腾空，属误报）；`lafan1_g1` 的 12 条走段**全部仍告警**（0.4% 触地），
另有 5 条骨盆塌陷（z_min 0.08–0.44 m），见 [walk_only_grounding.md](walk_only_grounding.md)。

相关工具与报告：
[reground_to_mjlab_model.py](tools/reground_to_mjlab_model.py)、
[apply_rolling_ground.py](tools/apply_rolling_ground.py)、
[grounding_report.py](tools/grounding_report.py)、
[compare_grounding_variants.py](tools/compare_grounding_variants.py) →
[grounding_comparison.md](grounding_comparison.md)、
[compare_umr_vs_gmr.py](tools/compare_umr_vs_gmr.py) → [umr_vs_gmr.md](umr_vs_gmr.md)。

## 已知限制

- **采样权重未迁移**：`dataset_g1.yaml` 里 ASE 的逐 clip 权重（站姿 5%、停步 25% 等）在 mjlab 里没有对应机制，`MotionLoader` 会按片段等概率采样。若需要复现该配方，得在 `supermimic/runners/reset_sampling.py` 或数据层自己实现。
- **`lafan1_g1` 是 GMR 重定向结果且接地不合格**：12 条走段触地率仅 0.4%、5 条骨盆塌陷，**不建议直接进训练集**；GMR 侧的两个旋钮（`offset_to_ground`、缩小 `human_scale_table`）实测都无效，只有后处理 rolling 接地有效（见 [grounding_comparison.md](grounding_comparison.md)）。行走训练请优先用 `g1_ase_45`。
- **`g1_ase_45` 仍有 6 条轻微悬空**（触地 57–72%，脚底 p50 1.2–1.8 cm，未处理）：`Female1_B5_walk_backwards`、`Female1_B13_walk_turn_right_45`、`Male2_B14_turn_right_45_t2`、`Female1_B3_walk1`、`Female1_B7_walk_backwards_turn_forwards`、`Male2_B15_turn_around`。要收紧可对它们跑同一条 `apply_rolling_ground.py`。
- **未做人工可视化核验**：数值校验只证明运动学合法，不代表"看起来像人"；抽查视频见 `czy/dataset_previews/` 与 `czy/dataset_previews_umr_flagged/`。
- `source/lafan1_bvh` 是原始素材（332 MB），只用于重跑重定向；训练不需要它。
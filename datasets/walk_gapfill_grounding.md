# walk-only 子集的接地质量（mjlab 模型实测）

判定：中位脚底高度 > 2.0 cm **或** 触地帧占比 < 10% 即判为有问题。

**0 / 30 条有问题**（占比 0%）

| 类别 | 总数 | 有问题 | 中位脚底 cm | 触地率 |
|---|---|---|---|---|
| unclassified | 30 | 0 | 0.0 | 0.0% |

## 逐条明细

| 数据集 | clip | 类别 | 时长 | 脚底 p10 | p50 | p90 | 触地率 | 根高 std | 判定 |
|---|---|---|---|---|---|---|---|---|---|
| walk_gapfill | `HDM_tr_05-03_04_120_stageii_g1` | unclassified | 53.8s | +0.1 | +0.8 | +23.2 | 74.7% | 5.0 cm | ✅ |
| walk_gapfill | `38_04_stageii_g1` | unclassified | 71.9s | +0.1 | +0.8 | +2.0 | 89.6% | 1.0 cm | ✅ |
| walk_gapfill | `HDM_bk_02-02_02_120_stageii_g1` | unclassified | 59.1s | +0.1 | +0.6 | +1.5 | 96.8% | 1.4 cm | ✅ |
| walk_gapfill | `HDM_bk_01-01_03_120_stageii_g1` | unclassified | 74.7s | +0.1 | +0.6 | +2.1 | 88.9% | 1.2 cm | ✅ |
| walk_gapfill | `HDM_bk_02-02_01_120_stageii_g1` | unclassified | 55.7s | +0.1 | +0.6 | +1.5 | 96.6% | 1.6 cm | ✅ |
| walk_gapfill | `56_07_stageii_g1` | unclassified | 78.5s | +0.1 | +0.6 | +2.3 | 87.0% | 4.0 cm | ✅ |
| walk_gapfill | `HDM_bk_01-01_02_120_stageii_g1` | unclassified | 72.9s | +0.1 | +0.5 | +1.7 | 93.6% | 1.1 cm | ✅ |
| walk_gapfill | `HDM_dg_01-03_03_120_stageii_g1` | unclassified | 29.9s | +0.0 | +0.5 | +2.3 | 87.7% | 1.8 cm | ✅ |
| walk_gapfill | `35_27_stageii_g1` | unclassified | 81.7s | +0.1 | +0.5 | +1.3 | 98.7% | 0.7 cm | ✅ |
| walk_gapfill | `HDM_bk_01-01_01_120_stageii_g1` | unclassified | 85.7s | +0.1 | +0.5 | +1.6 | 93.8% | 1.0 cm | ✅ |
| walk_gapfill | `41_09_stageii_g1` | unclassified | 64.3s | +0.0 | +0.4 | +6.7 | 87.3% | 5.8 cm | ✅ |
| walk_gapfill | `15_14_stageii_g1` | unclassified | 74.9s | +0.1 | +0.4 | +1.6 | 96.7% | 0.8 cm | ✅ |
| walk_gapfill | `15_03_stageii_g1` | unclassified | 104.6s | +0.1 | +0.4 | +1.6 | 96.6% | 0.8 cm | ✅ |
| walk_gapfill | `HDM_dg_01-03_02_120_stageii_g1` | unclassified | 31.6s | +0.0 | +0.4 | +2.3 | 89.0% | 1.7 cm | ✅ |
| walk_gapfill | `HDM_dg_03-01_01_120_stageii_g1` | unclassified | 69.5s | +0.0 | +0.4 | +1.5 | 95.8% | 1.4 cm | ✅ |
| walk_gapfill | `HDM_dg_01-01_02_120_stageii_g1` | unclassified | 63.8s | +0.0 | +0.4 | +1.1 | 99.6% | 1.6 cm | ✅ |
| walk_gapfill | `56_05_stageii_g1` | unclassified | 61.3s | +0.0 | +0.4 | +1.3 | 96.0% | 2.3 cm | ✅ |
| walk_gapfill | `HDM_mm_01-03_02_120_stageii_g1` | unclassified | 33.5s | +0.0 | +0.4 | +1.8 | 91.2% | 1.4 cm | ✅ |
| walk_gapfill | `HDM_dg_01-01_01_120_stageii_g1` | unclassified | 62.9s | +0.0 | +0.4 | +1.2 | 99.0% | 1.7 cm | ✅ |
| walk_gapfill | `HDM_dg_01-01_03_120_stageii_g1` | unclassified | 64.1s | +0.0 | +0.4 | +1.2 | 99.1% | 1.7 cm | ✅ |
| walk_gapfill | `HDM_tr_03-01_01_120_stageii_g1` | unclassified | 78.8s | +0.0 | +0.3 | +1.0 | 99.6% | 1.2 cm | ✅ |
| walk_gapfill | `HDM_mm_01-03_01_120_stageii_g1` | unclassified | 32.7s | +0.0 | +0.3 | +1.8 | 91.3% | 1.5 cm | ✅ |
| walk_gapfill | `86_10_stageii_g1` | unclassified | 63.2s | +0.0 | +0.3 | +1.4 | 95.1% | 3.1 cm | ✅ |
| walk_gapfill | `HDM_tr_01-01_01_120_stageii_g1` | unclassified | 74.1s | +0.0 | +0.3 | +1.0 | 98.8% | 1.2 cm | ✅ |
| walk_gapfill | `HDM_mm_01-01_02_120_stageii_g1` | unclassified | 68.8s | +0.0 | +0.3 | +1.2 | 96.5% | 1.8 cm | ✅ |
| walk_gapfill | `HDM_mm_01-01_01_120_stageii_g1` | unclassified | 69.7s | +0.0 | +0.2 | +1.2 | 96.4% | 1.9 cm | ✅ |
| walk_gapfill | `HDM_mm_02-02_03_120_stageii_g1` | unclassified | 35.3s | +0.0 | +0.2 | +0.9 | 98.8% | 1.7 cm | ✅ |
| walk_gapfill | `HDM_mm_02-02_02_120_stageii_g1` | unclassified | 70.8s | +0.0 | +0.2 | +1.0 | 99.8% | 1.5 cm | ✅ |
| walk_gapfill | `120_19_stageii_g1` | unclassified | 106.6s | +0.0 | +0.1 | +0.8 | 99.9% | 0.7 cm | ✅ |
| walk_gapfill | `120_20_stageii_g1` | unclassified | 89.5s | +0.0 | +0.1 | +0.9 | 99.7% | 0.7 cm | ✅ |

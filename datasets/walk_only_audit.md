# walk-only 数据集齐备性审计

本报告只读扫描 `datasets/` 下已转换的 clip。除时长与根速外，每条 clip 都用 **mjlab 的 G1 模型**
做正运动学，统计**腾空帧比例**（双脚脚底同时高于 2 cm 即判腾空）与**脚底最低高度**。

## 按动作类别汇总

| 类别 | clips | 时长 | 根速 p95 区间 (m/s) | 腾空帧占比 | 脚底最低高度 |
|---|---|---|---|---|---|
| stand | 2 | 0.3 min | 0.02 – 0.13 | 0.0% – 0.0% | 0.0 – 0.0 cm |
| turn | 19 | 2.6 min | 0.79 – 1.37 | 0.0% – 89.7% | -0.0 – 0.1 cm |
| backward | 5 | 0.7 min | 0.72 – 1.20 | 0.0% – 83.6% | 0.0 – 0.0 cm |
| run_transition | 7 | 0.6 min | 2.17 – 3.21 | 9.1% – 40.2% | 0.0 – 0.1 cm |
| start_stop | 5 | 0.4 min | 0.82 – 1.65 | 0.0% – 76.0% | 0.0 – 0.1 cm |
| walk_forward | 14 | 48.4 min | 0.58 – 2.20 | 6.2% – 99.7% | -0.0 – 0.3 cm |
| run | 11 | 25.4 min | 2.46 – 4.70 | 20.0% – 100.0% | -0.0 – 0.4 cm |
| jump_fall | 16 | 47.4 min | 0.64 – 2.97 | 81.4% – 100.0% | -0.0 – 0.7 cm |
| dance_fight_other | 26 | 92.3 min | 0.80 – 3.12 | 89.5% – 100.0% | -0.0 – 1.0 cm |
| unclassified | 11 | 41.5 min | 1.09 – 2.70 | 91.5% – 99.9% | -0.0 – 0.3 cm |

**walk-only 候选：45 条 / 52.4 min**（全量 116 条 / 259.5 min，占 20%）

## walk-only 明细

| 数据集 | clip | 类别 | 时长 | v_mean | v_p95 | v_max | 腾空% | 脚底最低 |
|---|---|---|---|---|---|---|---|---|
| g1_ase_45 | `A1_stand_g1_29` | stand | 3.0s | 0.01 | 0.02 | 0.06 | 0.0 | +0.0 cm |
| g1_ase_45 | `A2_sway_g1_29` | stand | 12.0s | 0.05 | 0.13 | 0.18 | 0.0 | +0.0 cm |
| g1_ase_45 | `Female1_B10_walk_turn_left_45_g1_29` | turn | 7.9s | 0.73 | 1.03 | 1.08 | 0.0 | +0.1 cm |
| g1_ase_45 | `Female1_B11_walk_turn_left_135_g1_29` | turn | 9.4s | 0.64 | 1.02 | 1.08 | 0.0 | +0.0 cm |
| g1_ase_45 | `Female1_B13_walk_turn_right_45_g1_29` | turn | 8.3s | 0.76 | 1.04 | 1.09 | 42.3 | +0.0 cm |
| g1_ase_45 | `Female1_B14_walk_turn_right_135_g1_29` | turn | 9.0s | 0.71 | 1.03 | 1.09 | 0.0 | +0.0 cm |
| g1_ase_45 | `Female1_B15_walk_turn_around_g1_29` | turn | 10.6s | 0.62 | 0.98 | 1.07 | 0.0 | +0.0 cm |
| g1_ase_45 | `Female1_B16_walk_turn_change_direction_g1_29` | turn | 11.0s | 0.55 | 1.03 | 1.08 | 10.9 | +0.0 cm |
| g1_ase_45 | `Female1_B7_walk_backwards_turn_forwards_g1_29` | turn | 8.7s | 0.72 | 1.05 | 1.14 | 31.3 | +0.0 cm |
| g1_ase_45 | `Female1_B9_walk_turn_left_90_g1_29` | turn | 7.1s | 0.65 | 0.98 | 1.06 | 0.0 | +0.0 cm |
| g1_ase_45 | `Male1_Walk_B10_turn_left_45_g1_29` | turn | 8.4s | 0.74 | 0.97 | 1.05 | 75.4 | +0.0 cm |
| g1_ase_45 | `Male1_Walk_B13_turn_right_45_g1_29` | turn | 9.3s | 0.70 | 1.01 | 1.07 | 89.7 | +0.0 cm |
| g1_ase_45 | `Male1_Walk_B15_turn_around_g1_29` | turn | 11.4s | 0.61 | 0.99 | 1.06 | 68.8 | -0.0 cm |
| g1_ase_45 | `Male1_Walk_B16_turn_change_g1_29` | turn | 12.1s | 0.52 | 0.93 | 1.01 | 77.9 | -0.0 cm |
| g1_ase_45 | `Male2_B10_turn_left_45_g1_29` | turn | 6.0s | 0.85 | 1.22 | 1.37 | 9.3 | +0.0 cm |
| g1_ase_45 | `Male2_B11_turn_left_135_g1_29` | turn | 6.5s | 0.88 | 1.30 | 1.41 | 2.5 | +0.0 cm |
| g1_ase_45 | `Male2_B13_turn_right_90_g1_29` | turn | 5.6s | 0.95 | 1.33 | 1.40 | 87.2 | +0.0 cm |
| g1_ase_45 | `Male2_B14_turn_right_45_g1_29` | turn | 4.1s | 0.19 | 0.79 | 0.99 | 0.0 | +0.0 cm |
| g1_ase_45 | `Male2_B14_turn_right_45_t2_g1_29` | turn | 5.8s | 0.99 | 1.37 | 1.44 | 42.1 | +0.0 cm |
| g1_ase_45 | `Male2_B15_turn_around_g1_29` | turn | 7.5s | 0.84 | 1.35 | 1.46 | 27.7 | +0.0 cm |
| g1_ase_45 | `Male2_B9_turn_left_90_g1_29` | turn | 5.0s | 0.87 | 1.18 | 1.32 | 0.8 | +0.0 cm |
| g1_ase_45 | `Female1_B4_stand_to_walk_back_g1_29` | backward | 7.6s | 0.39 | 0.72 | 0.79 | 0.0 | +0.0 cm |
| g1_ase_45 | `Female1_B5_walk_backwards_g1_29` | backward | 11.5s | 0.56 | 0.94 | 0.99 | 42.9 | +0.0 cm |
| g1_ase_45 | `Female1_B6_walk_backwards_to_stand_g1_29` | backward | 8.2s | 0.35 | 0.73 | 0.81 | 0.0 | +0.0 cm |
| g1_ase_45 | `Male1_Walk_B4_stand_to_walk_back_g1_29` | backward | 13.0s | 0.53 | 0.82 | 0.88 | 83.6 | +0.0 cm |
| g1_ase_45 | `Male2_B4_stand_to_walk_backwards_g1_29` | backward | 4.6s | 0.78 | 1.20 | 1.27 | 4.3 | +0.0 cm |
| g1_ase_45 | `Female1_A14_stand_to_skip_g1_29` | start_stop | 5.2s | 0.79 | 1.50 | 1.68 | 76.0 | +0.1 cm |
| g1_ase_45 | `Female1_A15_skip_to_stand_g1_29` | start_stop | 4.3s | 0.78 | 1.65 | 1.86 | 18.7 | +0.1 cm |
| g1_ase_45 | `Female1_B1_stand_to_walk_g1_29` | start_stop | 6.2s | 0.46 | 0.82 | 0.87 | 0.0 | +0.0 cm |
| g1_ase_45 | `Female1_B2_walk_to_stand_g1_29` | start_stop | 5.7s | 0.50 | 0.88 | 0.97 | 0.0 | +0.0 cm |
| g1_ase_45 | `Female1_B2_walk_to_stand_t2_g1_29` | start_stop | 4.9s | 0.54 | 0.92 | 1.00 | 0.0 | +0.0 cm |
| g1_ase_45 | `Female1_B3_walk1_g1_29` | walk_forward | 7.6s | 0.82 | 1.02 | 1.06 | 39.8 | +0.0 cm |
| g1_ase_45 | `S007_qk_walk1_g1_29` | walk_forward | 2.6s | 1.64 | 2.20 | 2.29 | 6.2 | +0.0 cm |
| lafan1_g1 | `walk1_subject1` | walk_forward | 261.3s | 0.60 | 1.51 | 1.87 | 84.0 | -0.0 cm |
| lafan1_g1 | `walk1_subject2` | walk_forward | 261.3s | 0.92 | 1.62 | 2.17 | 91.1 | +0.3 cm |
| lafan1_g1 | `walk1_subject5` | walk_forward | 261.3s | 0.92 | 1.53 | 1.84 | 99.5 | +0.0 cm |
| lafan1_g1 | `walk2_subject1` | walk_forward | 238.2s | 0.66 | 1.65 | 3.15 | 99.7 | -0.0 cm |
| lafan1_g1 | `walk2_subject3` | walk_forward | 238.2s | 0.57 | 1.67 | 3.38 | 98.7 | +0.0 cm |
| lafan1_g1 | `walk2_subject4` | walk_forward | 238.2s | 0.59 | 1.32 | 2.97 | 86.5 | -0.0 cm |
| lafan1_g1 | `walk3_subject1` | walk_forward | 246.6s | 0.37 | 0.79 | 1.20 | 99.7 | +0.0 cm |
| lafan1_g1 | `walk3_subject2` | walk_forward | 246.6s | 0.75 | 1.88 | 2.93 | 99.7 | -0.0 cm |
| lafan1_g1 | `walk3_subject3` | walk_forward | 246.6s | 0.27 | 0.58 | 1.32 | 99.7 | +0.0 cm |
| lafan1_g1 | `walk3_subject4` | walk_forward | 246.6s | 0.60 | 1.29 | 4.01 | 99.5 | +0.0 cm |
| lafan1_g1 | `walk3_subject5` | walk_forward | 246.6s | 0.84 | 1.50 | 2.07 | 99.6 | +0.2 cm |
| lafan1_g1 | `walk4_subject1` | walk_forward | 163.9s | 0.39 | 0.76 | 1.05 | 99.7 | +0.0 cm |

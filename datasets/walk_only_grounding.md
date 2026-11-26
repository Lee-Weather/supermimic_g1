# walk-only 子集的接地质量（mjlab 模型实测）

判定：中位脚底高度 > 2.0 cm **或** 触地帧占比 < 10% 即判为有问题。

**13 / 45 条有问题**（占比 29%）

| 类别 | 总数 | 有问题 | 中位脚底 cm | 触地率 |
|---|---|---|---|---|
| turn | 19 | 0 | 0.0 | 0.0% |
| walk_forward | 14 | 12 | 5.1 | 0.4% |
| start_stop | 5 | 1 | 3.7 | 24.0% |
| backward | 5 | 0 | 0.0 | 0.0% |
| stand | 2 | 0 | 0.0 | 0.0% |

## 逐条明细

| 数据集 | clip | 类别 | 时长 | 脚底 p10 | p50 | p90 | 触地率 | 根高 std | 判定 |
|---|---|---|---|---|---|---|---|---|---|
| g1_ase_45 | `Male2_B4_stand_to_walk_backwards_g1_29` | backward | 4.6s | +0.3 | +1.0 | +1.8 | 95.7% | 1.7 cm | ✅ |
| g1_ase_45 | `Male1_Walk_B4_stand_to_walk_back_g1_29` | backward | 13.0s | +0.0 | +0.4 | +1.7 | 95.4% | 0.9 cm | ✅ |
| g1_ase_45 | `Female1_B5_walk_backwards_g1_29` | backward | 11.5s | +0.0 | +0.3 | +0.8 | 100.0% | 1.3 cm | ✅ |
| g1_ase_45 | `Female1_B6_walk_backwards_to_stand_g1_29` | backward | 8.2s | +0.0 | +0.3 | +0.7 | 100.0% | 1.3 cm | ✅ |
| g1_ase_45 | `Female1_B4_stand_to_walk_back_g1_29` | backward | 7.6s | +0.0 | +0.1 | +0.3 | 100.0% | 1.3 cm | ✅ |
| g1_ase_45 | `A2_sway_g1_29` | stand | 12.0s | +0.0 | +0.0 | +0.1 | 100.0% | 1.1 cm | ✅ |
| g1_ase_45 | `A1_stand_g1_29` | stand | 3.0s | +0.0 | +0.0 | +0.0 | 100.0% | 0.0 cm | ✅ |
| g1_ase_45 | `Female1_A14_stand_to_skip_g1_29` | start_stop | 5.2s | +0.9 | +3.7 | +4.2 | 24.0% | 4.6 cm | ❌ |
| g1_ase_45 | `Female1_A15_skip_to_stand_g1_29` | start_stop | 4.3s | +0.2 | +1.8 | +2.8 | 81.3% | 5.0 cm | ✅ |
| g1_ase_45 | `Female1_B1_stand_to_walk_g1_29` | start_stop | 6.2s | +0.1 | +0.4 | +0.9 | 100.0% | 1.6 cm | ✅ |
| g1_ase_45 | `Female1_B2_walk_to_stand_g1_29` | start_stop | 5.7s | +0.0 | +0.1 | +0.5 | 100.0% | 1.4 cm | ✅ |
| g1_ase_45 | `Female1_B2_walk_to_stand_t2_g1_29` | start_stop | 4.9s | +0.0 | +0.1 | +0.4 | 100.0% | 1.6 cm | ✅ |
| g1_ase_45 | `Female1_B16_walk_turn_change_direction_g1_29` | turn | 11.0s | +0.1 | +0.8 | +2.1 | 89.1% | 1.6 cm | ✅ |
| g1_ase_45 | `Male1_Walk_B13_turn_right_45_g1_29` | turn | 9.3s | +0.1 | +0.5 | +1.7 | 97.4% | 0.9 cm | ✅ |
| g1_ase_45 | `Male1_Walk_B15_turn_around_g1_29` | turn | 11.4s | +0.1 | +0.4 | +1.6 | 95.4% | 1.0 cm | ✅ |
| g1_ase_45 | `Male1_Walk_B10_turn_left_45_g1_29` | turn | 8.4s | +0.1 | +0.4 | +1.8 | 92.9% | 0.9 cm | ✅ |
| g1_ase_45 | `Male2_B13_turn_right_90_g1_29` | turn | 5.6s | +0.1 | +0.4 | +1.1 | 98.6% | 1.6 cm | ✅ |
| g1_ase_45 | `Male2_B15_turn_around_g1_29` | turn | 7.5s | +0.1 | +0.4 | +1.2 | 98.9% | 1.6 cm | ✅ |
| g1_ase_45 | `Male1_Walk_B16_turn_change_g1_29` | turn | 12.1s | +0.1 | +0.4 | +1.7 | 95.0% | 0.8 cm | ✅ |
| g1_ase_45 | `Male2_B14_turn_right_45_t2_g1_29` | turn | 5.8s | +0.1 | +0.3 | +1.4 | 97.2% | 1.6 cm | ✅ |
| g1_ase_45 | `Female1_B13_walk_turn_right_45_g1_29` | turn | 8.3s | +0.0 | +0.3 | +0.7 | 100.0% | 1.6 cm | ✅ |
| g1_ase_45 | `Male2_B11_turn_left_135_g1_29` | turn | 6.5s | +0.0 | +0.2 | +1.2 | 97.5% | 1.5 cm | ✅ |
| g1_ase_45 | `Female1_B7_walk_backwards_turn_forwards_g1_29` | turn | 8.7s | +0.0 | +0.2 | +0.9 | 100.0% | 1.6 cm | ✅ |
| g1_ase_45 | `Male2_B10_turn_left_45_g1_29` | turn | 6.0s | +0.0 | +0.2 | +1.8 | 90.7% | 1.6 cm | ✅ |
| g1_ase_45 | `Female1_B10_walk_turn_left_45_g1_29` | turn | 7.9s | +0.1 | +0.2 | +0.6 | 100.0% | 1.7 cm | ✅ |
| g1_ase_45 | `Male2_B9_turn_left_90_g1_29` | turn | 5.0s | +0.0 | +0.2 | +1.0 | 99.2% | 1.3 cm | ✅ |
| g1_ase_45 | `Female1_B11_walk_turn_left_135_g1_29` | turn | 9.4s | +0.1 | +0.2 | +0.7 | 100.0% | 1.6 cm | ✅ |
| g1_ase_45 | `Female1_B14_walk_turn_right_135_g1_29` | turn | 9.0s | +0.1 | +0.2 | +0.6 | 100.0% | 1.5 cm | ✅ |
| g1_ase_45 | `Female1_B9_walk_turn_left_90_g1_29` | turn | 7.1s | +0.1 | +0.1 | +0.5 | 100.0% | 1.5 cm | ✅ |
| g1_ase_45 | `Female1_B15_walk_turn_around_g1_29` | turn | 10.6s | +0.1 | +0.1 | +0.5 | 100.0% | 1.4 cm | ✅ |
| g1_ase_45 | `Male2_B14_turn_right_45_g1_29` | turn | 4.1s | +0.0 | +0.0 | +0.1 | 100.0% | 0.7 cm | ✅ |
| g1_ase_45 | `S007_qk_walk1_g1_29` | walk_forward | 2.6s | +0.0 | +0.4 | +1.6 | 93.8% | 1.4 cm | ✅ |
| g1_ase_45 | `Female1_B3_walk1_g1_29` | walk_forward | 7.6s | +0.0 | +0.3 | +0.9 | 100.0% | 1.5 cm | ✅ |
| lafan1_g1 | `walk3_subject1` | walk_forward | 246.6s | +8.6 | +9.6 | +10.7 | 0.3% | 18.9 cm | ❌ |
| lafan1_g1 | `walk2_subject1` | walk_forward | 238.2s | +7.3 | +8.4 | +9.5 | 0.3% | 6.4 cm | ❌ |
| lafan1_g1 | `walk3_subject4` | walk_forward | 246.6s | +6.4 | +7.5 | +8.7 | 0.5% | 21.5 cm | ❌ |
| lafan1_g1 | `walk2_subject3` | walk_forward | 238.2s | +5.1 | +6.3 | +7.7 | 1.3% | 11.4 cm | ❌ |
| lafan1_g1 | `walk3_subject3` | walk_forward | 246.6s | +4.9 | +5.9 | +10.5 | 0.3% | 18.1 cm | ❌ |
| lafan1_g1 | `walk4_subject1` | walk_forward | 163.9s | +4.6 | +5.2 | +5.7 | 0.3% | 2.1 cm | ❌ |
| lafan1_g1 | `walk3_subject5` | walk_forward | 246.6s | +4.0 | +4.9 | +5.7 | 0.4% | 3.7 cm | ❌ |
| lafan1_g1 | `walk3_subject2` | walk_forward | 246.6s | +3.2 | +4.0 | +4.9 | 0.3% | 2.1 cm | ❌ |
| lafan1_g1 | `walk1_subject5` | walk_forward | 261.3s | +3.2 | +3.8 | +4.9 | 0.5% | 3.1 cm | ❌ |
| lafan1_g1 | `walk2_subject4` | walk_forward | 238.2s | +1.9 | +2.8 | +3.9 | 13.5% | 4.1 cm | ❌ |
| lafan1_g1 | `walk1_subject2` | walk_forward | 261.3s | +2.0 | +2.7 | +3.6 | 8.9% | 2.2 cm | ❌ |
| lafan1_g1 | `walk1_subject1` | walk_forward | 261.3s | +1.8 | +2.6 | +3.3 | 16.0% | 2.8 cm | ❌ |

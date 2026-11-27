# G1 速度指令回放（云端离屏渲染）

`models/g1_walk_velocity_30k_20261126`（motion 0.5 / velocity 0.5，4096 envs × 30k 步）的**速度指令行为验证**。
在 flux 云 pod 上用 EGL 离屏渲染、`mediapy` 写 mp4，50 fps、640×480。

## 视频

| 文件 | 指令 `[vx, vy, wz]` | 时长 | 实测 `mean_speed` | 位移 | 位置误差 |
|---|---|---|---|---|---|
| `01_stand_0.0ms.mp4` | `0, 0, 0` | 6 s (300 步) | **0.025 m/s** | 0.09 m | — |
| `02_walk_0.3ms.mp4` | `0.3, 0, 0` | 8 s (400 步) | **0.280 m/s** | 2.06 m | −6.7% |
| `03_walk_0.8ms.mp4` | `0.8, 0, 0` | 8 s (400 步) | **0.815 m/s** | 4.66 m（6 s 段） | **+1.9%** |
| `04_walk0.3_turn_1.0rads.mp4` | `0.3, 0, 1.0` | 6 s (300 步) | **0.313 m/s** | 0.04 m | 边走边原地转，半径 `v/w ≈ 0.3 m`，转一圈回到原点 |

`mean_speed` 由 `MotionCommand.robot_anchor_pos_w` 逐帧位移 / 时间（50 Hz 控制）算出，不是指令值。

## 结论

- **站立**：指令为 0 时几乎不动（0.025 m/s，属于模型/控制器抖动）；
- **前进**：0.3 m/s 指令实测 0.28（−6.7%）、0.8 m/s 指令实测 **0.815（+1.9%）** —— **速度指令接口可用**；
- **转向**：`wz=1.0 rad/s` 且 `vx=0.3` 时走出半径约 0.3 m 的圆弧，6 s 后回到起点附近（位移 0.04 m 正确反映"绕圈"而非"没动"）。

## 复现方式

任务 `TASK_20261126_508`（训练）、回放任务 `TASK_20261127_106/119/130`（三次迭代修 bug 后成功）：

```bash
gm-run supermimic_g1/cloud/launch.py --replay \
  --model-dir models/g1_walk_velocity_30k_20261126 \
  --replay-task velocity --replay-steps 400 --flush-sleep 120 \
  --velocity-script 0.3,0,0:0.8,0,0 \
  --velocity-mix 0.5 --vel-range conservative \
  --motion-file datasets/walk_only/packed
```

三个踩过的坑（都已修在 `replay_headless.py`）：

1. `--velocity-script` 用 `:` 分隔，**不能用 `;`** —— 平台经 shell 执行，`;` 会截断参数（只剩第一段）；
2. mp4 要**镜像成 `.pt`** 才能被平台文件监听器上传（我们上一轮就用这个通道取回 `.mp4`/`.pt` 产物）；
3. 多段回放结束时必须 **`--flush-sleep 120`** 空转等上传队列排空，否则最后几段的 mp4 会随任务退出被丢弃。
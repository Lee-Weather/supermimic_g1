"""Headless replay of a SuperMimic actor, recording mp4 segments on a cloud pod.

Why this exists: ``scripts/play.py`` downloads weights from a wandb run and opens a GUI
viewer, neither of which works on a pod (it is built with ``WANDB_MODE=offline`` and has no
display). This script loads the local ``world_model.pt`` / ``actor.pt`` and renders offscreen
with EGL, so a cloud task can produce reviewable videos plus a metrics CSV.

Usage (inside the venv, after cloud/bootstrap.sh):

    MUJOCO_GL=egl python -u supermimic/src/supermimic/scripts/replay_headless.py \
      --model-dir models/g1_walk_velocity_30k_20261126 \
      --task motion --motion-file datasets/walk_only/packed \
      --clips 3 --steps 500 --out-dir replay_out

    # velocity: one segment per command in --velocity-script "vx,vy,wz;..."
    ... --task velocity --velocity-script "0.0,0,0;0.3,0,0;0.8,0,0;0.3,0,1.0"
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402
import torch  # noqa: E402

import mjlab.tasks  # noqa: F401,E402
import supermimic.tasks  # noqa: F401,E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_env_cfg  # noqa: E402
from mjlab.utils.torch import configure_torch_backends  # noqa: E402

from supermimic.runners.supermimic_runner import (  # noqa: E402
  SuperMimicRunnerCfg,
  build_actor,
  build_world_model,
)
from supermimic.scripts.viewers import SuperMimicViewerMixin  # noqa: E402
from supermimic.tasks.task_spec import build_actor_obs_td  # noqa: E402


class HeadlessReplay(SuperMimicViewerMixin):
  """The mixin's ground-truth rollout loop, minus any GUI, plus video capture."""

  def __init__(self, env, policy) -> None:
    self.env = env
    self.policy = policy
    self._gt_history: torch.Tensor | None = None

  def _step_once(self):
    obs = self.env.get_observations()
    curr_gt_states, curr_goals = obs["full_state"], obs["goal"]

    if self.curr_wm_aux is None:
      self.curr_wm_aux = self.representation.extract_aux_state(curr_gt_states)
      self.curr_pred_rep_state = self.representation.represent(curr_gt_states)
      self.sim_ctx = self.sim_context.read(self.env.unwrapped)
      self._gt_history = None
    if self.last_actions is None:
      self.last_actions = torch.zeros(
        (curr_gt_states.shape[0], self.dynamics.num_actions),
        device=curr_gt_states.device,
      )

    gt_aux = self.representation.extract_aux_state(curr_gt_states)
    actor_goal = self._build_actor_goal(curr_goals, gt_aux)
    gt_rep_state = self.representation.represent(curr_gt_states)
    proprio = self.representation.actor_obs_mode == "proprio"
    sensor = self.representation.compute_actor_sensor(
      gt_rep_state,
      self.sim_ctx,
      sim_actor_obs=obs["actor"] if proprio else None,
    )
    self._gt_history = self._advance_history(self._gt_history, sensor)
    actor_obs = build_actor_obs_td({self.current_task: actor_goal}, self._gt_history)
    actions = self.policy(actor_obs)

    _, _, dones, _ = self.env.step(actions)
    self.last_actions = actions.clone().detach()
    if bool(torch.any(dones)):
      self.reset_env_supermimic()
      self._gt_history = None
    return actor_goal, actions


  def deploy_rep_state(self) -> "np.ndarray":
    """``[z, g_b, v_b, w_b, qj, vj]`` built the way ``supermimic/core.h`` builds it.

    Uses the env's ground-truth robot state (pelvis pose/velocity + joint encoders), so it
    exercises the deploy math rather than the training-time representation modules.
    """
    mcmd = self.env.unwrapped.command_manager.get_term("motion")
    quat = mcmd.robot_anchor_quat_w[0]
    world_down = torch.zeros(3, device=quat.device)
    world_down[2] = -1.0
    g_b = _quat_rot_inv(quat, world_down)
    v_b = _quat_rot_inv(quat, mcmd.robot_anchor_lin_vel_w[0])
    w_b = _quat_rot_inv(quat, mcmd.robot_anchor_ang_vel_w[0])
    return np.concatenate([
      [float(mcmd.robot_anchor_pos_w[0, 2])],
      g_b.cpu().numpy(),
      v_b.cpu().numpy(),
      w_b.cpu().numpy(),
      mcmd.robot_joint_pos[0].cpu().numpy(),
      mcmd.robot_joint_vel[0].cpu().numpy(),
    ]).astype(np.float32)

  def deploy_step(self, session, cmd) -> "np.ndarray":
    """One 20 ms control step driven by the exported ONNX policy."""
    obs = np.concatenate([np.asarray(cmd, dtype=np.float32), self.deploy_rep_state()])[None, :]
    action = session.run(["actions"], {"obs": obs})[0][0]
    actions = torch.as_tensor(
      action, dtype=torch.float32, device=self.env.unwrapped.device
    )[None]
    _, _, dones, _ = self.env.step(actions)
    if bool(torch.any(dones)):
      self.reset_env_supermimic()
    return actions


def _quat_rot_inv(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
  """Rotate a world vector into the body frame (inverse rotation, wxyz quat)."""
  w, x, y, z = q[0], q[1], q[2], q[3]
  tmp = 2.0 * torch.stack([
    y * v[2] - z * v[1],
    z * v[0] - x * v[2],
    x * v[1] - y * v[0],
  ])
  return v - w * tmp + torch.stack([
    y * tmp[2] - z * tmp[1],
    z * tmp[0] - x * tmp[2],
    x * tmp[1] - y * tmp[0],
  ])


def _text_strip(lines: list[str], scale: int = 2):
  """Render lines into a small mask, then upscale so the bitmap font is readable."""
  from PIL import Image, ImageDraw, ImageFont

  font = ImageFont.load_default()
  probe = ImageDraw.Draw(Image.new("L", (1, 1)))
  metas = []
  for line in lines:
    box = probe.textbbox((0, 0), line, font=font)
    metas.append((line, box[2] - box[0], box[3] - box[1]))
  width = max(m[1] for m in metas) + 6
  height = sum(m[2] + 5 for m in metas) + 3
  strip = Image.new("L", (width, height), 0)
  draw = ImageDraw.Draw(strip)
  y = 2
  for line, _, hh in metas:
    draw.text((3, y), line, fill=255, font=font)
    y += hh + 5
  return strip.resize((width * scale, height * scale), Image.NEAREST)


def _draw_overlay(frame: np.ndarray, lines: list[str], scale: int = 2) -> np.ndarray:
  """Blit a dark banner with white text into the top-left of the frame."""
  try:
    mask = np.asarray(_text_strip(lines, scale), dtype=np.float32) / 255.0
  except Exception as exc:  # pillow missing or font problem: keep the video, lose the text
    print(f"[replay] overlay skipped: {exc}", flush=True)
    return frame
  h, w = mask.shape
  region = frame[:h, :w].astype(np.float32)
  shaded = region * 0.25
  frame[:h, :w] = np.where(mask[..., None] > 0.5, 235.0, shaded).astype(np.uint8)
  return frame


def _render(env) -> np.ndarray | None:
  frame = env.unwrapped.render()
  if frame is None:
    return None
  return np.asarray(frame, dtype=np.uint8)


def parse_velocity_script(text: str) -> list[tuple[float, float, float]]:
  # ':' is the shell-safe separator; ';' is accepted for local use but is stripped by shells.
  out = []
  for chunk in text.replace(";", ":").split(":"):
    chunk = chunk.strip()
    if not chunk:
      continue
    parts = [float(p) for p in chunk.split(",")]
    if len(parts) != 3:
      raise ValueError(f"velocity item must be 'vx,vy,wz': {chunk!r}")
    out.append((parts[0], parts[1], parts[2]))
  return out or [(0.0, 0.0, 0.0)]


def parse_ramp(text: str) -> list[tuple[int, tuple[float, float, float]]]:
  """'100@0,0,0:150@1.1,0,0' -> [(100, (0,0,0)), (150, (1.1,0,0))]."""
  phases: list[tuple[int, tuple[float, float, float]]] = []
  for tok in text.split(":"):
    tok = tok.strip()
    if not tok:
      continue
    steps_txt, vals_txt = tok.split("@", 1)
    vals = [float(v) for v in vals_txt.split(",")]
    if len(vals) != 3:
      raise ValueError(f"ramp phase must be '<steps>@vx,vy,wz': {tok!r}")
    phases.append((int(steps_txt), (vals[0], vals[1], vals[2])))
  return phases


def run_ramp(env, player, args, out_dir, mediapy) -> int:
  import csv as _csv
  import shutil as _shutil

  phases = parse_ramp(args.velocity_ramp)
  env.reset()
  player.reset_env_supermimic()
  player._gt_history = None

  mcmd = env.unwrapped.command_manager.get_term("motion")
  frames: list[np.ndarray] = []
  phase_rows: list[dict] = []
  prev = (0.0, 0.0, 0.0)

  p_idx_steps = 0
  for p_idx, (steps, target) in enumerate(phases):
    positions = []
    for i in range(steps):
      # linear ramp from the previous phase target to this one
      frac = (i + 1) / steps
      cmd = tuple(prev[j] + (target[j] - prev[j]) * frac for j in range(3))
      for axis in range(3):
        player.set_raw_goal(player.current_task, axis, cmd[axis])
      _, _ = player._step_once()
      frame = _render(env)
      if frame is not None:
        if args.overlay:
          lin_b = _quat_rot_inv(mcmd.robot_anchor_quat_w[0], mcmd.robot_anchor_lin_vel_w[0])
          ang_b = _quat_rot_inv(mcmd.robot_anchor_quat_w[0], mcmd.robot_anchor_ang_vel_w[0])
          elapsed = (p_idx_steps + i + 1) / 50.0
          frame = _draw_overlay(frame, [
            f"CMD  vx {cmd[0]:+.2f}  vy {cmd[1]:+.2f}  wz {cmd[2]:+.2f}",
            f"ACT  vx {float(lin_b[0]):+.2f}  vy {float(lin_b[1]):+.2f}  "
            f"wz {float(ang_b[2]):+.2f}",
            f"phase {p_idx + 1}/{len(phases)}   t {elapsed:5.1f}s",
          ])
        frames.append(frame)
      positions.append(mcmd.robot_anchor_pos_w[0, :2].detach().cpu().numpy())

    arr = np.stack(positions)
    deltas = np.linalg.norm(np.diff(arr, axis=0), axis=1) if len(arr) > 1 else np.zeros(1)
    duration = max((len(arr) - 1) * 0.02, 1e-6)
    mean_speed = float(deltas.sum() / duration)
    displacement = float(np.linalg.norm(arr[-1] - arr[0])) if len(arr) > 1 else 0.0
    cmd_txt = f"{target[0]:+.2f},{target[1]:+.2f},{target[2]:+.2f}"
    print(
      f"[replay] phase{p_idx:02d} cmd_target={cmd_txt} steps={steps} "
      f"mean_speed={mean_speed:.3f} m/s disp={displacement:.2f} m",
      flush=True,
    )
    phase_rows.append({"phase": p_idx, "steps": steps, "cmd_target": cmd_txt,
                       "mean_speed_mps": f"{mean_speed:.3f}",
                       "displacement_m": f"{displacement:.2f}"})
    p_idx_steps += steps
    prev = target

  video = out_dir / "joystick_sweep.mp4"
  if frames:
    mediapy.write_video(str(video), frames, fps=50)
    _shutil.copy2(video, out_dir / "joystick_sweep_mp4.pt")
    mediapy.write_image(str(out_dir / "joystick_sweep_mid.png"), frames[len(frames) // 2])
    _shutil.copy2(out_dir / "joystick_sweep_mid.png", out_dir / "joystick_sweep_mid_png.pt")
    print(f"[replay] wrote {video} ({len(frames)} frames = {len(frames)/50:.1f}s)", flush=True)

  with (out_dir / "joystick_sweep_phases.csv").open("w", newline="", encoding="utf-8") as fh:
    w = _csv.DictWriter(fh, fieldnames=list(phase_rows[0]))
    w.writeheader()
    w.writerows(phase_rows)
  print("[replay] ramp done", flush=True)
  return 0


def run_deploy(env, player, args, out_dir, mediapy) -> int:
  """Ramp-mode rollout where the policy is the exported ONNX artifact."""
  import csv as _csv
  import shutil as _shutil

  import onnxruntime as _ort

  session = _ort.InferenceSession(args.deploy_onnx, providers=["CPUExecutionProvider"])
  meta = session.get_inputs()[0]
  print(
    f"[replay] deploy ONNX {Path(args.deploy_onnx).name} "
    f"input {meta.name}{tuple(meta.shape)}",
    flush=True,
  )

  phases = parse_ramp(args.velocity_ramp)
  env.reset()
  player.reset_env_supermimic()
  player._gt_history = None

  mcmd = env.unwrapped.command_manager.get_term("motion")
  frames: list[np.ndarray] = []
  rows: list[dict] = []
  prev = (0.0, 0.0, 0.0)
  step_total = 0

  for p_idx, (steps, target) in enumerate(phases):
    positions = []
    for i in range(steps):
      frac = (i + 1) / steps
      cmd = tuple(prev[j] + (target[j] - prev[j]) * frac for j in range(3))
      player.deploy_step(session, cmd)
      step_total += 1
      positions.append(mcmd.robot_anchor_pos_w[0, :2].detach().cpu().numpy())
      frame = _render(env)
      if frame is not None:
        if args.overlay:
          quat = mcmd.robot_anchor_quat_w[0]
          lin_b = _quat_rot_inv(quat, mcmd.robot_anchor_lin_vel_w[0]).cpu().numpy()
          ang_b = _quat_rot_inv(quat, mcmd.robot_anchor_ang_vel_w[0]).cpu().numpy()
          frame = _draw_overlay(frame, [
            f"CMD  vx {cmd[0]:+.2f}  vy {cmd[1]:+.2f}  wz {cmd[2]:+.2f}",
            f"ACT  vx {lin_b[0]:+.2f}  vy {lin_b[1]:+.2f}  wz {ang_b[2]:+.2f}",
            f"phase {p_idx + 1}/{len(phases)}   t {step_total / 50.0:5.1f}s   "
            f"ONNX deploy path",
          ])
        frames.append(frame)

    arr = np.stack(positions)
    deltas = np.linalg.norm(np.diff(arr, axis=0), axis=1) if len(arr) > 1 else np.zeros(1)
    duration = max((len(arr) - 1) * 0.02, 1e-6)
    mean_speed = float(deltas.sum() / duration)
    displacement = float(np.linalg.norm(arr[-1] - arr[0])) if len(arr) > 1 else 0.0
    cmd_txt = f"{target[0]:+.2f},{target[1]:+.2f},{target[2]:+.2f}"
    print(
      f"[replay] phase{p_idx:02d} cmd_target={cmd_txt} steps={steps} "
      f"mean_speed={mean_speed:.3f} m/s disp={displacement:.2f} m",
      flush=True,
    )
    rows.append({"phase": p_idx, "steps": steps, "cmd_target": cmd_txt,
                 "mean_speed_mps": f"{mean_speed:.3f}",
                 "displacement_m": f"{displacement:.2f}"})
    prev = target

  video = out_dir / "deploy_onnx_mujoco.mp4"
  if frames:
    mediapy.write_video(str(video), frames, fps=50)
    _shutil.copy2(video, out_dir / "deploy_onnx_mujoco_mp4.pt")
    mediapy.write_image(str(out_dir / "deploy_onnx_mujoco_mid.png"), frames[len(frames) // 2])
    _shutil.copy2(out_dir / "deploy_onnx_mujoco_mid.png",
                  out_dir / "deploy_onnx_mujoco_mid_png.pt")
    print(f"[replay] wrote {video} ({len(frames)} frames = {len(frames) / 50:.1f}s)", flush=True)
  with (out_dir / "deploy_onnx_phases.csv").open("w", newline="", encoding="utf-8") as fh:
    w = _csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
  print("[replay] deploy rollout done", flush=True)
  return 0


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--task-id", default="SuperMimic-GenCoord-G1")
  ap.add_argument("--model-dir", required=True)
  ap.add_argument("--out-dir", default="replay_out")
  ap.add_argument("--task", choices=("motion", "velocity"), default="motion")
  ap.add_argument("--motion-file", default=None)
  ap.add_argument("--clips", type=int, default=1, help="how many motion clips to replay")
  ap.add_argument("--steps", type=int, default=500, help="steps per segment (50 Hz)")
  ap.add_argument("--velocity-script", default="0,0,0")
  ap.add_argument(
    "--velocity-ramp",
    default=None,
    help="joystick sweep: '<steps>@vx,vy,wz' phases joined by ':' , linearly interpolated "
    "between phase targets; records one long video plus per-phase measured speeds",
  )
  ap.add_argument("--overlay", action="store_true", default=True,
                  help="burn target/actual velocity text into the video")
  ap.add_argument("--no-overlay", dest="overlay", action="store_false")
  ap.add_argument("--deploy-onnx", default=None,
                  help="exported policy .onnx; runs the deploy obs/action path")
  ap.add_argument("--flush-sleep", type=float, default=0.0,
                  help="seconds to idle at the end so the uploader can drain its queue")
  ap.add_argument("--width", type=int, default=640)
  ap.add_argument("--height", type=int, default=480)
  args = ap.parse_args()

  configure_torch_backends()
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  out_dir = Path(args.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  env_cfg = load_env_cfg(args.task_id, play=True)
  env_cfg.scene.num_envs = 1
  env_cfg.observations["actor"].enable_corruption = True
  env_cfg.viewer.width = args.width
  env_cfg.viewer.height = args.height
  if args.motion_file:
    motion_cmd = env_cfg.commands["motion"]
    motion_cmd.motion_file = args.motion_file

  # The runner cfg's slot_mix / velocity ranges come from the same env vars training used
  # (SM_MIX_VELOCITY, SM_VEL_RANGE), so replay matches the trained network's slot layout.
  runner_cfg = SuperMimicRunnerCfg()
  print(
    f"[replay] slot_mix={runner_cfg.slot_mix} vel_range_env="
    f"{os.environ.get('SM_VEL_RANGE', 'wide')}",
    flush=True,
  )

  env = RslRlVecEnvWrapper(
    ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
  )
  env.reset()

  wm = build_world_model(runner_cfg, env, device)
  wm.load_bundle(torch.load(Path(args.model_dir) / "world_model.pt", map_location=device))
  actor = build_actor(runner_cfg, wm, device)
  actor.load_state_dict(torch.load(Path(args.model_dir) / "actor.pt", map_location=device))
  actor.eval()
  wm.dynamics.eval()

  player = HeadlessReplay(env, actor)
  player.setup_supermimic(
    dynamics=wm.dynamics,
    representation=wm.representation,
    num_steps_rollout=args.steps,
    sim_context=wm.sim_context,
    task_specs=actor.task_specs,
    initial_task=args.task,
  )

  import mediapy

  if args.deploy_onnx:
    code = run_deploy(env, player, args, out_dir, mediapy)
    if args.flush_sleep > 0:
      import time

      print(f"[replay] idling {args.flush_sleep:.0f}s so uploads drain", flush=True)
      time.sleep(args.flush_sleep)
    return code

  if args.velocity_ramp:
    code = run_ramp(env, player, args, out_dir, mediapy)
    if args.flush_sleep > 0:
      import time

      print(f"[replay] idling {args.flush_sleep:.0f}s so uploads drain", flush=True)
      time.sleep(args.flush_sleep)
    return code

  rows: list[dict] = []
  if args.task == "motion":
    # The env samples clips from the (packed) motion file itself; each segment is a fresh
    # reset, so several segments cover several clips without reconfiguring the command.
    segments = [(f"motion_seg{i:02d}", None) for i in range(max(1, args.clips))]
  else:
    segments = [
      (f"vel_{i:02d}_{cmd[0]:+.2f}_{cmd[1]:+.2f}_{cmd[2]:+.2f}", cmd)
      for i, cmd in enumerate(parse_velocity_script(args.velocity_script))
    ]

  for name, cmd in segments:
    env.reset()
    player.reset_env_supermimic()
    player._gt_history = None
    if cmd is not None:
      for idx, value in enumerate(cmd):
        player.set_raw_goal(player.current_task, idx, value)
      print(f"[replay] segment {name}: cmd={cmd}", flush=True)

    mcmd = env.unwrapped.command_manager.get_term("motion")
    frames, positions = [], []
    for step in range(args.steps):
      _, _ = player._step_once()
      frame = _render(env)
      if frame is not None:
        frames.append(frame)
      positions.append(mcmd.robot_anchor_pos_w[0, :2].detach().cpu().numpy())
    video = out_dir / f"{name}.mp4"
    if frames:
      mediapy.write_video(str(video), frames, fps=50)
      # The platform's file watcher only uploads *.pt, so mirror the video under a .pt name:
      # we download that artifact and rename it back to .mp4 locally.
      shutil.copy2(video, out_dir / f"{name}_mp4.pt")
      mid = frames[len(frames) // 2]
      still = out_dir / f"{name}_mid.png"
      mediapy.write_image(str(still), mid)
      shutil.copy2(still, out_dir / f"{name}_mid_png.pt")
      print(f"[replay] wrote {video} ({len(frames)} frames)", flush=True)
    else:
      print("[replay] no frames captured (render returned None)", flush=True)
    mean_speed = 0.0
    displacement = 0.0
    if len(positions) > 1:
      arr = np.stack(positions)
      deltas = np.linalg.norm(np.diff(arr, axis=0), axis=1)
      duration = (len(arr) - 1) * 0.02  # 50 Hz control
      mean_speed = float(deltas.sum() / duration)
      displacement = float(np.linalg.norm(arr[-1] - arr[0]))
    cmd_txt = "motion" if cmd is None else f"{cmd[0]:+.2f},{cmd[1]:+.2f},{cmd[2]:+.2f}"
    print(
      f"[replay] {name}: cmd={cmd_txt} mean_speed={mean_speed:.3f} m/s "
      f"displacement={displacement:.2f} m",
      flush=True,
    )
    rows.append({"segment": name, "cmd": cmd_txt, "steps": args.steps,
                 "frames": len(frames),
                 "mean_speed_mps": f"{mean_speed:.3f}",
                 "displacement_m": f"{displacement:.2f}",
                 "video": video.name if frames else ""})

  with (out_dir / "replay_summary.csv").open("w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])) if rows else None
    if w:
      w.writeheader()
      w.writerows(rows)
  print(f"[replay] done -> {out_dir}", flush=True)
  if args.flush_sleep > 0:
    import time

    print(f"[replay] idling {args.flush_sleep:.0f}s so uploads drain", flush=True)
    time.sleep(args.flush_sleep)
  return 0


if __name__ == "__main__":
  sys.exit(main())
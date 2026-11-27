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


def _render(env) -> np.ndarray | None:
  frame = env.unwrapped.render()
  if frame is None:
    return None
  return np.asarray(frame, dtype=np.uint8)


def parse_velocity_script(text: str) -> list[tuple[float, float, float]]:
  out = []
  for chunk in text.split(";"):
    chunk = chunk.strip()
    if not chunk:
      continue
    parts = [float(p) for p in chunk.split(",")]
    if len(parts) != 3:
      raise ValueError(f"velocity item must be 'vx,vy,wz': {chunk!r}")
    out.append((parts[0], parts[1], parts[2]))
  return out or [(0.0, 0.0, 0.0)]


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

  rows: list[dict] = []
  if args.task == "motion":
    clips = sorted(Path(args.motion_file or "").glob("*.npz"))[: max(1, args.clips)]
    segments = [(f"clip{i:02d}", None) for i in range(max(1, len(clips)))]
  else:
    segments = [
      (f"vel_{i:02d}_{vx:+.2f}_{vy:+.2f}_{wz:+.2f}", cmd)
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

    frames, rewards, done_steps = [], [], 0
    for step in range(args.steps):
      _, _ = player._step_once()
      frame = _render(env)
      if frame is not None:
        frames.append(frame)
      if env.unwrapped.episode_length_buf is not None:
        done_steps = step
    video = out_dir / f"{name}.mp4"
    if frames:
      mediapy.write_video(str(video), frames, fps=50)
      # The platform's file watcher only uploads *.pt, so mirror the video under a .pt name:
      # we download that artifact and rename it back to .mp4 locally.
      shutil.copy2(video, out_dir / f"{name}_mp4.pt")
      print(f"[replay] wrote {video} ({len(frames)} frames)", flush=True)
    else:
      print("[replay] no frames captured (render returned None)", flush=True)
    rows.append({"segment": name, "steps": args.steps, "frames": len(frames),
                 "video": video.name if frames else ""})

  with (out_dir / "replay_summary.csv").open("w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])) if rows else None
    if w:
      w.writeheader()
      w.writerows(rows)
  print(f"[replay] done -> {out_dir}", flush=True)
  return 0


if __name__ == "__main__":
  sys.exit(main())
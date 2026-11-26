"""Render before/after replays for the flagged UMR walk-only clips.

Every flagged clip is rendered twice from the same camera:

  before   the clip as it currently sits in ``datasets/g1_ase_45/npz`` (foot hover)
  after    the same clip with rolling grounding applied in memory (no files touched):
           the root is lowered per frame by the local (0.3 s window) minimum sole height,
           measured on mjlab's G1 model

The overlay shows time, root height and the lowest sole height, so the hover is directly
readable in the video.  Nothing under ``datasets/`` is modified.

Usage::

    python render_flagged_umr.py --out ../../czy/dataset_previews_umr_flagged
"""

from __future__ import annotations

import argparse
import importlib.util
import os

os.environ.setdefault("MUJOCO_GL", "glfw")

from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent

FLAGGED = [
  ("Male1_Walk_B13_turn_right_45_g1_29", "turn · 脚底 5.0cm / 触地 10%"),
  ("Male1_Walk_B4_stand_to_walk_back_g1_29", "backward · 脚底 4.8cm / 触地 16%"),
  ("Female1_A14_stand_to_skip_g1_29", "start_stop · 起跳，腾空属真实动作"),
  ("Male1_Walk_B16_turn_change_g1_29", "turn · 脚底 3.6cm / 触地 22%"),
  ("Male1_Walk_B10_turn_left_45_g1_29", "turn · 脚底 3.5cm / 触地 25%"),
  ("Male2_B13_turn_right_90_g1_29", "turn · 脚底 3.5cm / 触地 13%"),
  ("Male1_Walk_B15_turn_around_g1_29", "turn · 脚底 3.1cm / 触地 31%"),
]


def load_tools():
  spec = importlib.util.spec_from_file_location("rp", HERE / "render_previews.py")
  render = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(render)
  spec2 = importlib.util.spec_from_file_location("gv", HERE / "compare_grounding_variants.py")
  grounding = importlib.util.module_from_spec(spec2)
  spec2.loader.exec_module(grounding)
  return render, grounding


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--out", required=True)
  ap.add_argument("--window", type=float, default=8.0)
  ap.add_argument("--fps", type=float, default=25.0)
  ap.add_argument("--width", type=int, default=768)
  ap.add_argument("--height", type=int, default=512)
  args = ap.parse_args()

  render, grounding = load_tools()
  out_dir = Path(args.out)
  out_dir.mkdir(parents=True, exist_ok=True)
  clip_dir = HERE.parent / "g1_ase_45" / "npz"

  model = render.build_model(args.width, args.height)
  renderer = mujoco.Renderer(model, height=args.height, width=args.width)
  # Same model, recoloured probe for sole heights (corrupts nothing on disk).
  probe = grounding.Probe()

  entries = []
  for stem, note in FLAGGED:
    clip_path = clip_dir / f"{stem}.npz"
    if not clip_path.exists():
      print(f"[skip] {clip_path} missing")
      continue
    base = grounding.load_clip(clip_path)
    fixed = grounding.rolling_ground(base, probe)
    fps_src = base["fps"]

    root_vel = np.gradient(base["root_pos_w"][:, :2], 1.0 / fps_src, axis=0)
    speed = np.linalg.norm(root_vel, axis=1)
    start, end = render.active_window(speed, fps_src, args.window)
    stride = max(1, int(round(fps_src / args.fps)))

    for tag, clip in (("before", base), ("after", fixed)):
      data = mujoco.MjData(model)
      cam = render.make_camera(model, 2.6)
      fnt = render.font(15)
      sole = probe.soles(grounding.qpos_of(clip))
      frames = []
      for t in range(start, end, stride):
        data.qpos[:3] = clip["root_pos_w"][t]
        data.qpos[3:7] = clip["root_quat_w"][t]
        data.qpos[7:] = clip["joint_pos"][t]
        mujoco.mj_forward(model, data)
        cam.lookat[:] = [clip["root_pos_w"][t, 0], clip["root_pos_w"][t, 1], 0.78]
        renderer.update_scene(data, camera=cam)
        frames.append(
          render.overlay(
            renderer.render(),
            f"{tag} :: {stem}",
            f"t={t / fps_src:5.2f}s  root_z={clip['root_pos_w'][t, 2]:.3f}m  "
            f"sole={sole[t]*100:+.1f}cm  |v|={speed[t]:.2f}m/s",
            fnt,
          )
        )
      name = f"{tag}__{stem}.mp4"
      imageio.mimsave(
        out_dir / name, frames, fps=args.fps, codec="libx264", quality=7,
        macro_block_size=1,
      )
      Image.fromarray(frames[len(frames) // 2]).save(
        out_dir / name.replace(".mp4", ".png")
      )
      entries.append((tag, stem, note, name, len(frames)))
      print(f"[ok] {name:<56} {len(frames):>4} frames", flush=True)

  lines = [
    "# UMR walk-only 接地问题片段的回放（修前/修后）",
    "",
    "渲染设置：mjlab 的 G1 模型 + 地面，跟随相机（lookat 跟随根节点、距离 2.6 m），25 fps，",
    "左上角叠加 时间 / 根高 / **最低脚底高度**——脚底那行在 `before` 里会稳定停在 3–5 cm，",
    "`after` 里应贴到 0–1 cm。`after` 只是内存里做了 rolling(0.3 s) 接地，**数据集文件未改动**。",
    "",
    "| 片段 | 说明 | 视频 |",
    "|---|---|---|",
  ]
  for tag, stem, note, name, _ in entries:
    if tag == "before":
      lines.append(f"| `{stem}` | {note} | [{name}]({name}) / [{name.replace('before','after')}]({name.replace('before','after')}) |")
  lines += [
    "",
    "## 怎么判读",
    "",
    "- 看**脚底那行数字**而不是只看画面：before 里它几乎不会低于 3 cm，说明支撑脚一直悬着；",
    "- 看视频里脚和地面的缝隙：before 应能看到脚底与地面之间始终有一条亮缝；",
    "- `Female1_A14_stand_to_skip` 是起跳动作，before 的腾空是真实动作，不是缺陷，修后画面会整体下沉一点，属预期。",
  ]
  (out_dir / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
  print(f"\n[done] {len(entries)} video(s) -> {out_dir}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
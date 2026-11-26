"""Render spot-check preview videos for prepared motion clips.

Each clip becomes one MP4: the G1 is replayed in MuJoCo with a floor plane and a tracking
camera, and an overlay shows clip name / time / root height.  Long clips are cut to the
most active window (highest mean root speed) so a few seconds of video show real motion.

Rendering uses MuJoCo's offscreen GLFW context, so ``MUJOCO_GL=glfw`` is set below.

Usage::

    python render_previews.py --out <dir> --clips a.npz b.npz ... [--window 15] [--fps 25]
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("MUJOCO_GL", "glfw")

from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
G1_XML = REPO / "mjlab_research/src/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml"


def build_model(width: int = 768, height: int = 512, xml: Path = G1_XML) -> mujoco.MjModel:
  """Load the G1, add a floor plane (the mjlab asset ships without one) and size the
  offscreen framebuffer so the renderer can produce frames at the requested resolution."""
  spec = mujoco.MjSpec.from_file(str(xml))
  floor = spec.worldbody.add_geom()
  floor.name = "preview_floor"
  floor.type = mujoco.mjtGeom.mjGEOM_PLANE
  floor.size = [0.0, 0.0, 0.02]
  floor.rgba = [0.36, 0.38, 0.42, 1.0]
  spec.visual.global_.offwidth = int(width)
  spec.visual.global_.offheight = int(height)
  return spec.compile()


def active_window(speed: np.ndarray, fps: float, seconds: float) -> tuple[int, int]:
  """Index range of the ``seconds``-long window with the highest mean speed."""
  total = len(speed)
  width = int(round(seconds * fps))
  if width >= total:
    return 0, total
  # Cumulative sum over a stride-1 window; cheap enough for these clip lengths.
  csum = np.concatenate([[0.0], np.cumsum(speed)])
  means = (csum[width:] - csum[:-width]) / width
  start = int(np.argmax(means))
  return start, start + width


def make_camera(model: mujoco.MjModel, distance: float) -> mujoco.MjvCamera:
  """Free camera that is re-aimed at the root every frame (reliable follow cam).

  MuJoCo's ``mjCAMERA_TRACKING`` mode depends on where ``update_scene`` resolves the
  tracked body, which does not follow a free-floating base predictably here; setting
  ``lookat`` ourselves gives the same effect with no surprises.
  """
  cam = mujoco.MjvCamera()
  mujoco.mjv_defaultFreeCamera(model, cam)
  cam.distance = distance
  cam.elevation = -8.0
  cam.azimuth = 140.0
  return cam


def font(size: int) -> ImageFont.ImageFont:
  for candidate in ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"):
    try:
      return ImageFont.truetype(candidate, size)
    except OSError:
      continue
  return ImageFont.load_default()


def overlay(frame: np.ndarray, title: str, subtitle: str, fnt) -> np.ndarray:
  img = Image.fromarray(frame)
  draw = ImageDraw.Draw(img)
  draw.rectangle([0, 0, img.width, 44], fill=(12, 14, 18))
  draw.text((10, 8), title, font=fnt, fill=(240, 242, 245))
  draw.text((10, 26), subtitle, font=fnt, fill=(150, 200, 255))
  return np.asarray(img)


def render_clip(
  clip: Path,
  out_dir: Path,
  model: mujoco.MjModel,
  renderer,
  *,
  fps: float,
  window_seconds: float,
  camera_distance: float,
  tag: str,
) -> dict:
  with np.load(clip) as data:
    joint_pos = data["joint_pos"].astype(np.float64)
    root_pos = data["root_pos_w"].astype(np.float64)
    root_quat = data["root_quat_w"].astype(np.float64)
    root_vel = data["root_lin_vel_w"].astype(np.float64)
    src_fps = float(np.asarray(data["fps"]).reshape(-1)[0]) if "fps" in data.files else 50.0

  speed = np.linalg.norm(root_vel[:, :2], axis=1)
  start, end = active_window(speed, src_fps, window_seconds)
  stride = max(1, int(round(src_fps / fps)))

  data_mj = mujoco.MjData(model)
  cam = make_camera(model, camera_distance)
  fnt = font(15)

  frames = []
  for t in range(start, end, stride):
    data_mj.qpos[:3] = root_pos[t]
    data_mj.qpos[3:7] = root_quat[t]
    data_mj.qpos[7:] = joint_pos[t]
    mujoco.mj_forward(model, data_mj)
    cam.lookat[:] = [root_pos[t, 0], root_pos[t, 1], 0.78]
    renderer.update_scene(data_mj, camera=cam)
    frame = renderer.render()
    frames.append(
      overlay(
        frame,
        f"{tag} :: {clip.stem}",
        f"t={t / src_fps:6.2f}s   root_z={root_pos[t, 2]:.2f}m   |v|={speed[t]:.2f}m/s",
        fnt,
      )
    )

  out_dir.mkdir(parents=True, exist_ok=True)
  name = f"{tag}__{clip.stem}.mp4"
  out_path = out_dir / name
  imageio.mimsave(
    out_path, frames, fps=fps, codec="libx264", quality=7, macro_block_size=1
  )

  # Poster frame (middle of the rendered window) for quick browsing without a player.
  poster = out_dir / f"{name.replace('.mp4', '.png')}"
  Image.fromarray(frames[len(frames) // 2]).save(poster)

  return {
    "clip": clip.name,
    "video": name,
    "poster": poster.name,
    "seconds_rendered": round((end - start) / src_fps, 2),
    "frames": len(frames),
    "window": [round(start / src_fps, 2), round(end / src_fps, 2)],
    "size_mb": round(out_path.stat().st_size / 1e6, 2),
  }


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--out", required=True)
  ap.add_argument("--clips", nargs="+", required=True)
  ap.add_argument("--fps", type=float, default=25.0)
  ap.add_argument("--window", type=float, default=15.0)
  ap.add_argument("--width", type=int, default=768)
  ap.add_argument("--height", type=int, default=512)
  ap.add_argument("--camera-distance", type=float, default=2.6)
  args = ap.parse_args()

  model = build_model(args.width, args.height)
  renderer = mujoco.Renderer(model, height=args.height, width=args.width)
  out_dir = Path(args.out).expanduser()

  results = []
  for clip_arg in args.clips:
    clip = Path(clip_arg).expanduser().resolve()
    if not clip.exists():
      print(f"[skip] missing {clip}")
      continue
    tag = clip.parent.parent.name or "clip"
    if tag == "npz":
      tag = "clip"
    info = render_clip(
      clip,
      out_dir,
      model,
      renderer,
      fps=args.fps,
      window_seconds=args.window,
      camera_distance=args.camera_distance,
      tag=tag,
    )
    results.append(info)
    print(
      f"[ok] {info['video']:<52} {info['frames']:>4} frames  "
      f"{info['seconds_rendered']:>6.1f}s  {info['size_mb']:>5.2f} MB",
      flush=True,
    )

  total_mb = sum(r["size_mb"] for r in results)
  print(f"\n[done] {len(results)} video(s) -> {out_dir}  ({total_mb:.1f} MB)")
  return 0


if __name__ == "__main__":
  sys.exit(main())

"""Compare grounding variants for the retargeted clips.

Variants measured on the same clip:
  baseline   current dataset (per-clip min alignment in the *source* FK model)
  og         GMR ``offset_to_ground=True`` (per-frame human ground offset)
  rolling    post-process: subtract the local (rolling-window) minimum sole height,
             measured with **mjlab's** G1 model, keeping short-term vertical motion

Metrics: sole-height percentiles, share of frames in contact (sole <= 2 cm), flight share,
root-height median/std (bob).

Usage::

    python compare_grounding_variants.py --clips <baseline npz> ... --og-dir <dir> --out <md>
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "glfw")

from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
G1_XML = HERE.parent.parent / "mjlab_research/src/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml"
CONTACT_Z = 0.02  # m
FLIGHT_Z = 0.02


class Probe:
  def __init__(self, xml: Path = G1_XML):
    self.model = mujoco.MjModel.from_xml_path(str(xml))
    self.data = mujoco.MjData(self.model)
    names = [
      mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g)
      for g in range(self.model.ngeom)
    ]
    self.geoms = [g for g, n in enumerate(names) if n and "foot" in n and "_collision" in n]

  def soles(self, qpos: np.ndarray) -> np.ndarray:
    """Lowest sole height per frame (single probe over both feet)."""
    out = np.empty(len(qpos))
    for i in range(len(qpos)):
      self.data.qpos[:] = qpos[i]
      mujoco.mj_kinematics(self.model, self.data)
      out[i] = min(
        self.data.geom_xpos[g][2] - self.model.geom_size[g][0] for g in self.geoms
      )
    return out


def load_clip(path: Path) -> dict:
  with np.load(path) as d:
    return {
      "joint_pos": np.asarray(d["joint_pos"], dtype=np.float64),
      "root_pos_w": np.asarray(d["root_pos_w"], dtype=np.float64),
      "root_quat_w": np.asarray(d["root_quat_w"], dtype=np.float64),
      "root_lin_vel_w": np.asarray(d["root_lin_vel_w"], dtype=np.float64),
      "fps": float(np.asarray(d["fps"]).reshape(-1)[0]) if "fps" in d.files else 50.0,
    }


def qpos_of(clip: dict) -> np.ndarray:
  return np.concatenate([clip["root_pos_w"], clip["root_quat_w"], clip["joint_pos"]], axis=1)


def rolling_ground(clip: dict, probe: Probe, window_s: float = 0.30) -> dict:
  """Lower the clip per frame by the local minimum sole height (window-smoothed)."""
  fps = clip["fps"]
  sole = probe.soles(qpos_of(clip))
  half = max(1, int(round(window_s * fps)))
  # Rolling minimum with a stride-1 window (clip lengths here are small enough).
  idx = np.arange(len(sole))
  rolled = np.array(
    [sole[max(0, i - half) : min(len(sole), i + half + 1)].min() for i in idx]
  )
  out = dict(clip)
  root = clip["root_pos_w"].copy()
  root[:, 2] -= rolled
  out["root_pos_w"] = root
  lin = clip["root_lin_vel_w"].copy()
  if len(root) > 1:
    lin[:, 2] = np.gradient(root[:, 2], 1.0 / fps)
  out["root_lin_vel_w"] = lin
  return out


def metrics(clip: dict, probe: Probe) -> dict:
  sole = probe.soles(qpos_of(clip))
  rz = clip["root_pos_w"][:, 2]
  return {
    "sole_p10": float(np.percentile(sole, 10)),
    "sole_p50": float(np.percentile(sole, 50)),
    "sole_p90": float(np.percentile(sole, 90)),
    "contact_pct": float(100.0 * (sole <= CONTACT_Z).mean()),
    "root_z_p50": float(np.median(rz)),
    "root_z_std": float(rz.std()),
  }


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--clips", nargs="+", required=True, help="baseline npz paths")
  ap.add_argument("--og-dir", default=None, help="dir holding offset_to_ground variants")
  ap.add_argument(
    "--extra",
    nargs="*",
    default=[],
    help="additional variants as label=dir (clips matched by filename)",
  )
  ap.add_argument("--out", default=None)
  args = ap.parse_args()

  probe = Probe()
  lines = [
    "# 接地方案对比（同一批 clip，指标均由 mjlab 的 G1 模型测得）",
    "",
    "| clip | 方案 | 脚底 p10 | p50 | p90 | 触地帧(≤2cm) | 根高中位 | 根高 std |",
    "|---|---|---|---|---|---|---|---|",
  ]
  for clip_path in args.clips:
    base = load_clip(Path(clip_path))
    variants = [("baseline", base), ("rolling(0.3s)", rolling_ground(base, probe))]
    if args.og_dir:
      og = Path(args.og_dir) / Path(clip_path).name
      if og.exists():
        variants.insert(1, ("GMR offset_to_ground", load_clip(og)))
    for spec_text in args.extra:
      label, _, directory = spec_text.partition("=")
      candidate = Path(directory) / Path(clip_path).name
      if candidate.exists():
        variants.insert(1, (label, load_clip(candidate)))
    for name, clip in variants:
      m = metrics(clip, probe)
      lines.append(
        f"| {Path(clip_path).stem} | {name} | {m['sole_p10']*100:+.1f} cm | "
        f"{m['sole_p50']*100:+.1f} cm | {m['sole_p90']*100:+.1f} cm | "
        f"{m['contact_pct']:.1f}% | {m['root_z_p50']:.3f} m | {m['root_z_std']*100:.1f} cm |"
      )

  text = "\n".join(lines) + "\n"
  if args.out:
    Path(args.out).write_text(text, encoding="utf-8")
    print(f"[ok] report -> {args.out}")
  print(text)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
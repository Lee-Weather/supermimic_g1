"""Re-ground clips against **mjlab's** G1 model.

Grounding must be measured with the same robot geometry that training uses.  The two
sources were aligned against other models (UMR's ``g1_29dof_umr.xml`` for ``g1_ase_45``,
GMR's ``g1_mocap_29dof.xml`` for ``lafan1_g1``), so their soles sit a few centimetres off
in mjlab's MJCF -- LAFAN1 clips hover 3-9 cm and never touch the ground.

This tool measures, per clip, the height of the lowest foot capsule sole and shifts the
clip's root ``z`` by a single constant so that the lowest frame touches the ground exactly
(no penetration, contact at the lowest frame).  Joints and root twist are untouched.

Usage::

    python reground_to_mjlab_model.py --root .. --report          # report only
    python reground_to_mjlab_model.py --root .. --apply           # rewrite clips
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

# Fields that mjlab's MotionLoader reads; everything else in a clip is carried over verbatim.
KEEP = (
  "joint_pos",
  "joint_vel",
  "root_pos_w",
  "root_quat_w",
  "root_lin_vel_w",
  "root_ang_vel_w",
  "joint_names",
  "fps",
)


class SoleProbe:
  """Lowest foot-sole height, using the foot capsules on the ankle-roll links."""

  def __init__(self, xml: Path = G1_XML):
    self.model = mujoco.MjModel.from_xml_path(str(xml))
    self.data = mujoco.MjData(self.model)
    names = [
      mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g)
      for g in range(self.model.ngeom)
    ]
    self.geoms = [g for g, n in enumerate(names) if n and "foot" in n and "_collision" in n]
    if not self.geoms:
      raise RuntimeError("foot geoms not found in " + str(xml))

  def lowest_sole(self, qpos: np.ndarray) -> float:
    """Lowest sole height over the frames (horizontal capsules: centre minus radius)."""
    lowest = np.inf
    for i in range(len(qpos)):
      self.data.qpos[:] = qpos[i]
      mujoco.mj_kinematics(self.model, self.data)
      for g in self.geoms:
        z = self.data.geom_xpos[g][2] - self.model.geom_size[g][0]
        if z < lowest:
          lowest = z
    return float(lowest)


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--root", default=str(HERE.parent))
  ap.add_argument("--apply", action="store_true", help="rewrite clips (default: report only)")
  ap.add_argument("--stride", type=int, default=1, help="stride for the min-height estimate")
  args = ap.parse_args()

  probe = SoleProbe()
  root = Path(args.root)
  clips = sorted(root.glob("*/npz/*.npz"))
  if not clips:
    print(f"[fail] no clips under {root}")
    return 2

  shifted = 0
  per_dataset: dict[str, list[float]] = {}
  for clip in clips:
    with np.load(clip) as data:
      payload = {k: data[k] for k in data.files if k in KEEP}
    dataset = clip.parent.parent.name
    qpos = np.concatenate(
      [payload["root_pos_w"], payload["root_quat_w"], payload["joint_pos"]], axis=1
    ).astype(np.float64)
    sole = probe.lowest_sole(qpos[:: args.stride])
    per_dataset.setdefault(dataset, []).append(sole)

    if args.apply and abs(sole) > 5e-4:
      root_pos = np.asarray(payload["root_pos_w"], dtype=np.float64).copy()
      root_pos[:, 2] -= sole
      payload["root_pos_w"] = root_pos.astype(np.float32)
      np.savez_compressed(clip, **payload)
      shifted += 1
      print(f"[fix] {dataset}/{clip.stem:<34} sole {sole:+.4f} -> 0.0000 m")
    else:
      print(f"[ok ] {dataset}/{clip.stem:<34} sole {sole:+.4f} m")

  print()
  for name, soles in per_dataset.items():
    a = np.abs(np.array(soles))
    print(
      f"{name:<12} clips={len(a):>3}  |sole| median {np.median(a)*100:5.2f} cm  "
      f"max {a.max()*100:5.2f} cm  already-grounded {(a <= 5e-4).sum():>3}"
    )
  if args.apply:
    print(f"\n[applied] {shifted} clip(s) shifted; rerun pack_motions + build_manifest")
  else:
    print("\n[report only] pass --apply to rewrite the clips")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
"""Convert UMR G1 ``qpos`` output into mjlab ``MotionLoader`` NPZ.

UMR (``retarget_smpl_to_humanoid_surface_vector.py``) writes ``qpos (T, 36)`` =
``[root_pos(3), root_quat wxyz(4), 29 joints]`` at the source frame rate, with
``robot_joint_names`` and ``fps``.  This conversion:

  1. verifies the joint order equals mjlab's G1 order;
  2. resamples to 50 Hz (linear for positions/joints, slerp for the root quaternion);
  3. measures the lowest foot-sole height with **mjlab's** G1 model and applies rolling
     grounding (+-0.3 s local minimum), so the stance foot lands on the ground while the
     short-term pelvic bob survives;
  4. derives ``root_lin_vel_w`` / ``root_ang_vel_w`` and writes the six mjlab fields.

Usage::

    python umr_qpos_to_mjlab_npz.py --input <umr out.npz> --output <out dir|file>
"""

from __future__ import annotations

import argparse
import importlib.util
import os

os.environ.setdefault("MUJOCO_GL", "glfw")

from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
G1_XML = HERE.parent.parent / "mjlab_research/src/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml"
TARGET_FPS = 50.0


def load(name: str, filename: str):
  spec = importlib.util.spec_from_file_location(name, HERE / filename)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


class SoleProbe:
  """Lowest foot-sole height per frame on mjlab's G1."""

  def __init__(self, xml: Path = G1_XML):
    self.model = mujoco.MjModel.from_xml_path(str(xml))
    self.data = mujoco.MjData(self.model)
    names = [
      mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g)
      for g in range(self.model.ngeom)
    ]
    self.geoms = [g for g, n in enumerate(names) if n and "foot" in n and "_collision" in n]
    if not self.geoms:
      raise RuntimeError("foot geoms not found")

  def soles(self, qpos: np.ndarray) -> np.ndarray:
    out = np.empty(len(qpos))
    for i, q in enumerate(qpos):
      self.data.qpos[:] = q
      mujoco.mj_kinematics(self.model, self.data)
      out[i] = min(
        self.data.geom_xpos[g][2] - self.model.geom_size[g][0] for g in self.geoms
      )
    return out


def convert(umr_npz: Path, probe: SoleProbe, cv, rp, window_s: float = 0.30) -> dict:
  with np.load(umr_npz, allow_pickle=True) as d:
    qpos = np.asarray(d["qpos"], dtype=np.float64)
    fps_in = float(np.asarray(d["fps"]).reshape(-1)[0])
    names = [str(n) for n in np.asarray(d["robot_joint_names"]).reshape(-1)]
  if names != cv.EXPECTED_JOINTS:
    raise ValueError("UMR joint order differs from mjlab order")
  if qpos.shape[1] != 7 + len(cv.EXPECTED_JOINTS):
    raise ValueError(f"unexpected qpos width {qpos.shape[1]}")

  qpos50 = rp.resample_qpos(qpos, fps_in, TARGET_FPS)

  joint = qpos50[:, 7:]
  root = qpos50[:, :3].copy()
  quat = qpos50[:, 3:7]

  # Rolling grounding on mjlab geometry.
  sole = probe.soles(qpos50)
  half = max(1, int(round(window_s * TARGET_FPS)))
  rolled = np.array(
    [
      sole[max(0, i - half) : min(len(sole), i + half + 1)].min()
      for i in range(len(sole))
    ]
  )
  root[:, 2] -= rolled

  dt = 1.0 / TARGET_FPS
  out = {
    "joint_pos": joint.astype(np.float32),
    "joint_vel": np.gradient(joint, dt, axis=0).astype(np.float32),
    "root_pos_w": root.astype(np.float32),
    "root_quat_w": quat.astype(np.float32),
    "root_lin_vel_w": np.gradient(root, dt, axis=0).astype(np.float32),
    "root_ang_vel_w": rp.angular_velocity(quat, dt).astype(np.float32),
    "joint_names": np.array(cv.EXPECTED_JOINTS),
    "fps": np.array([TARGET_FPS], dtype=np.float32),
  }
  return out, sole, rolled


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--input", required=True, help="UMR npz file or dir of them")
  ap.add_argument("--output", required=True, help="output dir (one npz per input)")
  args = ap.parse_args()

  cv = load("ase_to_mjlab_npz", "ase_to_mjlab_npz.py")
  rp = load("retarget_lafan1_to_g1", "retarget_lafan1_to_g1.py")
  probe = SoleProbe()

  src = Path(args.input)
  files = sorted(src.glob("*.npz")) if src.is_dir() else [src]
  out_dir = Path(args.output)
  out_dir.mkdir(parents=True, exist_ok=True)

  for f in files:
    out, sole_before, rolled = convert(f, probe, cv, rp)
    problems = cv.validate(out, TARGET_FPS, f.stem)
    dst = out_dir / f"{f.stem}.npz"
    if problems:
      print(f"[FAIL] {f.name}: {'; '.join(problems)}")
      continue
    np.savez_compressed(dst, **out)
    sole_after = probe.soles(
      np.concatenate([out["root_pos_w"], out["root_quat_w"], out["joint_pos"]], axis=1)
    )
    print(
      f"[ok] {f.stem:<40} {out['joint_pos'].shape[0]:>5} f  "
      f"{out['joint_pos'].shape[0]/TARGET_FPS:>6.1f}s  "
      f"sole p50 {np.percentile(sole_before,50)*100:+5.1f} -> {np.percentile(sole_after,50)*100:+4.1f} cm  "
      f"contact {100*(sole_after<=0.02).mean():3.0f}%"
    )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
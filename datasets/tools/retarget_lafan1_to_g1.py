"""Retarget LAFAN1 BVH clips to Unitree G1 with GMR, then export mjlab MotionLoader NPZ.

Pipeline per clip:
  1. GMR (general_motion_retargeting, bvh -> unitree_g1 IK config) produces MuJoCo qpos
     at LAFAN1's 30 fps: ``[root_pos(3), root_quat wxyz(4), 29 joints]``.
  2. Resample to 50 Hz (linear for positions/joints, slerp for the root quaternion).
  3. Forward-kinematics on the G1 model to get the pelvis (floating base) world pose,
     then take finite differences for root twist and joint velocities.
  4. Ground-align once per clip (lowest foot geom over the clip sits at z=0) and zero the
     first-frame XY, matching what the ASE converter emits.
  5. Write the six mjlab fields + ``joint_names`` (wxyz quaternions, mjlab joint order).

Only mujoco / numpy / scipy plus GMR's IK core are needed; GMR's package ``__init__``
imports torch, which is unusable on this box, so the package is registered as a stub and
the IK module is imported directly.

Usage::

    python retarget_lafan1_to_g1.py --input <bvh dir> --out <npz dir> [--limit 1]
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
import types
from pathlib import Path

import numpy as np

GMR_ROOT = Path(r"E:\Retargeting\GMR")
GMR_G1_XML = GMR_ROOT / "assets" / "unitree_g1" / "g1_mocap_29dof.xml"
HERE = Path(__file__).resolve().parent
TARGET_FPS = 50.0


def load_validator():
  """Load the validate() helper from the sibling converter (no package install needed)."""
  spec = importlib.util.spec_from_file_location(
    "ase_to_mjlab_npz", HERE / "ase_to_mjlab_npz.py"
  )
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def load_gmr():
  """Import GMR's IK core without executing its torch-hungry package __init__."""
  if "general_motion_retargeting" not in sys.modules:
    pkg = types.ModuleType("general_motion_retargeting")
    pkg.__path__ = [str(GMR_ROOT / "general_motion_retargeting")]
    sys.modules["general_motion_retargeting"] = pkg
  if str(GMR_ROOT) not in sys.path:
    sys.path.insert(0, str(GMR_ROOT))
  from general_motion_retargeting.motion_retarget import GeneralMotionRetargeting
  from general_motion_retargeting.utils.lafan1 import load_bvh_file

  def loader(path: str):
    return load_bvh_file(path, format="lafan1")

  return GeneralMotionRetargeting, loader


def quat_slerp(q0: np.ndarray, q1: np.ndarray, blend: np.ndarray) -> np.ndarray:
  """wxyz slerp for rows of q0/q1 with per-row blend in [0, 1]."""
  q0 = q0 / np.linalg.norm(q0, axis=1, keepdims=True)
  q1 = q1 / np.linalg.norm(q1, axis=1, keepdims=True)
  b = np.asarray(blend, dtype=np.float64).reshape(-1, 1)
  dot = np.sum(q0 * q1, axis=1, keepdims=True)
  flip = dot < 0.0
  q1 = np.where(flip, -q1, q1)
  dot = np.abs(dot)
  theta = np.arccos(np.clip(dot, -1.0, 1.0))
  sin_theta = np.sin(theta)
  small = sin_theta < 1e-6
  denom = np.where(small, 1.0, sin_theta)
  w0 = np.where(small, 1.0 - b, np.sin((1.0 - b) * theta) / denom)
  w1 = np.where(small, b, np.sin(b * theta) / denom)
  out = w0 * q0 + w1 * q1
  return out / np.linalg.norm(out, axis=1, keepdims=True)


def resample_qpos(qpos: np.ndarray, fps_in: float, fps_out: float) -> np.ndarray:
  """Resample a (T, 7 + n_joints) qpos array; quaternion rows are slerped."""
  times_in = np.arange(len(qpos)) / fps_in
  times_out = np.arange(0.0, times_in[-1], 1.0 / fps_out)
  if len(times_out) < 2:
    raise ValueError("clip too short to resample")
  idx1 = np.clip(np.searchsorted(times_in, times_out, side="right"), 1, len(times_in) - 1)
  idx0 = idx1 - 1
  span = times_in[idx1] - times_in[idx0]
  blend = np.clip((times_out - times_in[idx0]) / np.where(span == 0, 1.0, span), 0.0, 1.0)

  out = np.empty((len(times_out), qpos.shape[1]), dtype=np.float64)
  out[:, :3] = (1.0 - blend)[:, None] * qpos[idx0, :3] + blend[:, None] * qpos[idx1, :3]
  out[:, 3:7] = quat_slerp(qpos[idx0, 3:7], qpos[idx1, 3:7], blend)
  out[:, 7:] = (1.0 - blend)[:, None] * qpos[idx0, 7:] + blend[:, None] * qpos[idx1, 7:]
  return out


class G1Kinematics:
  """Forward kinematics + foot-height probe for a 29-DoF floating-base G1 model."""

  def __init__(self, xml_path: Path):
    import mujoco

    self.mujoco = mujoco
    self.model = mujoco.MjModel.from_xml_path(str(xml_path))
    self.data = mujoco.MjData(self.model)
    if self.model.nq != 7 + 29:
      raise ValueError(f"expected nq=36, got {self.model.nq}")

    self.joint_names = [
      mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
      for i in range(self.model.njnt)
      if self.model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE
    ]
    self.body_names = [
      mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
      for i in range(1, self.model.nbody)
    ]
    ankle_bodies = {
      mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
      for i in range(1, self.model.nbody)
      if (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i) or "").endswith(
        "ankle_roll_link"
      )
    }
    self.foot_geoms = [
      g
      for g in range(self.model.ngeom)
      if self.model.geom_contype[g] != 0
      and mujoco.mj_id2name(
        self.model, mujoco.mjtObj.mjOBJ_BODY, int(self.model.geom_bodyid[g])
      )
      in ankle_bodies
    ]
    if not self.foot_geoms:
      raise ValueError("no foot collision geoms found in " + str(xml_path))

  def run(self, qpos: np.ndarray):
    frames = len(qpos)
    body_pos = np.empty((frames, self.model.nbody - 1, 3))
    body_quat = np.empty((frames, self.model.nbody - 1, 4))
    foot_z = np.empty(frames)
    for i in range(frames):
      self.data.qpos[:] = qpos[i]
      self.mujoco.mj_kinematics(self.model, self.data)
      body_pos[i] = self.data.xpos[1:]
      body_quat[i] = self.data.xquat[1:]
      foot_z[i] = min(
        self.data.geom_xpos[g][2]
        - (
          self.model.geom_size[g][1] * abs(self.data.geom_xmat[g][8])
          + self.model.geom_size[g][0]
        )
        for g in self.foot_geoms
      )
    return body_pos, body_quat, foot_z


def angular_velocity(quats: np.ndarray, dt: float) -> np.ndarray:
  """World-frame angular velocity from a wxyz quaternion sequence (central diff)."""
  q = quats.astype(np.float64)
  if len(q) == 1:
    return np.zeros((1, 3))
  if len(q) == 2:
    rel = quat_mul(q[1:2], quat_conj(q[0:1]))
    return quat_to_axis_angle(rel) / dt
  rel = quat_mul(q[2:], quat_conj(q[:-2]))
  mid = quat_to_axis_angle(rel) / (2.0 * dt)
  return np.concatenate([mid[:1], mid, mid[-1:]], axis=0)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
  aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
  bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
  return np.stack(
    [
      aw * bw - ax * bx - ay * by - az * bz,
      aw * bx + ax * bw + ay * bz - az * by,
      aw * by - ax * bz + ay * bw + az * bx,
      aw * bz + ax * by - ay * bx + az * bw,
    ],
    axis=-1,
  )


def quat_conj(q: np.ndarray) -> np.ndarray:
  return np.concatenate([q[..., :1], -q[..., 1:]], axis=-1)


def quat_to_axis_angle(q: np.ndarray) -> np.ndarray:
  q = q / np.linalg.norm(q, axis=-1, keepdims=True)
  q = np.where(q[..., :1] < 0.0, -q, q)
  w = np.clip(q[..., 0], -1.0, 1.0)
  angle = 2.0 * np.arccos(w)
  sin_half = np.sqrt(np.maximum(0.0, 1.0 - w * w))
  safe = sin_half > 1e-8
  scale = np.where(safe, angle / np.where(safe, sin_half, 1.0), 2.0)
  return q[..., 1:] * scale[..., None]


def retarget_clip(
  bvh_path: Path, kin: G1Kinematics, gmr_cls, loader, offset_to_ground: bool = False
) -> dict:
  frames, human_height = loader(str(bvh_path))
  retarget = gmr_cls(
    src_human="bvh_lafan1", tgt_robot="unitree_g1", actual_human_height=human_height
  )
  qpos = np.stack(
    [retarget.retarget(frame, offset_to_ground=offset_to_ground) for frame in frames]
  )
  if qpos.shape[1] != 7 + 29:
    raise ValueError(f"unexpected qpos width {qpos.shape[1]}")

  qpos50 = resample_qpos(qpos, 30.0, TARGET_FPS)
  body_pos, body_quat, foot_z = kin.run(qpos50)

  shift = float(foot_z.min())
  body_pos[:, :, 2] -= shift
  qpos50 = qpos50.copy()
  qpos50[:, 2] -= shift

  offset = body_pos[0, 0, :2].copy()
  body_pos[:, :, :2] -= offset

  root = kin.body_names.index("pelvis")
  joint_pos = qpos50[:, 7:].astype(np.float64)
  dt = 1.0 / TARGET_FPS

  out = {
    "joint_pos": joint_pos.astype(np.float32),
    "joint_vel": np.gradient(joint_pos, dt, axis=0).astype(np.float32),
    "root_pos_w": body_pos[:, root, :].astype(np.float32),
    "root_quat_w": body_quat[:, root, :].astype(np.float32),
    "root_lin_vel_w": np.gradient(body_pos[:, root, :], dt, axis=0).astype(np.float32),
    "root_ang_vel_w": angular_velocity(body_quat[:, root, :], dt).astype(np.float32),
    "joint_names": np.array(kin.joint_names),
    "fps": np.array([TARGET_FPS], dtype=np.float32),
  }
  return out, human_height, len(frames)


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--input", required=True, help="dir (or glob) of LAFAN1 .bvh files")
  ap.add_argument("--out", required=True, help="output dir for mjlab npz clips")
  ap.add_argument("--limit", type=int, default=0, help="stop after N clips (0 = all)")
  ap.add_argument("--skip-existing", action="store_true")
  ap.add_argument(
    "--offset-to-ground",
    action="store_true",
    help="pass GMR's per-frame ground offset (pins the lowest human ankle 10cm up, "+
    "which puts the robot soles on the ground but removes pelvic bob)",
  )
  ap.add_argument(
    "--shard",
    default="0/1",
    help="process only clips with index %% n == i, as in 'i/n' (for parallel runs)",
  )
  args = ap.parse_args()

  try:
    shard_i, shard_n = (int(x) for x in str(args.shard).split("/"))
  except ValueError:
    print("[fail] --shard must look like 'i/n'", file=sys.stderr)
    return 2
  if not 0 <= shard_i < shard_n:
    print("[fail] --shard index out of range", file=sys.stderr)
    return 2

  validate_mod = load_validator()
  gmr_cls, loader = load_gmr()
  kin = G1Kinematics(GMR_G1_XML)
  if kin.joint_names != validate_mod.EXPECTED_JOINTS:
    print("[fail] GMR G1 joint order differs from mjlab order", file=sys.stderr)
    return 2

  in_path = Path(args.input).expanduser()
  bvhs = sorted(in_path.glob("*.bvh")) if in_path.is_dir() else sorted(Path().glob(str(in_path)))
  bvhs = [b for i, b in enumerate(bvhs) if i % shard_n == shard_i]
  if not bvhs:
    print(f"[fail] no .bvh under {args.input}", file=sys.stderr)
    return 2

  out_dir = Path(args.out).expanduser()
  out_dir.mkdir(parents=True, exist_ok=True)

  done = failed = 0
  for bvh in bvhs:
    dst = out_dir / (bvh.stem + ".npz")
    if args.skip_existing and dst.exists():
      print(f"[skip] {bvh.name}")
      continue
    t0 = time.perf_counter()
    try:
      out, height, src_frames = retarget_clip(bvh, kin, gmr_cls, loader, args.offset_to_ground)
      problems = validate_mod.validate(out, TARGET_FPS, bvh.stem)
      if problems:
        raise ValueError("; ".join(problems))
      np.savez_compressed(dst, **out)
      elapsed = time.perf_counter() - t0
      done += 1
      print(
        f"[ok]   {bvh.name:<34} {src_frames:>5} src -> {out['joint_pos'].shape[0]:>5} f  "
        f"{out['joint_pos'].shape[0]/TARGET_FPS:>6.1f}s  {elapsed:>6.1f}s  h={height:.3f}",
        flush=True,
      )
    except Exception as exc:  # noqa: BLE001 - keep going on bad clips
      failed += 1
      print(f"[FAIL] {bvh.name}: {exc}", flush=True)
    if args.limit and done + failed >= args.limit:
      break

  print(f"\n[done] {done} clips -> {out_dir}  ({failed} failed)")
  return 0 if failed == 0 else 1


if __name__ == "__main__":
  sys.exit(main())

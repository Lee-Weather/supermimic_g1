"""Convert ASE / Isaac-Gym nine-field G1 motion clips into mjlab ``MotionLoader`` NPZ.

Input (ASE ``G1MotionLib`` format, one file per clip)::

    fps, joint_pos(T,29), joint_vel(T,29),
    body_pos_w(T,B,3), body_quat_w(T,B,4), body_lin_vel_w(T,B,3), body_ang_vel_w(T,B,3),
    joint_names(29), body_names(B)

Output (mjlab format, see mjlab/tasks/tracking/mdp/commands.py::MotionLoader)::

    joint_pos(T,29), joint_vel(T,29),
    root_pos_w(T,3), root_quat_w(T,4) (wxyz), root_lin_vel_w(T,3), root_ang_vel_w(T,3),
    joint_names(29)

The root is the floating-base link (``pelvis``): its per-body arrays collapse into the
``root_*`` arrays.  mjlab recomputes every other body pose with forward kinematics from
``joint_pos``, so the remaining bodies are dropped.  Quaternions are already wxyz on both
sides (MuJoCo ``xquat`` -> mjlab ``root_quat_w``), so no convention change is applied.

Usage::

    python ase_to_mjlab_npz.py --input <dir with *.npz> --output <out dir>
    python ase_to_mjlab_npz.py --input <out dir> --output <out dir> --verify
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

# mjlab's G1 joint order: identical to the MJCF document order in
# mjlab_research/src/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml and to
# mjlab_research/src/mjlab/scripts/csv_to_npz.py::JOINT_NAMES.
EXPECTED_JOINTS = [
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
]

SOURCE_FIELDS = (
  "joint_pos",
  "joint_vel",
  "body_pos_w",
  "body_quat_w",
  "body_lin_vel_w",
  "body_ang_vel_w",
)

OUT_FIELDS = (
  "joint_pos",
  "joint_vel",
  "root_pos_w",
  "root_quat_w",
  "root_lin_vel_w",
  "root_ang_vel_w",
)

# Physical sanity bounds.  Deliberately loose: they are meant to catch unit / axis
# mistakes, not to police motion style.
MAX_JOINT_ABS = 3.2  # rad
MAX_JOINT_STEP = 1.5  # rad per frame (50 Hz)
MAX_LIN_VEL = 12.0  # m/s
MAX_ANG_VEL = 80.0  # rad/s
ROOT_Z_RANGE = (0.05, 1.8)  # m
MIN_CLIP_FRAMES = 20
QUAT_NORM_TOL = 2e-3


def sha256_file(path: Path) -> str:
  h = hashlib.sha256()
  with open(path, "rb") as fh:
    for chunk in iter(lambda: fh.read(1 << 20), b""):
      h.update(chunk)
  return h.hexdigest()


def reorder(joint_names: np.ndarray, field: np.ndarray) -> tuple[np.ndarray, list[str]]:
  """Return ``field`` with its joint axis permuted into EXPECTED_JOINTS order.

  Axis -1 is assumed to be the joint axis.  Returns the reordered array plus a log
  message describing the permutation (empty when the order already matches).
  """
  names = [str(n) for n in joint_names.reshape(-1)]
  if names == EXPECTED_JOINTS:
    return field, []
  missing = [n for n in EXPECTED_JOINTS if n not in names]
  extra = [n for n in names if n not in EXPECTED_JOINTS]
  if missing or extra:
    raise ValueError(f"joint name mismatch; missing={missing} extra={extra}")
  idx = [names.index(n) for n in EXPECTED_JOINTS]
  return np.take(field, idx, axis=-1), [f"permuted {len(idx)} joint columns"]


def validate(out: dict[str, np.ndarray], fps: float, label: str) -> list[str]:
  """Structural + physical checks.  Returns a list of failures (empty == ok)."""
  problems: list[str] = []
  joint_pos = out["joint_pos"]
  frames = joint_pos.shape[0]

  if frames < MIN_CLIP_FRAMES:
    problems.append(f"only {frames} frames")
  for field in OUT_FIELDS:
    if field not in out:
      problems.append(f"missing field {field}")
  if problems:
    return problems

  for field in OUT_FIELDS:
    if not np.isfinite(out[field]).all():
      problems.append(f"non-finite values in {field}")
  if out["joint_pos"].shape != (frames, len(EXPECTED_JOINTS)):
    problems.append(f"joint_pos shape {out['joint_pos'].shape}")
  if out["joint_vel"].shape != (frames, len(EXPECTED_JOINTS)):
    problems.append(f"joint_vel shape {out['joint_vel'].shape}")
  for field in ("root_pos_w", "root_lin_vel_w", "root_ang_vel_w"):
    if out[field].shape != (frames, 3):
      problems.append(f"{field} shape {out[field].shape}")
  if out["root_quat_w"].shape != (frames, 4):
    problems.append(f"root_quat_w shape {out['root_quat_w'].shape}")

  norm = np.linalg.norm(out["root_quat_w"], axis=-1)
  if np.abs(norm - 1.0).max() > QUAT_NORM_TOL:
    problems.append(f"quat norm off by {np.abs(norm - 1.0).max():.2e}")

  if np.abs(joint_pos).max() > MAX_JOINT_ABS:
    problems.append(f"|joint_pos| max {np.abs(joint_pos).max():.2f} rad")
  if frames > 1:
    step = np.abs(np.diff(joint_pos, axis=0)).max()
    if step > MAX_JOINT_STEP:
      problems.append(f"joint jump {step:.2f} rad in one frame")
  lin = np.linalg.norm(out["root_lin_vel_w"], axis=-1).max()
  ang = np.linalg.norm(out["root_ang_vel_w"], axis=-1).max()
  if lin > MAX_LIN_VEL:
    problems.append(f"root speed {lin:.1f} m/s")
  if ang > MAX_ANG_VEL:
    problems.append(f"root ang vel {ang:.0f} rad/s")
  z = out["root_pos_w"][:, 2]
  if z.min() < ROOT_Z_RANGE[0] or z.max() > ROOT_Z_RANGE[1]:
    problems.append(f"root z [{z.min():.2f}, {z.max():.2f}] m")
  if abs(fps - 50.0) > 1e-3:
    problems.append(f"fps {fps} (mjlab control runs at 50 Hz; resample first)")
  return problems


def convert_clip(src: Path, dst: Path) -> dict:
  with np.load(src, allow_pickle=True) as data:
    missing = [f for f in (*SOURCE_FIELDS, "joint_names", "body_names", "fps") if f not in data.files]
    if missing:
      raise ValueError(f"{src.name}: missing {missing}")
    fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    body_names = [str(n) for n in data["body_names"].reshape(-1)]
    if "pelvis" not in body_names:
      raise ValueError(f"{src.name}: no 'pelvis' among body_names")
    root = body_names.index("pelvis")
    if root != 0:
      print(f"  [warn] {src.name}: pelvis is body #{root}, not body 0")

    joint_pos, log_pos = reorder(data["joint_names"], data["joint_pos"])
    joint_vel, log_vel = reorder(data["joint_names"], data["joint_vel"])

    out = {
      "joint_pos": np.ascontiguousarray(joint_pos, dtype=np.float32),
      "joint_vel": np.ascontiguousarray(joint_vel, dtype=np.float32),
      "root_pos_w": np.ascontiguousarray(data["body_pos_w"][:, root, :], dtype=np.float32),
      "root_quat_w": np.ascontiguousarray(data["body_quat_w"][:, root, :], dtype=np.float32),
      "root_lin_vel_w": np.ascontiguousarray(
        data["body_lin_vel_w"][:, root, :], dtype=np.float32
      ),
      "root_ang_vel_w": np.ascontiguousarray(
        data["body_ang_vel_w"][:, root, :], dtype=np.float32
      ),
      "joint_names": np.array(EXPECTED_JOINTS),
      "fps": np.array([fps], dtype=np.float32),
    }

  problems = validate(out, fps, src.stem)
  if problems:
    raise ValueError(f"{src.name}: " + "; ".join(problems))

  dst.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(dst, **out)
  frames = int(out["joint_pos"].shape[0])
  return {
    "file": dst.name,
    "source": src.name,
    "frames": frames,
    "seconds": round(frames / fps, 2),
    "fps": fps,
    "root_body_index": root,
    "notes": log_pos + log_vel,
    "sha256": sha256_file(dst),
  }


def verify_dir(root: Path) -> list[str]:
  """Re-check an output directory (single clips or packed dir)."""
  problems: list[str] = []
  clips = sorted(root.glob("*.npz"))
  packed = (root / "joint_names.json").exists()
  if packed:
    names = json.loads((root / "joint_names.json").read_text())
    if names != EXPECTED_JOINTS:
      problems.append("packed joint_names.json != mjlab order")
    ends = np.load(root / "motion_ends.npy")
    for field in OUT_FIELDS:
      arr = np.load(root / f"{field}.npy", mmap_mode="r")
      if arr.shape[0] != int(ends[-1]):
        problems.append(f"{field}.npy rows {arr.shape[0]} != motion_ends[-1] {int(ends[-1])}")
    print(f"[verify] packed dir ok: {len(ends)} clips, {int(ends[-1])} frames")
  if clips:
    print(f"[verify] {len(clips)} clip files")
    for clip in clips:
      with np.load(clip, allow_pickle=True) as data:
        out = {f: data[f] for f in OUT_FIELDS if f in data.files}
        fps = float(np.asarray(data["fps"]).reshape(-1)[0]) if "fps" in data.files else 50.0
        names = [str(n) for n in data["joint_names"].reshape(-1)]
      if names != EXPECTED_JOINTS:
        problems.append(f"{clip.name}: joint order mismatch")
      problems += [f"{clip.name}: {p}" for p in validate(out, fps, clip.stem)]
  if not clips and not packed:
    problems.append(f"{root}: no clips and no packed dir")
  return problems


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--input", required=True, help="source dir with *.npz clips")
  ap.add_argument("--output", required=True, help="destination dir")
  ap.add_argument("--manifest", default=None, help="manifest JSON path (default <output>/manifest.json)")
  ap.add_argument("--verify", action="store_true", help="only verify an existing output dir")
  args = ap.parse_args()

  out_dir = Path(args.output).expanduser()
  if args.verify:
    problems = verify_dir(out_dir)
    for p in problems:
      print(f"[FAIL] {p}")
    print("[verify] OK" if not problems else f"[verify] {len(problems)} problem(s)")
    return 0 if not problems else 1

  in_dir = Path(args.input).expanduser()
  clips = sorted(in_dir.glob("*.npz"))
  if not clips:
    print(f"[fail] no *.npz under {in_dir}")
    return 2

  entries, failed = [], []
  for src in clips:
    try:
      entry = convert_clip(src, out_dir / src.name)
      entries.append(entry)
      print(
        f"[ok]   {entry['source']:<46} {entry['frames']:>5} frames  "
        f"{entry['seconds']:>6.1f}s"
      )
    except Exception as exc:  # noqa: BLE001 - report and continue
      failed.append({"source": src.name, "error": str(exc)})
      print(f"[FAIL] {src.name}: {exc}")

  total_frames = sum(e["frames"] for e in entries)
  total_seconds = sum(e["seconds"] for e in entries)
  manifest = {
    "schema": "mjlab MotionLoader NPZ (G1 29-DoF, 50 Hz, embedded wxyz quaternions)",
    "source_dir": str(in_dir),
    "output_dir": str(out_dir),
    "joint_names": EXPECTED_JOINTS,
    "clip_count": len(entries),
    "failed_count": len(failed),
    "total_frames": total_frames,
    "total_seconds": round(total_seconds, 2),
    "clips": entries,
    "failed": failed,
  }
  manifest_path = Path(args.manifest) if args.manifest else out_dir / "manifest.json"
  manifest_path.parent.mkdir(parents=True, exist_ok=True)
  manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

  print(
    f"\n[done] {len(entries)} clips -> {out_dir}  "
    f"({total_frames} frames, {total_seconds/60:.2f} min)"
  )
  print(f"[done] manifest: {manifest_path}")
  if failed:
    print(f"[warn] {len(failed)} clip(s) failed; see manifest")
    return 1
  return 0


if __name__ == "__main__":
  sys.exit(main())
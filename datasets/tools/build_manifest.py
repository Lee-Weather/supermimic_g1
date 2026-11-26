"""Scan the prepared SuperMimic datasets and emit a single manifest + summary table.

Every clip is re-validated with ``ase_to_mjlab_npz.validate`` so the manifest doubles as a
readiness report: if this script is green, ``MotionLoader`` will accept every clip.

Usage::

    python build_manifest.py --root <datasets dir>
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def load_validator():
  spec = importlib.util.spec_from_file_location(
    "ase_to_mjlab_npz", HERE / "ase_to_mjlab_npz.py"
  )
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def sha256_file(path: Path) -> str:
  h = hashlib.sha256()
  with open(path, "rb") as fh:
    for chunk in iter(lambda: fh.read(1 << 20), b""):
      h.update(chunk)
  return h.hexdigest()


def scan_clip_dir(clip_dir: Path, validator) -> dict:
  clips = sorted(clip_dir.glob("*.npz"))
  entries, problems, frames_total, seconds_total = [], [], 0, 0.0
  for clip in clips:
    with np.load(clip, allow_pickle=True) as data:
      out = {f: data[f] for f in validator.OUT_FIELDS if f in data.files}
      fps = float(np.asarray(data["fps"]).reshape(-1)[0]) if "fps" in data.files else 50.0
      names = [str(n) for n in data["joint_names"].reshape(-1)]
    issues = validator.validate(out, fps, clip.stem)
    if names != validator.EXPECTED_JOINTS:
      issues.append("joint order != mjlab order")
    frames = int(out["joint_pos"].shape[0])
    frames_total += frames
    seconds_total += frames / fps
    for issue in issues:
      problems.append(f"{clip.name}: {issue}")
    entries.append(
      {
        "file": clip.name,
        "frames": frames,
        "seconds": round(frames / fps, 2),
        "bytes": clip.stat().st_size,
        "sha256": sha256_file(clip),
        "ok": not issues,
      }
    )
  return {
    "clip_dir": str(clip_dir),
    "clip_count": len(clips),
    "total_frames": frames_total,
    "total_seconds": round(seconds_total, 2),
    "clips": entries,
    "problems": problems,
  }


def scan_packed(packed_dir: Path, validator) -> dict:
  if not (packed_dir / "joint_names.json").exists():
    return {}
  names = json.loads((packed_dir / "joint_names.json").read_text())
  ends = np.load(packed_dir / "motion_ends.npy")
  motion_names = []
  if (packed_dir / "motion_names.json").exists():
    motion_names = json.loads((packed_dir / "motion_names.json").read_text())
  shapes = {}
  problems = []
  for field in validator.OUT_FIELDS:
    arr = np.load(packed_dir / f"{field}.npy", mmap_mode="r")
    shapes[field] = list(arr.shape)
    if arr.shape[0] != int(ends[-1]):
      problems.append(f"{field}.npy rows {arr.shape[0]} != motion_ends[-1] {int(ends[-1])}")
  if names != validator.EXPECTED_JOINTS:
    problems.append("joint_names.json != mjlab order")
  return {
    "packed_dir": str(packed_dir),
    "clip_count": len(ends),
    "total_frames": int(ends[-1]),
    "total_seconds": round(int(ends[-1]) / 50.0, 2),
    "fields": shapes,
    "motion_names": motion_names,
    "bytes": sum(p.stat().st_size for p in packed_dir.glob("*")),
    "problems": problems,
  }


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--root", default=str(HERE.parent), help="datasets root")
  ap.add_argument("--out", default=None, help="manifest path (default <root>/manifest.json)")
  args = ap.parse_args()

  root = Path(args.root).expanduser().resolve()
  validator = load_validator()

  groups = {}
  # Any dataset directory laid out as <name>/npz (+ optional <name>/packed) is picked up.
  clip_dirs = sorted(p for p in root.glob("*/npz") if p.is_dir())
  for clip_dir in clip_dirs:
    name = clip_dir.parent.name
    if clip_dir.is_dir():
      groups[name] = scan_clip_dir(clip_dir, validator)
    packed = clip_dir.parent / "packed"
    if packed.is_dir():
      groups[name + "_packed"] = scan_packed(packed, validator)

  total_clips = sum(
    g.get("clip_count", 0) for k, g in groups.items() if not k.endswith("_packed")
  )
  total_seconds = sum(
    g.get("total_seconds", 0.0) for k, g in groups.items() if not k.endswith("_packed")
  )
  all_problems = [p for g in groups.values() for p in g.get("problems", [])]

  manifest = {
    "schema": "mjlab MotionLoader NPZ (Unitree G1 29-DoF, 50 Hz, wxyz quaternions)",
    "datasets_root": str(root),
    "summary": {
      "clip_count": total_clips,
      "total_seconds": round(total_seconds, 2),
      "total_minutes": round(total_seconds / 60.0, 2),
      "problems": len(all_problems),
    },
    "groups": groups,
    "problems": all_problems,
  }
  out_path = Path(args.out) if args.out else root / "manifest.json"
  out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

  print(f"{'group':<20}{'clips':>7}{'frames':>9}{'minutes':>9}")
  for key, g in groups.items():
    print(
      f"{key:<20}{g.get('clip_count', 0):>7}{g.get('total_frames', 0):>9}"
      f"{g.get('total_seconds', 0)/60:>9.2f}"
    )
  print(f"\n[manifest] {out_path}")
  if all_problems:
    print(f"[FAIL] {len(all_problems)} problem(s):")
    for p in all_problems[:20]:
      print("   ", p)
    return 1
  print("[ok] every clip passes structural + physical validation")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
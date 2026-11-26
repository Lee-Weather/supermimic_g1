"""Finalize the walk-only dataset: merge pilot + P0 clips, assemble the walk set, pack.

Steps (idempotent):
  1. copy the HDM05 pilot clips (``hdm05_walk/npz``) into ``walk_gapfill/npz`` unless a
     clip with the same stem is already there;
  2. assemble ``walk_only/npz`` = walk-class clips from ``g1_ase_45`` + all ``walk_gapfill``
     clips, so training can point at a single directory;
  3. validate every clip, pack both directories, rebuild ``datasets/manifest.json``.

Usage::

    python finalize_walk_gapfill.py
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASETS = HERE.parent
REPO = DATASETS.parent


def load_audit():
  spec = importlib.util.spec_from_file_location("audit", HERE / "audit_walk_coverage.py")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def verify(path: Path) -> tuple[bool, str]:
  proc = subprocess.run(
    [sys.executable, str(HERE / "ase_to_mjlab_npz.py"),
     "--input", str(path), "--output", str(path), "--verify"],
    capture_output=True, text=True,
  )
  lines = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()]
  return proc.returncode == 0, " | ".join(lines[-2:]) if lines else proc.stderr[-200:]


def pack(directory: Path) -> str:
  proc = subprocess.run(
    [sys.executable, str(REPO / "mjlab_research/src/mjlab/scripts/pack_motions.py"),
     "--input", str(directory / "*.npz"), "--output", str(directory.parent / "packed")],
    capture_output=True, text=True,
  )
  lines = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()]
  return " | ".join(lines[-2:])


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--gapfill", default=str(DATASETS / "walk_gapfill/npz"))
  ap.add_argument("--walk-only", default=str(DATASETS / "walk_only/npz"))
  ap.add_argument("--skip-pack", action="store_true")
  args = ap.parse_args()

  audit = load_audit()
  gapfill = Path(args.gapfill)
  walk_only = Path(args.walk_only)
  gapfill.mkdir(parents=True, exist_ok=True)
  walk_only.mkdir(parents=True, exist_ok=True)

  pilots = sorted((DATASETS / "hdm05_walk/npz").glob("*.npz"))
  copied = 0
  for clip in pilots:
    dst = gapfill / clip.name
    if not dst.exists():
      shutil.copy2(clip, dst)
      copied += 1
  print(f"[merge] {copied} pilot clip(s) copied -> {gapfill} ({len(pilots)} available)")

  # walk_only = walk-class clips from g1_ase_45 + everything in walk_gapfill (all walking).
  added = skipped = 0
  for clip in sorted((DATASETS / "g1_ase_45/npz").glob("*.npz")):
    if audit.classify(clip.stem) not in audit.WALK_LABELS:
      skipped += 1
      continue
    dst = walk_only / clip.name
    if not dst.exists():
      shutil.copy2(clip, dst)
      added += 1
  for clip in sorted(gapfill.glob("*.npz")):
    dst = walk_only / clip.name
    if not dst.exists():
      shutil.copy2(clip, dst)
      added += 1
  print(f"[walk_only] +{added} clip(s) ({skipped} non-walk clips skipped) -> {walk_only}")

  ok_all = True
  for directory in (gapfill, walk_only):
    ok, detail = verify(directory)
    ok_all &= ok
    n = len(list(directory.glob("*.npz")))
    print(f"[verify] {directory.name}: {n} clips — {detail}")
    if not ok:
      continue
    if not args.skip_pack:
      print(f"[pack] {directory.name}: {pack(directory)}")

  if not ok_all:
    print("[FAIL] validation failed; manifest not rebuilt")
    return 1

  manifest = subprocess.run(
    [sys.executable, str(HERE / "build_manifest.py"), "--root", str(DATASETS)],
    capture_output=True, text=True,
  )
  print(manifest.stdout.strip())
  return 0 if manifest.returncode == 0 else 1


if __name__ == "__main__":
  raise SystemExit(main())
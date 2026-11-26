"""Batch: HDM05 (SMPL-X) -> UMR retarget -> mjlab NPZ.

HDM05 ships as 215 ``*_stageii.npz`` sequences in SMPL-X parameter format (``poses``,
``trans``, ``betas``), which UMR consumes directly.  This script

  1. scans the archive and picks walking candidates by root speed
     (median 0.3-1.6 m/s, p95 <= 2.2 m/s) and minimum duration;
  2. extracts each sequence and runs UMR's surface-vector retargeter with the G1 config
     (``D:\\conda\\envs\\umr\\python.exe``, torch 2.4.1+cu121, RTX A2000);
  3. converts UMR ``qpos`` to mjlab NPZ via ``umr_qpos_to_mjlab_npz.py``.

Usage::

    python umr_batch_hdm05.py --auto 6 --min-seconds 8 --out ../lafan...    # pilot
"""

from __future__ import annotations

import argparse
import io
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATASETS = HERE.parent
UMR_ROOT = Path(r"E:\UMR")
UMR_PY = Path(r"D:\conda\envs\umr\python.exe")
UMR_CONFIG = UMR_ROOT / "czy/g1/umr_g1_config.json"
UMR_SCRIPT = UMR_ROOT / "scripts/retarget_smpl_to_humanoid_surface_vector.py"
ARCHIVE = Path(r"E:\F1\F1_ase_h\czy\npz\HDM05.tar.bz2")


def scan(tar: tarfile.TarFile, min_seconds: float, subjects: set[str] | None):
  """Walking candidates: (name, seconds, v50, v95, subject).

  A clip only counts as walking when the pelvis also stays in a walking band: HDM05
  sequences like ``04-01`` walk and then kneel/sit (pelvis drops to 0.05 m), which passes
  a pure speed filter but is not usable for a walk-only set.
  """
  out = []
  for member in tar.getmembers():
    if not member.name.endswith(".npz"):
      continue
    subject = member.name.split("/")[1]
    if subjects and subject not in subjects:
      continue
    data = np.load(io.BytesIO(tar.extractfile(member).read()), allow_pickle=True)
    if "poses" not in data.files:
      continue
    trans = np.asarray(data["trans"], dtype=float)
    fps = float(np.asarray(data["mocap_frame_rate"]).ravel()[0])
    if len(trans) < 2:
      continue
    speed = np.linalg.norm(np.diff(trans[:, :2], axis=0), axis=1) * fps
    seconds = len(trans) / fps
    v50, v95 = float(np.median(speed)), float(np.percentile(speed, 95))
    z = trans[:, 2]
    z_ok = z.min() > 0.6 and z.std() < 0.10 and 0.8 <= float(np.median(z)) <= 1.15
    if 0.3 <= v50 <= 1.6 and v95 <= 2.2 and seconds >= min_seconds and z_ok:
      out.append((member.name, seconds, v50, v95, subject))
  return out


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--out", required=True, help="mjlab npz output dir")
  ap.add_argument("--auto", type=int, default=6, help="how many clips to process (0 = all)")
  ap.add_argument("--min-seconds", type=float, default=8.0)
  ap.add_argument("--subjects", default=None, help="comma list, e.g. mm,bk")
  ap.add_argument("--work", default=str(DATASETS / "source/hdm05_qpos"))
  args = ap.parse_args()

  work = Path(args.work)
  work.mkdir(parents=True, exist_ok=True)
  out_dir = Path(args.out)
  out_dir.mkdir(parents=True, exist_ok=True)

  tar = tarfile.open(ARCHIVE)
  subjects = set(args.subjects.split(",")) if args.subjects else None
  candidates = scan(tar, args.min_seconds, subjects)
  print(f"[scan] {len(candidates)} walking candidates (>= {args.min_seconds}s)")
  if not candidates:
    return 2

  # Longest first, then round-robin across subjects for a diverse pilot.
  candidates.sort(key=lambda r: -r[1])
  if args.auto:
    picked, seen = [], {}
    for cand in candidates:
      seen[cand[4]] = seen.get(cand[4], 0) + 1
      if seen[cand[4]] <= max(1, args.auto // max(1, len({c[4] for c in candidates}))):
        picked.append(cand)
      if len(picked) >= args.auto:
        break
    picked = picked or candidates[: args.auto]
  else:
    picked = candidates

  print(f"[plan] {len(picked)} clips, {sum(c[1] for c in picked)/60:.1f} min of source motion")
  ok = failed = 0
  for i, (name, seconds, v50, v95, subject) in enumerate(picked, 1):
    stem = Path(name).stem
    src = work / f"{stem}.npz"
    if not src.exists():
      src.write_bytes(tar.extractfile(name).read())
    qpos_out = work / f"{stem}_g1.npz"
    if not qpos_out.exists():
      t0 = time.perf_counter()
      proc = subprocess.run(
        [
          str(UMR_PY), str(UMR_SCRIPT),
          "--config", str(UMR_CONFIG),
          "--data", str(src),
          "--seq-key", stem,
          "--out", str(qpos_out),
        ],
        cwd=str(UMR_ROOT),
        capture_output=True,
        text=True,
      )
      took = time.perf_counter() - t0
      if proc.returncode != 0:
        failed += 1
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
        print(f"[{i:02d}/{len(picked)}] FAIL {stem}: {' | '.join(tail)}", flush=True)
        continue
    else:
      took = 0.0
    conv = subprocess.run(
      [
        sys.executable, str(HERE / "umr_qpos_to_mjlab_npz.py"),
        "--input", str(qpos_out), "--output", str(out_dir),
      ],
      capture_output=True,
      text=True,
    )
    if conv.returncode != 0:
      failed += 1
      print(f"[{i:02d}/{len(picked)}] FAIL convert {stem}: {conv.stderr[-300:]}", flush=True)
      continue
    ok += 1
    print(
      f"[{i:02d}/{len(picked)}] {stem:<34} {subject} {seconds:6.1f}s "
      f"v50 {v50:.2f}  umr {took:5.0f}s  {conv.stdout.strip().splitlines()[-1]}",
      flush=True,
    )

  print(f"\n[done] {ok} ok / {failed} failed -> {out_dir}")
  return 0 if failed == 0 else 1


if __name__ == "__main__":
  raise SystemExit(main())
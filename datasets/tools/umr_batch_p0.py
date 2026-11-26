"""P0 batch: fill the straight-walking gap from CMU + HDM05 via UMR.

Both archives ship SMPL-X ``*_stageii.npz`` sequences, which UMR consumes directly.

Pipeline (single command, resumable):

  1. stream each archive once, filter walking candidates by root speed **and** pelvis-height
     stability (a walk-then-sit sequence otherwise slips through), select a balanced set;
  2. extract just those sequences to ``source/<archive>_sel``;
  3. run UMR (``D:\\conda\\envs\\umr``) on N worker processes, then convert each qpos to
     mjlab NPZ with rolling grounding;
  4. write ``batch_report.json`` with per-clip timings and grounding metrics.

Usage::

    python umr_batch_p0.py --out ../walk_gapfill/npz --workers 2 --cmu 10 --hdm05 20
"""

from __future__ import annotations

import argparse
import collections
import io
import json
import multiprocessing as mp
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
ARCHIVES = {
  "CMU": Path(r"E:\F1\F1_ase_h\czy\npz\CMU.tar.bz2"),
  "HDM05": Path(r"E:\F1\F1_ase_h\czy\npz\HDM05.tar.bz2"),
}
# Walking bands used to keep the speed coverage even (m/s, median root speed).
BANDS = [(0.30, 0.55), (0.55, 0.75), (0.75, 0.95), (0.95, 1.60)]

_WORKER_CTX: dict = {}


def scan_archive(archive: str) -> list[dict]:
  """Walk candidates with speed + pelvis-height filters."""
  tar = tarfile.open(ARCHIVES[archive])
  rows = []
  for member in tar.getmembers():
    if not member.name.endswith(".npz"):
      continue
    subject = member.name.split("/")[1]
    data = np.load(io.BytesIO(tar.extractfile(member).read()), allow_pickle=True)
    if "poses" not in data.files:
      continue
    trans = np.asarray(data["trans"], dtype=float)
    fps = float(np.asarray(data["mocap_frame_rate"]).ravel()[0])
    if len(trans) < 2:
      continue
    speed = np.linalg.norm(np.diff(trans[:, :2], axis=0), axis=1) * fps
    z = trans[:, 2]
    v50, v95 = float(np.median(speed)), float(np.percentile(speed, 95))
    seconds = len(trans) / fps
    if not (0.3 <= v50 <= 1.6 and v95 <= 2.2 and seconds >= 8.0):
      continue
    if not (z.min() > 0.6 and z.std() < 0.10 and 0.8 <= float(np.median(z)) <= 1.15):
      continue
    rows.append(
      {
        "archive": archive,
        "member": member.name,
        "stem": Path(member.name).stem,
        "subject": subject,
        "seconds": seconds,
        "v50": v50,
        "v95": v95,
      }
    )
  tar.close()
  return rows


def select_longest(rows: list[dict], total: int, per_subject: int) -> list[dict]:
  picked, count = [], collections.Counter()
  for row in sorted(rows, key=lambda r: -r["seconds"]):
    if count[row["subject"]] >= per_subject:
      continue
    picked.append(row)
    count[row["subject"]] += 1
    if len(picked) >= total:
      break
  return picked


def select_by_band(rows: list[dict], per_band: int, per_subject: int) -> list[dict]:
  picked = []
  for lo, hi in BANDS:
    band = [r for r in rows if lo <= r["v50"] < hi]
    count = collections.Counter()
    taken = 0
    for row in sorted(band, key=lambda r: -r["seconds"]):
      if count[row["subject"]] >= per_subject:
        continue
      picked.append(row)
      count[row["subject"]] += 1
      taken += 1
      if taken >= per_band:
        break
  return picked


def extract(archive: str, wanted: dict[str, dict], dest: Path) -> None:
  """Stream the archive once and write out only the selected sequences."""
  dest.mkdir(parents=True, exist_ok=True)
  tar = tarfile.open(ARCHIVES[archive])
  written = 0
  for member in tar.getmembers():
    row = wanted.get(member.name)
    if row is None:
      continue
    target = dest / f"{row['stem']}.npz"
    if not target.exists():
      target.write_bytes(tar.extractfile(member).read())
      written += 1
  tar.close()
  print(f"[extract] {archive}: {written} new, {len(wanted)} selected", flush=True)


def process_one(task: dict) -> dict:
  """Worker: UMR retarget then convert to mjlab NPZ."""
  src = Path(task["src"])
  qpos = Path(task["qpos"])
  log = task["log"]
  started = time.perf_counter()
  if not qpos.exists():
    proc = subprocess.run(
      [
        str(UMR_PY), str(UMR_SCRIPT),
        "--config", str(UMR_CONFIG),
        "--data", str(src),
        "--seq-key", task["stem"],
        "--out", str(qpos),
      ],
      cwd=str(UMR_ROOT),
      capture_output=True,
      text=True,
    )
    if proc.returncode != 0:
      tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-2:]
      return {**task, "status": "umr-fail", "detail": " | ".join(tail),
              "umr_sec": round(time.perf_counter() - started, 1)}
  umr_sec = round(time.perf_counter() - started, 1)

  conv = subprocess.run(
    [sys.executable, str(HERE / "umr_qpos_to_mjlab_npz.py"),
     "--input", str(qpos), "--output", str(task["out_dir"])],
    capture_output=True,
    text=True,
  )
  status = "ok" if conv.returncode == 0 else "convert-fail"
  detail = conv.stdout.strip().splitlines()[-1] if conv.returncode == 0 else conv.stdout.strip()[-200:]
  with open(log, "a", encoding="utf-8") as fh:
    fh.write(f"[{status}] {task['stem']:<36} {task['subject']:<4} {task['seconds']:6.1f}s "
             f"umr {umr_sec:6.0f}s  {detail}\n")
  return {**task, "status": status, "detail": detail, "umr_sec": umr_sec}


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--out", required=True, help="mjlab npz output dir")
  ap.add_argument("--workers", type=int, default=2)
  ap.add_argument("--cmu", type=int, default=10, help="how many CMU clips (longest, balanced)")
  ap.add_argument("--hdm05", type=int, default=20, help="how many HDM05 clips (speed bands)")
  ap.add_argument("--per-subject", type=int, default=2)
  ap.add_argument("--source-dir", default=None)
  ap.add_argument("--report", default=None)
  args = ap.parse_args()

  out_dir = Path(args.out)
  out_dir.mkdir(parents=True, exist_ok=True)
  source_root = Path(args.source_dir) if args.source_dir else DATASETS / "source/p0_src"
  qpos_root = DATASETS / "source/p0_qpos"
  qpos_root.mkdir(parents=True, exist_ok=True)
  log = out_dir.parent / "p0_progress.log"
  log.write_text("", encoding="utf-8")

  selections: list[dict] = []
  if args.cmu:
    print("[scan] CMU ...", flush=True)
    cmu_rows = [r for r in scan_archive("CMU") if not (out_dir / f"{r['stem']}.npz").exists()]
    print(f"[scan] CMU candidates: {len(cmu_rows)}", flush=True)
    cmu_pick = select_longest(cmu_rows, args.cmu, args.per_subject)
    extract("CMU", {r["member"]: r for r in cmu_pick}, source_root / "CMU")
    selections += cmu_pick
  if args.hdm05:
    print("[scan] HDM05 ...", flush=True)
    hdm_rows = [r for r in scan_archive("HDM05") if not (out_dir / f"{r['stem']}.npz").exists()]
    print(f"[scan] HDM05 candidates: {len(hdm_rows)}", flush=True)
    per_band = max(1, args.hdm05 // len(BANDS))
    hdm_pick = select_by_band(hdm_rows, per_band, args.per_subject)
    extract("HDM05", {r["member"]: r for r in hdm_pick}, source_root / "HDM05")
    selections += hdm_pick

  print(f"[plan] {len(selections)} clips, "
        f"{sum(s['seconds'] for s in selections)/60:.1f} min of source motion", flush=True)

  tasks = []
  for row in selections:
    src = source_root / row["archive"] / f"{row['stem']}.npz"
    if not src.exists():
      print(f"[warn] missing source for {row['stem']}", flush=True)
      continue
    tasks.append(
      {
        **row,
        "src": str(src),
        "qpos": str(qpos_root / f"{row['stem']}_g1.npz"),
        "out_dir": str(out_dir),
        # Passed explicitly: on Windows the spawn start method re-imports this module in
        # each worker, so module-level state set in main would not be visible.
        "log": str(log),
      }
    )

  if not tasks:
    print("[done] nothing to do")
    return 0


  started = time.perf_counter()
  with mp.Pool(args.workers) as pool:
    results = pool.map(process_one, tasks)
  elapsed = time.perf_counter() - started

  ok = sum(1 for r in results if r["status"] == "ok")
  report = {
    "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "workers": args.workers,
    "elapsed_sec": round(elapsed, 1),
    "clips": len(results),
    "ok": ok,
    "failed": len(results) - ok,
    "source_minutes": round(sum(r["seconds"] for r in results) / 60, 2),
    "umr_seconds": round(sum(r.get("umr_sec", 0) for r in results), 1),
    "results": results,
  }
  report_path = Path(args.report) if args.report else out_dir.parent / "batch_report.json"
  report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
  print(f"[done] {ok}/{len(results)} ok in {elapsed/60:.1f} min -> {out_dir}", flush=True)
  print(f"[report] {report_path}", flush=True)
  return 0 if ok == len(results) else 1


if __name__ == "__main__":
  raise SystemExit(main())

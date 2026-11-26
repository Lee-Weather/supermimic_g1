"""Scan a SMPL-X motion archive (CMU / HDM05 ``*_stageii.npz``) for walking candidates.

Reports, per archive: sequence count, total duration, walking candidates
(median root speed 0.3-1.6 m/s, p95 <= 2.2 m/s), per-subject coverage, and the longest
continuous walks.  Nothing is extracted or written; the archive is streamed.

Usage::

    python scan_smplx_archive.py --archive CMU --out ../cmu_scan.md
"""

from __future__ import annotations

import argparse
import collections
import io
import tarfile
from pathlib import Path

import numpy as np

ARCHIVES = {
  "CMU": Path(r"E:\F1\F1_ase_h\czy\npz\CMU.tar.bz2"),
  "HDM05": Path(r"E:\F1\F1_ase_h\czy\npz\HDM05.tar.bz2"),
}


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--archive", choices=sorted(ARCHIVES), required=True)
  ap.add_argument("--out", default=None)
  args = ap.parse_args()

  tar = tarfile.open(ARCHIVES[args.archive])
  rows = []
  for member in tar.getmembers():
    if not member.name.endswith(".npz"):
      continue
    data = np.load(io.BytesIO(tar.extractfile(member).read()), allow_pickle=True)
    if "poses" not in data.files:
      continue
    trans = np.asarray(data["trans"], dtype=float)
    fps = float(np.asarray(data["mocap_frame_rate"]).ravel()[0])
    if len(trans) < 2:
      continue
    speed = np.linalg.norm(np.diff(trans[:, :2], axis=0), axis=1) * fps
    rows.append(
      (
        member.name,
        member.name.split("/")[1],
        len(trans) / fps,
        float(np.median(speed)),
        float(np.percentile(speed, 95)),
      )
    )

  total_min = sum(r[2] for r in rows) / 60
  walk = [r for r in rows if 0.3 <= r[3] <= 1.6 and r[4] <= 2.2 and r[2] >= 3.0]
  run = [r for r in rows if r[3] > 2.2]
  per_subject = collections.defaultdict(lambda: [0, 0.0, 0])
  for name, subj, sec, v50, v95 in rows:
    per_subject[subj][0] += 1
    per_subject[subj][1] += sec
    if (name, subj, sec, v50, v95) in walk:
      per_subject[subj][2] += 1

  lines = [
    f"# {args.archive} 库扫描（SMPL-X ``stageii.npz``）",
    "",
    f"- 序列数：**{len(rows)}**，总时长 **{total_min:.1f} min**",
    f"- 步行候选（v50 0.3–1.6 m/s、v95 ≤ 2.2 m/s、≥3 s）：**{len(walk)} 条 / {sum(r[2] for r in walk)/60:.1f} min**",
    f"- 跑动（v50 > 2.2 m/s）：{len(run)} 条 / {sum(r[2] for r in run)/60:.1f} min",
    "",
    "| 受试者 | 序列 | 时长 | 步行候选 |",
    "|---|---|---|---|",
  ]
  for subj, (n, sec, w) in sorted(per_subject.items(), key=lambda kv: -kv[1][1]):
    lines.append(f"| {subj} | {n} | {sec/60:.1f} min | {w} |")
  lines += ["", "## 最长的 10 条连续步行", "", "| 序列 | 受试者 | 时长 | v50 | v95 |", "|---|---|---|---|---|"]
  for name, subj, sec, v50, v95 in sorted(walk, key=lambda r: -r[2])[:10]:
    lines.append(f"| `{name.split('/')[-1]}` | {subj} | {sec:.1f}s | {v50:.2f} | {v95:.2f} |")

  text = "\n".join(lines) + "\n"
  if args.out:
    Path(args.out).write_text(text, encoding="utf-8")
  print(text)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
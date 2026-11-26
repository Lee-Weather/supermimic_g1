"""Per-clip grounding quality of the walk-only subset (read-only).

Grounding is measured with **mjlab's** G1 model: the lowest foot-capsule sole height per
frame.  For walking, at least one foot should sit on the ground almost all the time, so a
clip is flagged when the median sole height is above ``--p50-cm`` or the share of frames
with a sole within 2 cm (contact) is below ``--min-contact``.

Usage::

    python grounding_report.py --out ../walk_only_grounding.md
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "glfw")

import importlib.util
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
G1_XML = HERE.parent.parent / "mjlab_research/src/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml"


def load_audit():
  spec = importlib.util.spec_from_file_location("audit", HERE / "audit_walk_coverage.py")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


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
    out = np.empty(len(qpos))
    for i in range(len(qpos)):
      self.data.qpos[:] = qpos[i]
      mujoco.mj_kinematics(self.model, self.data)
      out[i] = min(
        self.data.geom_xpos[g][2] - self.model.geom_size[g][0] for g in self.geoms
      )
    return out


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--out", default=str(HERE.parent / "walk_only_grounding.md"))
  ap.add_argument("--stride", type=int, default=2)
  ap.add_argument("--p50-cm", type=float, default=2.0, help="median sole height limit")
  ap.add_argument("--min-contact", type=float, default=10.0, help="minimum contact share (%%)")
  ap.add_argument(
    "--roots",
    default=None,
    help="comma list of clip dirs (default: g1_ase_45/npz,lafan1_g1/npz)",
  )
  ap.add_argument("--all-clips", action="store_true", help="do not filter to walk-only classes")
  args = ap.parse_args()

  audit = load_audit()
  probe = Probe()
  if args.roots:
    roots = [Path(p) for p in args.roots.split(",")]
  else:
    roots = [HERE.parent / "g1_ase_45" / "npz", HERE.parent / "lafan1_g1" / "npz"]
  rows = []
  for root in roots:
    for clip in sorted(root.glob("*.npz")):
      if not args.all_clips and audit.classify(clip.stem) not in audit.WALK_LABELS:
        continue
      with np.load(clip) as d:
        qpos = np.concatenate(
          [d["root_pos_w"], d["root_quat_w"], d["joint_pos"]], axis=1
        ).astype(np.float64)[:: args.stride]
        fps = float(np.asarray(d["fps"]).reshape(-1)[0]) if "fps" in d.files else 50.0
        frames = d["joint_pos"].shape[0]
        root_z = np.asarray(d["root_pos_w"])[:, 2]
      sole = probe.soles(qpos)
      rows.append(
        {
          "dataset": root.parent.name,
          "clip": clip.stem,
          "category": audit.classify(clip.stem),
          "seconds": frames / fps,
          "p10": float(np.percentile(sole, 10)) * 100,
          "p50": float(np.percentile(sole, 50)) * 100,
          "p90": float(np.percentile(sole, 90)) * 100,
          "contact": float(100.0 * (sole <= 0.02).mean()),
          "root_std": float(root_z.std()) * 100,
        }
      )

  flagged = [r for r in rows if r["p50"] > args.p50_cm or r["contact"] < args.min_contact]
  lines = [
    "# walk-only 子集的接地质量（mjlab 模型实测）",
    "",
    f"判定：中位脚底高度 > {args.p50_cm:.1f} cm **或** 触地帧占比 < {args.min_contact:.0f}% 即判为有问题。",
    "",
    f"**{len(flagged)} / {len(rows)} 条有问题**（占比 {100*len(flagged)/max(len(rows),1):.0f}%）",
    "",
  ]
  by_cat: dict[str, list[dict]] = {}
  for r in rows:
    by_cat.setdefault(r["category"], []).append(r)
  lines += ["| 类别 | 总数 | 有问题 | 中位脚底 cm | 触地率 |", "|---|---|---|---|---|"]
  for cat, rs in sorted(by_cat.items(), key=lambda kv: -len(kv[1])):
    bad = [r for r in rs if r in flagged]
    lines.append(
      f"| {cat} | {len(rs)} | {len(bad)} | "
      f"{np.median([r['p50'] for r in bad]) if bad else 0:.1f} | "
      f"{np.median([r['contact'] for r in bad]) if bad else 0:.1f}% |"
    )
  lines += [
    "",
    "## 逐条明细",
    "",
    "| 数据集 | clip | 类别 | 时长 | 脚底 p10 | p50 | p90 | 触地率 | 根高 std | 判定 |",
    "|---|---|---|---|---|---|---|---|---|---|",
  ]
  for r in sorted(rows, key=lambda r: (r["dataset"], r["category"], -r["p50"])):
    verdict = "❌" if r in flagged else "✅"
    lines.append(
      f"| {r['dataset']} | `{r['clip']}` | {r['category']} | {r['seconds']:.1f}s | "
      f"{r['p10']:+.1f} | {r['p50']:+.1f} | {r['p90']:+.1f} | {r['contact']:.1f}% | "
      f"{r['root_std']:.1f} cm | {verdict} |"
    )

  Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
  print(f"[ok] {len(rows)} walk-only clips, {len(flagged)} flagged -> {args.out}")
  for cat, rs in sorted(by_cat.items(), key=lambda kv: -len(kv[1])):
    bad = [r for r in rs if r in flagged]
    print(f"   {cat:<14} {len(bad):>2}/{len(rs):<2} flagged")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
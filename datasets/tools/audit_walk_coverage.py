"""Audit walking coverage of the prepared datasets (read-only).

For every clip it reports duration, root speed stats and a flight-phase fraction measured
by forward kinematics on mjlab's G1: a frame counts as flight when the lowest foot *sole*
(centimetres above ground, capsule centre minus radius) of BOTH feet is above 2 cm.
Walking clips must stay near 0 %; running must show a clear flight phase, which is a cheap
data-driven check that the walk/run labels are trustworthy.

Usage::

    python audit_walk_coverage.py --out ../walk_only_audit.md
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "glfw")

from collections import defaultdict
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
G1_XML = REPO / "mjlab_research/src/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml"
FLIGHT_Z = 0.02  # m, sole height above ground for both feet == airborne

CATEGORIES = (
  # (label, keywords) - first match wins; order matters.  Turning and backward walking
  # must be matched before the generic forward-walk keywords, because those clip names
  # contain the word "walk" too.
  ("stand", ("a1_stand", "a2_sway")),
  ("turn", ("turn_left", "turn_right", "turn_around", "turn_change", "_turn_")),
  ("backward", ("walk_backwards", "walk_back", "backwards_turn", "stand_to_walk_back")),
  ("run_transition", ("stand_to_run", "run_to_stand", "walk_to_run", "run_to_walk")),
  ("start_stop", ("stand_to_walk", "walk_to_stand", "stand_to_skip", "skip_to_stand")),
  ("walk_forward", ("walk1_subject", "walk2_subject", "walk3_subject", "walk4_subject",
                    "qk_walk", "_walk1", "_walk_")),
  ("run", ("run", "sprint")),
  ("jump_fall", ("jump", "fall", "getup", "ground", "skip")),
  ("dance_fight_other", ("dance", "fight", "aim", "push", "multiple")),
)

WALK_LABELS = ("stand", "turn", "backward", "start_stop", "walk_forward")


def classify(name: str) -> str:
  low = name.lower()
  for label, keys in CATEGORIES:
    if any(k in low for k in keys):
      return label
  return "unclassified"


class FeetFK:
  """Lowest foot-capsule sole height per side for a whole clip."""

  def __init__(self, xml: Path = G1_XML):
    self.model = mujoco.MjModel.from_xml_path(str(xml))
    self.data = mujoco.MjData(self.model)
    names = [
      mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g)
      for g in range(self.model.ngeom)
    ]
    self.left = [g for g, n in enumerate(names) if n and n.startswith("left_foot")]
    self.right = [g for g, n in enumerate(names) if n and n.startswith("right_foot")]
    if not self.left or not self.right:
      raise RuntimeError("foot geoms not found")

  def soles(self, qpos_all: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left = np.empty(len(qpos_all))
    right = np.empty(len(qpos_all))
    for i, q in enumerate(qpos_all):
      self.data.qpos[:] = q
      mujoco.mj_kinematics(self.model, self.data)
      left[i] = min(
        self.data.geom_xpos[g][2] - self.model.geom_size[g][0] for g in self.left
      )
      right[i] = min(
        self.data.geom_xpos[g][2] - self.model.geom_size[g][0] for g in self.right
      )
    return left, right


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--out", default=str(HERE.parent / "walk_only_audit.md"))
  ap.add_argument("--stride", type=int, default=2, help="analyse every Nth frame")
  args = ap.parse_args()

  fk = FeetFK()
  roots = [HERE.parent / "g1_ase_45" / "npz", HERE.parent / "lafan1_g1" / "npz"]
  rows = []
  for root in roots:
    for clip in sorted(root.glob("*.npz")):
      with np.load(clip) as d:
        qpos = np.concatenate(
          [d["root_pos_w"], d["root_quat_w"], d["joint_pos"]], axis=1
        ).astype(np.float64)[:: args.stride]
        vel = np.linalg.norm(
          np.asarray(d["root_lin_vel_w"])[:: args.stride][:, :2], axis=1
        )
        fps = float(np.asarray(d["fps"]).reshape(-1)[0]) if "fps" in d.files else 50.0
        frames = d["joint_pos"].shape[0]
      left, right = fk.soles(qpos)
      flight = (left > FLIGHT_Z) & (right > FLIGHT_Z)
      rows.append(
        {
          "dataset": root.parent.name,
          "clip": clip.stem,
          "category": classify(clip.stem),
          "seconds": frames / fps,
          "v_mean": float(vel.mean()),
          "v_p95": float(np.percentile(vel, 95)),
          "v_max": float(vel.max()),
          "flight_pct": float(100.0 * flight.mean()),
          "sole_min": float(min(left.min(), right.min())),
          "sole_p50": float(np.percentile(np.minimum(left, right), 50)),
        }
      )

  by_cat = defaultdict(lambda: {"clips": 0, "seconds": 0.0, "v_p95": [], "flight": [],
                                "sole": []})
  for r in rows:
    g = by_cat[r["category"]]
    g["clips"] += 1
    g["seconds"] += r["seconds"]
    g["v_p95"].append(r["v_p95"])
    g["flight"].append(r["flight_pct"])
    g["sole"].append(r["sole_min"])

  walk_rows = [r for r in rows if r["category"] in WALK_LABELS]
  walk_seconds = sum(r["seconds"] for r in walk_rows)
  total_seconds = sum(r["seconds"] for r in rows)

  lines = [
    "# walk-only 数据集齐备性审计",
    "",
    "本报告只读扫描 `datasets/` 下已转换的 clip。除时长与根速外，每条 clip 都用 **mjlab 的 G1 模型**",
    f"做正运动学，统计**腾空帧比例**（双脚脚底同时高于 {FLIGHT_Z*100:.0f} cm 即判腾空）与**脚底最低高度**。",
    "",
    "## 按动作类别汇总",
    "",
    "| 类别 | clips | 时长 | 根速 p95 区间 (m/s) | 腾空帧占比 | 脚底最低高度 |",
    "|---|---|---|---|---|---|",
  ]
  for label in [c[0] for c in CATEGORIES] + ["unclassified"]:
    if label not in by_cat:
      continue
    g = by_cat[label]
    lines.append(
      f"| {label} | {g['clips']} | {g['seconds']/60:.1f} min | "
      f"{min(g['v_p95']):.2f} – {max(g['v_p95']):.2f} | "
      f"{min(g['flight']):.1f}% – {max(g['flight']):.1f}% | "
      f"{min(g['sole'])*100:.1f} – {max(g['sole'])*100:.1f} cm |"
    )
  lines += [
    "",
    f"**walk-only 候选：{len(walk_rows)} 条 / {walk_seconds/60:.1f} min**"
    f"（全量 {len(rows)} 条 / {total_seconds/60:.1f} min，占 {100*walk_seconds/total_seconds:.0f}%）",
    "",
    "## walk-only 明细",
    "",
    "| 数据集 | clip | 类别 | 时长 | v_mean | v_p95 | v_max | 腾空% | 脚底最低 |",
    "|---|---|---|---|---|---|---|---|---|",
  ]
  for r in sorted(walk_rows, key=lambda r: (WALK_LABELS.index(r["category"]), r["clip"])):
    lines.append(
      f"| {r['dataset']} | `{r['clip']}` | {r['category']} | {r['seconds']:.1f}s | "
      f"{r['v_mean']:.2f} | {r['v_p95']:.2f} | {r['v_max']:.2f} | {r['flight_pct']:.1f} | "
      f"{r['sole_min']*100:+.1f} cm |"
    )

  Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
  print(f"[ok] {len(rows)} clips audited -> {args.out}")
  print(f"[walk-only] {len(walk_rows)} clips / {walk_seconds/60:.1f} min")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
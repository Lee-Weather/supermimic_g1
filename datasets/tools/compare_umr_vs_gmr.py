"""UMR-pipeline vs GMR-pipeline: quantitative comparison of the retargeted clips.

The two datasets come from different retargeters *and* different motion sources
(``g1_ase_45`` = UMR on ACCAD, ``lafan1_g1`` = GMR on LAFAN1), so this compares the
end-to-end pipelines as delivered.  Per clip we measure, with mjlab's G1 model:

  contact   share of frames with the lowest foot sole within 2 cm of the ground
  slide     mean horizontal speed of the *stance* foot while it is in contact (m/s);
            a good retarget keeps the planted foot still, so lower is better
  z_min/std pelvis height extremes (a collapse shows up as z_min << 0.5 m)
  qlim      share of joint samples outside the MJCF joint range
  jerk      p99 |delta q| per frame (rad/frame at 50 Hz) as a smoothness proxy
  vel_p99   p99 joint speed (rad/s)

Usage::

    python compare_umr_vs_gmr.py --out ../umr_vs_gmr.md
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
REPO = HERE.parent.parent
G1_XML = REPO / "mjlab_research/src/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml"
CONTACT_Z = 0.02


def load_audit():
  spec = importlib.util.spec_from_file_location("audit", HERE / "audit_walk_coverage.py")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


class Probe:
  """Per-frame sole heights plus the horizontal position of each foot."""

  def __init__(self, xml: Path = G1_XML):
    self.model = mujoco.MjModel.from_xml_path(str(xml))
    self.data = mujoco.MjData(self.model)
    names = [
      mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g)
      for g in range(self.model.ngeom)
    ]
    self.left = [g for g, n in enumerate(names) if n and n.startswith("left_foot")]
    self.right = [g for g, n in enumerate(names) if n and n.startswith("right_foot")]
    self.l_body = mujoco.mj_name2id(
      self.model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link"
    )
    self.r_body = mujoco.mj_name2id(
      self.model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link"
    )
    idx = [
      i for i in range(self.model.njnt)
      if self.model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE
    ]
    self.lo = np.array([self.model.jnt_range[i][0] for i in idx])
    self.hi = np.array([self.model.jnt_range[i][1] for i in idx])

  def scan(self, qpos: np.ndarray) -> dict:
    sole_l = np.empty(len(qpos))
    sole_r = np.empty(len(qpos))
    xy_l = np.empty((len(qpos), 2))
    xy_r = np.empty((len(qpos), 2))
    for i, q in enumerate(qpos):
      self.data.qpos[:] = q
      mujoco.mj_kinematics(self.model, self.data)
      sole_l[i] = min(
        self.data.geom_xpos[g][2] - self.model.geom_size[g][0] for g in self.left
      )
      sole_r[i] = min(
        self.data.geom_xpos[g][2] - self.model.geom_size[g][0] for g in self.right
      )
      xy_l[i] = self.data.xpos[self.l_body][:2]
      xy_r[i] = self.data.xpos[self.r_body][:2]
    return {"sole_l": sole_l, "sole_r": sole_r, "xy_l": xy_l, "xy_r": xy_r}


def metrics(clip_path: Path, probe: Probe, fps: float = 50.0, stride: int = 2) -> dict:
  with np.load(clip_path) as d:
    joint = np.asarray(d["joint_pos"], dtype=np.float64)
    qpos = np.concatenate(
      [
        np.asarray(d["root_pos_w"], dtype=np.float64),
        np.asarray(d["root_quat_w"], dtype=np.float64),
        joint,
      ],
      axis=1,
    )[::stride]
    root_z = np.asarray(d["root_pos_w"])[:, 2]
    seconds = joint.shape[0] / fps
    jv = np.asarray(d["joint_vel"], dtype=np.float64)

  scan = probe.scan(qpos)
  sole = np.minimum(scan["sole_l"], scan["sole_r"])
  contact = sole <= CONTACT_Z

  # Foot slide: horizontal speed of whichever foot is lower, on contact frames only.
  if contact.any():
    left_lower = scan["sole_l"] <= scan["sole_r"]
    xy = np.where(left_lower[:, None], scan["xy_l"], scan["xy_r"])
    dt = (1.0 / fps) * stride
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1) / dt
    pair_contact = contact[1:] & contact[:-1]
    slide = float(np.median(step[pair_contact])) if pair_contact.any() else float("nan")
  else:
    slide = float("nan")

  dq = (
    np.abs(np.diff(joint, axis=0)).max(axis=1) if len(joint) > 1 else np.array([0.0])
  )
  qlim = float(100.0 * ((joint < probe.lo - 1e-6) | (joint > probe.hi + 1e-6)).mean())

  return {
    "seconds": seconds,
    "contact_pct": float(100.0 * contact.mean()),
    "slide": slide,
    "z_min": float(root_z.min()),
    "z_std": float(root_z.std()),
    "qlim_pct": qlim,
    "jerk_p99": float(np.percentile(dq, 99)),
    "vel_p99": float(np.percentile(np.abs(jv), 99)),
  }


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--out", default=str(HERE.parent / "umr_vs_gmr.md"))
  ap.add_argument("--stride", type=int, default=2)
  args = ap.parse_args()

  audit = load_audit()
  probe = Probe()
  datasets = {
    "UMR (g1_ase_45 / ACCAD)": HERE.parent / "g1_ase_45" / "npz",
    "GMR (lafan1_g1 / LAFAN1)": HERE.parent / "lafan1_g1" / "npz",
  }

  rows: dict[str, list[dict]] = {}
  for label, root in datasets.items():
    entries = []
    for clip in sorted(root.glob("*.npz")):
      cat = audit.classify(clip.stem)
      m = metrics(clip, probe, stride=args.stride)
      m["category"] = cat
      m["walk_only"] = cat in audit.WALK_LABELS
      entries.append(m)
    rows[label] = entries

  def summarize(entries: list[dict]) -> dict:
    def med(key):
      vals = [e[key] for e in entries if np.isfinite(e[key])]
      return float(np.median(vals)) if vals else float("nan")

    return {
      "clips": len(entries),
      "seconds": sum(e["seconds"] for e in entries) / 60.0,
      "contact": med("contact_pct"),
      "slide": med("slide"),
      "z_min": med("z_min"),
      "z_std": med("z_std") * 100,
      "qlim": med("qlim_pct"),
      "jerk": med("jerk_p99"),
      "vel": med("vel_p99"),
      # Falls / dance / ground motions legitimately reach the floor, so a "collapse"
      # only counts among locomotion clips.
      "collapsed": sum(1 for e in entries if e["walk_only"] and e["z_min"] < 0.5),
    }

  lines = [
    "# UMR vs GMR 重定向结果对比",
    "",
    "两批数据的**重定向器与动捕源都不同**（`g1_ase_45` = UMR on ACCAD，`lafan1_g1` = GMR on LAFAN1），",
    "所以这里是「两条端到端管线」的对比，不是同一输入下的 A/B。所有指标都用 mjlab 的 G1 模型现算。",
    "",
    "| 指标 | 含义 | 好的一侧 |",
    "|---|---|---|",
    "| 触地率 | 最低脚底 ≤2 cm 的帧占比 | 走路应高（>60%） |",
    "| 脚滑 | 触地帧中支撑脚的水平速度 | 越低越好 |",
    "| 根高 z_min / std | 骨盆最低点 / 起伏 | z_min 不应远低于 0.5 m |",
    "| 关节越界 | 超出 MJCF 关节范围的采样占比 | 0% |",
    "| jerk p99 | 单帧最大关节变化（rad/帧） | 越低越平滑 |",
    "| 关节速度 p99 | rad/s | 不宜贴近执行器上限 |",
    "",
  ]

  for subset, key in (("全部 clip", None), ("walk-only 子集", "walk_only")):
    lines += [
      f"## {subset}",
      "",
      "| 管线 | clips | 时长 | 触地率 | 脚滑 | z_min | z_std | 关节越界 | jerk p99 | 速度 p99 | 塌陷条数 |",
      "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for label, entries in rows.items():
      sel = entries if key is None else [e for e in entries if e[key]]
      s = summarize(sel)
      lines.append(
        f"| {label} | {s['clips']} | {s['seconds']:.1f} min | {s['contact']:.1f}% | "
        f"{s['slide']*100:.1f} cm/s | {s['z_min']:.2f} m | {s['z_std']:.1f} cm | "
        f"{s['qlim']:.2f}% | {s['jerk']:.3f} | {s['vel']:.1f} | {s['collapsed']} |"
      )
    lines.append("")

  cats = sorted({e["category"] for es in rows.values() for e in es if e["walk_only"]})
  lines += [
    "## walk-only 分类别（两边都有的类别才列入）",
    "",
    "| 类别 | 管线 | clips | 时长 | 触地率 | 脚滑 | z_min | jerk p99 |",
    "|---|---|---|---|---|---|---|---|",
  ]
  for cat in cats:
    present = [lbl for lbl, es in rows.items() if any(e["category"] == cat for e in es)]
    if len(present) < 2:
      continue
    for label in present:
      sel = [e for e in rows[label] if e["category"] == cat]
      s = summarize(sel)
      lines.append(
        f"| {cat} | {label} | {s['clips']} | {s['seconds']:.1f} min | {s['contact']:.1f}% | "
        f"{s['slide']*100:.1f} cm/s | {s['z_min']:.2f} m | {s['jerk']:.3f} |"
      )

  Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
  print(f"[ok] report -> {args.out}")
  for label, entries in rows.items():
    s = summarize(entries)
    print(
      f"{label:<26} clips={s['clips']:>3} contact={s['contact']:5.1f}% "
      f"slide={s['slide']*100:5.1f}cm/s z_min={s['z_min']:.2f} jerk={s['jerk']:.3f} "
      f"collapsed={s['collapsed']}"
    )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
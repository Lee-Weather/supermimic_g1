"""One-off experiment: does shrinking GMR's human scale table fix the hovering walk?

GMR's effective human scale is ``human_scale_table["Hips"] * (actual_height / 1.8)``
= 0.9 * 0.972 = 0.875, i.e. a 1.75 m human is mapped to a ~1.53 m robot.  The G1 is
1.323 m tall, so the retargeted pelvis sits ~5 cm too high and the feet never reach the
ground.  This script rescales the table by ``--scale`` and reports the resulting sole
profile measured with mjlab's G1 model.
"""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("MUJOCO_GL", "glfw")

import sys
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GMR_ROOT = Path(r"E:\Retargeting\GMR")
GMR_IK = GMR_ROOT / "general_motion_retargeting/ik_configs/bvh_lafan1_to_g1.json"

sys.path.insert(0, str(HERE))
import importlib.util

spec = importlib.util.spec_from_file_location("rp", HERE / "retarget_lafan1_to_g1.py")
rp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rp)


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--clip", required=True)
  ap.add_argument("--scale", type=float, required=True, help="multiplier on human_scale_table")
  ap.add_argument("--out", required=True)
  args = ap.parse_args()

  # Rewrite the IK config with a scaled human_scale_table and point GMR at it.
  cfg = json.loads(GMR_IK.read_text())
  cfg["human_scale_table"] = {
    k: v * args.scale for k, v in cfg["human_scale_table"].items()
  }
  patched = Path(args.out).with_suffix(".ik.json")
  patched.parent.mkdir(parents=True, exist_ok=True)
  patched.write_text(json.dumps(cfg, indent=2))

  pkg = types.ModuleType("general_motion_retargeting")
  pkg.__path__ = [str(GMR_ROOT / "general_motion_retargeting")]
  sys.modules["general_motion_retargeting"] = pkg
  sys.path.insert(0, str(GMR_ROOT))
  from general_motion_retargeting import params
  from general_motion_retargeting.motion_retarget import GeneralMotionRetargeting
  from general_motion_retargeting.utils.lafan1 import load_bvh_file

  params.IK_CONFIG_DICT["bvh_lafan1"]["unitree_g1"] = patched

  kin = rp.G1Kinematics(rp.GMR_G1_XML)
  frames, human_height = load_bvh_file(args.clip, format="lafan1")
  retarget = GeneralMotionRetargeting(
    src_human="bvh_lafan1", tgt_robot="unitree_g1", actual_human_height=human_height
  )
  print(f"[info] human_scale_table x{args.scale} -> Hips="
        f"{cfg['human_scale_table']['Hips']:.3f}")
  qpos = np.stack([retarget.retarget(frame) for frame in frames])
  qpos50 = rp.resample_qpos(qpos, 30.0, 50.0)
  body_pos, body_quat, foot_z = kin.run(qpos50)
  shift = float(foot_z.min())
  body_pos[:, :, 2] -= shift
  qpos50 = qpos50.copy()
  qpos50[:, 2] -= shift
  body_pos[:, :, :2] -= body_pos[0, 0, :2]

  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  dt = 1.0 / 50.0
  np.savez_compressed(
    out,
    joint_pos=qpos50[:, 7:].astype(np.float32),
    joint_vel=np.gradient(qpos50[:, 7:], dt, axis=0).astype(np.float32),
    root_pos_w=body_pos[:, 0, :].astype(np.float32),
    root_quat_w=body_quat[:, 0, :].astype(np.float32),
    root_lin_vel_w=np.gradient(body_pos[:, 0, :], dt, axis=0).astype(np.float32),
    root_ang_vel_w=rp.angular_velocity(body_quat[:, 0, :], dt).astype(np.float32),
    joint_names=np.array(kin.joint_names),
    fps=np.array([50.0], dtype=np.float32),
  )
  print(f"[ok] wrote {out} ({len(qpos50)} frames, scale x{args.scale})")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
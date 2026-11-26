"""Apply rolling grounding to selected clips (writes the npz back in place).

Same transform as the ``after`` variant used for the replays: the root is lowered per frame
by the local minimum sole height (window 0.3 s, measured on mjlab's G1 model), which pins
the stance foot to the ground while keeping the short-term pelvic bob.  ``root_lin_vel_w``
is recomputed in z so the velocity field stays consistent; joints are untouched.

Usage::

    python apply_rolling_ground.py --clips <a.npz> <b.npz> ...        # report only
    python apply_rolling_ground.py --clips ... --apply                # rewrite
"""

from __future__ import annotations

import argparse
import importlib.util
import os

os.environ.setdefault("MUJOCO_GL", "glfw")

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
KEEP = (
  "joint_pos",
  "joint_vel",
  "root_pos_w",
  "root_quat_w",
  "root_lin_vel_w",
  "root_ang_vel_w",
  "joint_names",
  "fps",
)


def load_grounding():
  spec = importlib.util.spec_from_file_location(
    "compare_grounding_variants", HERE / "compare_grounding_variants.py"
  )
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--clips", nargs="+", required=True)
  ap.add_argument("--apply", action="store_true")
  ap.add_argument("--window", type=float, default=0.30)
  args = ap.parse_args()

  gv = load_grounding()
  probe = gv.Probe()

  print(f"{'clip':<46}{'sole p50':>12}{'contact':>18}")
  for clip_arg in args.clips:
    path = Path(clip_arg).expanduser()
    if not path.exists():
      print(f"[skip] {path} missing")
      continue
    base = gv.load_clip(path)
    fixed = gv.rolling_ground(base, probe, window_s=args.window)
    before = probe.soles(gv.qpos_of(base))
    after = probe.soles(gv.qpos_of(fixed))
    print(
      f"{path.stem:<46}{np.percentile(before,50)*100:>+7.1f} -> "
      f"{np.percentile(after,50)*100:>+4.1f} cm"
      f"{100*(before<=0.02).mean():>9.0f}% -> {100*(after<=0.02).mean():>3.0f}%"
    )
    if args.apply:
      with np.load(path) as data:
        payload = {k: data[k] for k in data.files if k in KEEP}
      payload["root_pos_w"] = np.asarray(fixed["root_pos_w"], dtype=np.float32)
      payload["root_lin_vel_w"] = np.asarray(fixed["root_lin_vel_w"], dtype=np.float32)
      np.savez_compressed(path, **payload)

  if args.apply:
    print("\n[applied] rerun pack_motions + build_manifest")
  else:
    print("\n[report only] pass --apply to rewrite the clips")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
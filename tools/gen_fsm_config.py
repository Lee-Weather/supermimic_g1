#!/usr/bin/env python3
"""Generate the SuperMimic FSM config, including the ``FixStand`` entry pose.

Why this script exists: ``State_FixStand`` interpolates joint targets from the pose it
measures at entry (``qs_[0] = q0``) to the waypoints in ``qs``. In the BeyondMimic config
those waypoints are the first frame of a dance reference — useless for us. For SuperMimic
the entry pose has to be a pose the *velocity* policy is comfortable in, so it is taken
from the actor checkpoint's own observation statistics (the mean joint angles of the
training data, i.e. the centre of the distribution), not from any reference clip.

``FixStand`` only sets joint targets; the pelvis height follows physics, so the robot
settles onto its feet by itself.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from supermimic_ckpt import load_state_dict, strip_index_suffix  # noqa: E402

HEADER = """\
# SuperMimic velocity-command state for the G1 EDU U2 (29 DoF).
#
# Operator flow:
#   [L2 + UP]        FixStand: 3 s transition to the SuperMimic entry pose, then hold
#   [L2] 2 s + [RIGHT]  SuperMimicVelocity: exported ONNX policy at 50 Hz
#   edit config/velocity_cmd.yaml   change vx/vy/wz (polled at 10 Hz, then clipped)
#   [L2 + B]        abort to Passive damping at any point
#
# FixStand.qs: row 0 is overwritten at runtime with the measured pose; row 1 is the
# standing target. It is generated from the checkpoint's observation statistics (mean
# joint angles of the training data) by tools/gen_fsm_config.py -- do not hand-edit,
# the policy's zero-action pose is default_joint_pos + 0.5 * action and only a pose
# inside the training distribution keeps the first frames in-distribution.
#
# Velocity goal is body-frame [vx, vy, wz], clamped inside the package to
# vx[-0.10, 1.00], vy[-0.10, 0.10], wz[-1.50, 1.50]. Measured in sim: 1.1 m/s tracks
# within 2 %, backward commands diverge (0.3 -> 0.83 m/s), sideways is weak and
# asymmetric. Keep the gantry attached: damping on abort does not hold the robot up.
"""


def block_passive(source: dict) -> dict:
    return source["FSM"]["Passive"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=".")
    ap.add_argument("--model-dir", default="models/g1_walk_velocity_30k_20261126")
    ap.add_argument("--source-config",
                    default="deploy/G1_deploy/deploy/robots/g1_29dof/config/"
                            "dance2_subject3_30s.yaml",
                    help="BeyondMimic config to borrow Passive kp/kd from")
    ap.add_argument("--targets", nargs="+", default=[
        "deploy/supermimic_patch/robots/g1_29dof/config/supermimic_velocity.yaml",
        "deploy/G1_deploy/deploy/robots/g1_29dof/config/supermimic_velocity.yaml",
    ])
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    source = yaml.safe_load((repo / args.source_config).read_text(encoding="utf-8"))
    weights = strip_index_suffix(load_state_dict(str(repo / args.model_dir / "actor.pt")))
    mean = weights["obs_normalizer._mean"].reshape(-1)
    entry_pose = [round(float(v), 5) for v in mean[10:39]]  # qj block of rep_state
    if len(entry_pose) != 29:
        raise SystemExit("unexpected rep_state layout")

    fix_stand_src = source["FSM"]["FixStand"]
    cfg = {
        "FSM": {
            "_": {
                "Passive": {"id": 1},
                "FixStand": {"id": 2},
                "SuperMimicVelocity": {"id": 301, "type": "SuperMimic"},
            },
            "Passive": dict(block_passive(source)),
            "FixStand": {
                "transitions": {
                    "Passive": "LT + B.on_pressed",
                    "SuperMimicVelocity": "LT + right.on_pressed",
                },
                "kp": [round(float(v), 6) for v in fix_stand_src["kp"]],
                "kd": [round(float(v), 6) for v in fix_stand_src["kd"]],
                "ts": [0, 3],
                # row 0 is a placeholder overwritten by State_FixStand at entry
                "qs": [[], entry_pose],
            },
            "SuperMimicVelocity": {
                "transitions": {"Passive": "LT + B.on_pressed"},
                "policy_dir": "config/policy/supermimic/velocity_v1",
                "command_file": "config/velocity_cmd.yaml",
            },
        }
    }

    text = HEADER + yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
    written = []
    for target in args.targets:
        path = repo / target
        if not path.parent.exists():
            continue
        path.write_text(text, encoding="utf-8")
        written.append(path)

    print("entry pose (training data mean qj):")
    print(" ", [round(v, 3) for v in entry_pose])
    print("kp/kd from Passive/FixStand config, ts = [0, 3] s")
    for path in written:
        print(f"wrote {path.relative_to(repo)} ({path.stat().st_size} bytes)")
    if not written:
        print("[FAIL] no target directory existed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
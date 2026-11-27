"""G1 29-DoF numbers needed by a deployment package, without torch/mjlab imports.

Mirrors ``mjlab_research/src/mjlab/asset_zoo/robots/unitree_g1/g1_constants.py``:
the PD gains are derived from reflected rotor inertia and a 10 Hz natural frequency,
and the actuator→joint assignment follows the same regex table. Kept in pure numpy so
the exporter and the MuJoCo replay harness run on a machine without torch.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

G1_XML = Path(
    "mjlab_research/src/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml"
)

JOINT_ORDER = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

# Actuator families: (rotor inertias, gear ratios, velocity limit, effort limit)
_MOTORS = {
    "5020": ((0.139e-4, 0.017e-4, 0.169e-4), (1, 1 + 46 / 18, 1 + 56 / 16), 37.0, 25.0),
    "7520_14": ((0.489e-4, 0.098e-4, 0.533e-4), (1, 4.5, 1 + 48 / 22), 32.0, 88.0),
    "7520_22": ((0.489e-4, 0.109e-4, 0.738e-4), (1, 4.5, 5), 20.0, 139.0),
    "4010": ((0.068e-4, 0.0, 0.0), (1, 5, 5), 22.0, 5.0),
}

NATURAL_FREQ = 10 * 2.0 * math.pi  # 10 Hz
DAMPING_RATIO = 2.0


def _reflected_inertia(rotor, gears) -> float:
    r1 = rotor[0] * (gears[1] * gears[2]) ** 2
    r2 = rotor[1] * gears[2] ** 2
    r3 = rotor[2]
    return r1 + r2 + r3


ARMATURE = {k: _reflected_inertia(v[0], v[1]) for k, v in _MOTORS.items()}
STIFFNESS = {k: ARMATURE[k] * NATURAL_FREQ**2 for k in _MOTORS}
DAMPING = {k: 2.0 * DAMPING_RATIO * ARMATURE[k] * NATURAL_FREQ for k in _MOTORS}
EFFORT = {k: _MOTORS[k][3] for k in _MOTORS}
VELOCITY = {k: _MOTORS[k][2] for k in _MOTORS}


def _family(joint: str) -> str:
    if joint.endswith(("_wrist_pitch_joint", "_wrist_yaw_joint")):
        return "4010"
    if joint.endswith(("_hip_roll_joint", "_knee_joint")):
        return "7520_22"
    if joint.endswith(("_hip_pitch_joint", "_hip_yaw_joint")) or joint == "waist_yaw_joint":
        return "7520_14"
    # 5020 singles, plus the doubled 4-bar linkages (waist pitch/roll, ankles)
    return "5020"


def _doubled(joint: str) -> bool:
    return joint.endswith(("_ankle_pitch_joint", "_ankle_roll_joint")) or joint in (
        "waist_pitch_joint",
        "waist_roll_joint",
    )


def actuator_table() -> dict[str, dict[str, float]]:
    out = {}
    for joint in JOINT_ORDER:
        fam = _family(joint)
        mult = 2.0 if _doubled(joint) else 1.0
        out[joint] = {
            "family": fam,
            "kp": STIFFNESS[fam] * mult,
            "kd": DAMPING[fam] * mult,
            "effort": EFFORT[fam] * mult,
            "velocity": VELOCITY[fam],
            "armature": ARMATURE[fam] * mult,
        }
    return out


def default_pose() -> tuple[dict[str, float], float]:
    """``KNEES_BENT_KEYFRAME``: the pose mjlab spawns (and ``use_default_offset`` uses)."""
    return (
        {
            "hip_pitch": -0.312,
            "knee": 0.669,
            "ankle_pitch": -0.363,
            "elbow": 0.6,
            "left_shoulder_roll": 0.2,
            "left_shoulder_pitch": 0.2,
            "right_shoulder_roll": -0.2,
            "right_shoulder_pitch": 0.2,
        },
        0.76,
    )


def default_joint_pos() -> list[float]:
    """Expand the pattern-based keyframe into ``JOINT_ORDER``."""
    table, _ = default_pose()
    out = []
    for joint in JOINT_ORDER:
        value = 0.0
        for key, val in table.items():
            if key in ("left_shoulder_roll", "right_shoulder_roll",
                       "left_shoulder_pitch", "right_shoulder_pitch"):
                if joint == f"{key}_joint":
                    value = val
                    break
            elif joint.endswith(f"_{key}_joint"):
                value = val
                break
        out.append(value)
    return out


def joint_limits(xml_path: Path | None = None) -> tuple[list[float], list[float]]:
    """Per-joint ``[lower, upper]`` from the MJCF (order follows ``JOINT_ORDER``)."""
    path = xml_path or G1_XML
    root = ET.parse(path).getroot()
    ranges: dict[str, tuple[float, float]] = {}
    for joint in root.iter("joint"):
        name = joint.get("name")
        rng = joint.get("range")
        if name and rng:
            lo, hi = (float(x) for x in rng.split())
            ranges[name] = (lo, hi)
        elif name:
            ranges[name] = (-math.pi, math.pi)
    lower = [ranges.get(n, (-math.pi, math.pi))[0] for n in JOINT_ORDER]
    upper = [ranges.get(n, (-math.pi, math.pi))[1] for n in JOINT_ORDER]
    return lower, upper


def sdk_indices() -> list[int]:
    """Policy order already equals the G1 SDK motor order (verified against the NPZ joint
    names and the MJCF declaration order), so this mapping is the identity."""
    return list(range(len(JOINT_ORDER)))


if __name__ == "__main__":
    table = actuator_table()
    print(f"{'joint':<28}{'family':<9}{'kp':>9}{'kd':>8}{'effort':>8}")
    for joint in JOINT_ORDER:
        row = table[joint]
        print(f"{joint:<28}{row['family']:<9}{row['kp']:>9.2f}{row['kd']:>8.3f}"
              f"{row['effort']:>8.1f}")
    print("\ndefault pose:", [round(v, 3) for v in default_joint_pos()])
    lo, hi = joint_limits()
    print("limits lo:", [round(v, 2) for v in lo])
    print("limits hi:", [round(v, 2) for v in hi])
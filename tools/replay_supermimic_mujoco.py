#!/usr/bin/env python3
"""Closed-loop MuJoCo replay of an exported SuperMimic ONNX policy.

Deployment-equivalent harness: it rebuilds exactly the observation/action path that
``deploy/include/supermimic/core.h`` implements on the robot

    obs    = [goal(task), rep_state]        rep_state = [z, g_b, v_b, w_b, qj, vj]
    ctrl   = default_joint_pos + 0.5 * action      (implicit MuJoCo position servo)

and drives the mjlab G1 MJCF at 50 Hz, so the video shows the policy through the deploy
contract rather than through the training env.

Implementation notes that cost real debugging time:

* MuJoCo free joints store the linear velocity in the WORLD frame but the angular
  velocity already in the BODY frame — rotating ``qvel[3:6]`` again corrupts ``w_b``.
* The spawn height must be solved by forward kinematics: dropping the robot with a
  bent-knee pose at the reference height penetrates the floor and launches it.
* The actuators are implicit position servos (``set_to_position(kp, kd)``, force
  clamped to the effort limit), which is what mjlab builds from the same numbers.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort
import yaml
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g1_robot_constants as g1  # noqa: E402

FOOT_GEOM_PREFIXES = ("left_foot", "right_foot")


def quat_mat(q: np.ndarray) -> np.ndarray:
    """wxyz quaternion -> 3x3 rotation matrix (body -> world)."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def build_model(xml_path: Path, kp, kd, effort, timestep: float = 0.005):
    spec = mujoco.MjSpec.from_file(str(xml_path))
    spec.option.timestep = timestep
    # mirror mjlab's SimCfg defaults: implicitfast integrator + newton solver is what the
    # policy was trained against (mjlab_research/src/mjlab/envs/sim.py)
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    spec.option.solver = mujoco.mjtSolver.mjSOL_NEWTON
    spec.option.iterations = 100
    spec.option.ls_iterations = 50
    spec.option.impratio = 1.0

    floor = spec.worldbody.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = np.array([0.0, 0.0, 0.05])
    floor.contype = 1
    floor.conaffinity = 1
    floor.friction = np.array([1.0, 0.005, 0.0001])
    floor.rgba = np.array([0.32, 0.35, 0.40, 1.0])

    for i, joint in enumerate(g1.JOINT_ORDER):
        act = spec.add_actuator()
        act.trntype = mujoco.mjtTrn.mjTRN_JOINT
        act.target = joint
        act.name = f"motor_{joint}"
        act.set_to_position(float(kp[i]), float(kd[i]))
        act.gear = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        act.forcelimited = True
        act.forcerange = np.array([-float(effort[i]), float(effort[i])])
    model = spec.compile()
    # mjlab puts the reflected rotor inertia on the joint as armature; without it the
    # servo has no high-frequency mass to push against and the sim blows up.
    table = g1.actuator_table()
    for joint in g1.JOINT_ORDER:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        model.dof_armature[model.jnt_dofadr[jid]] = table[joint]["armature"]
    return model


def place_on_floor(model, data, q0, z_guess: float) -> float:
    """Set the pose and solve the base height so the lowest sole just touches the floor."""
    data.qpos[:] = 0.0
    data.qpos[3] = 1.0
    data.qpos[7:36] = q0
    data.qvel[:] = 0.0
    z = float(z_guess)
    for _ in range(30):
        data.qpos[2] = z
        mujoco.mj_forward(model, data)
        lowest = np.inf
        for gid in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            if name.startswith(FOOT_GEOM_PREFIXES):
                lowest = min(lowest, data.geom_xpos[gid][2] - model.geom_rbound[gid] * 0.0)
        # exact sole height: use the geom's lowest point via rbound is too coarse; instead
        # ask MuJoCo for the contact distance by placing and stepping once is unstable, so
        # use the capsule endpoints directly.
        lowest = np.inf
        for gid in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            if not name.startswith(FOOT_GEOM_PREFIXES):
                continue
            if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_CAPSULE:
                radius = model.geom_size[gid][0]
                half = model.geom_size[gid][1]
                rot = data.geom_xmat[gid].reshape(3, 3)
                axis = rot[:, 2]  # capsule axis is the local z
                for sign in (-1.0, 1.0):
                    point = data.geom_xpos[gid] + axis * half * sign
                    lowest = min(lowest, point[2] - radius)
            elif model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX:
                r = model.geom_size[gid]
                rot = data.geom_xmat[gid].reshape(3, 3)
                corners = np.array([
                    data.geom_xpos[gid] + rot @ (r * np.array(s))
                    for s in ((-1, -1, -1), (-1, 1, -1), (1, -1, -1), (1, 1, -1))
                ])
                lowest = min(lowest, corners[:, 2].min())
        if not np.isfinite(lowest):
            break
        if abs(lowest - 0.001) < 1e-4:
            break
        z += 0.001 - lowest
    data.qpos[2] = z
    mujoco.mj_forward(model, data)
    return z


def parse_phases(text: str) -> list[tuple[int, tuple[float, float, float]]]:
    """``'250@0.4,0,0:250@0.8,0,0'`` -> stages of 250 control steps (5 s) each."""
    phases = []
    for token in text.split(":"):
        token = token.strip()
        if not token:
            continue
        steps, values = token.split("@")
        vx, vy, wz = (float(x) for x in values.split(","))
        phases.append((int(steps), (vx, vy, wz)))
    return phases


def make_mask(lines: list[str], scale: int = 2) -> np.ndarray:
    font = ImageFont.load_default()
    probe = ImageDraw.Draw(Image.new("L", (1, 1)))
    metas = []
    for line in lines:
        box = probe.textbbox((0, 0), line, font=font)
        metas.append((line, box[2] - box[0], box[3] - box[1]))
    width = max(m[1] for m in metas) + 6
    height = sum(m[2] + 5 for m in metas) + 3
    strip = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(strip)
    y = 2
    for line, _, hh in metas:
        draw.text((3, y), line, fill=255, font=font)
        y += hh + 5
    return np.asarray(strip.resize((width * scale, height * scale), Image.NEAREST)) > 127


def overlay(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    mask = make_mask(lines)
    h, w = mask.shape
    h = min(h, frame.shape[0])
    w = min(w, frame.shape[1])
    region = frame[:h, :w].astype(np.float32)
    frame[:h, :w] = np.where(mask[:h, :w, None], np.float32(240.0), region * 0.22).astype(np.uint8)
    return frame


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=".")
    ap.add_argument(
        "--policy-dir",
        default="deploy/G1_deploy/deploy/robots/g1_29dof/config/policy/supermimic/velocity_v1",
    )
    ap.add_argument("--task", default="velocity", choices=["velocity"])
    ap.add_argument(
        "--phases",
        default="250@0.0,0,0:250@0.4,0,0:250@0.4,0,0:250@0.8,0,0:250@0.8,0,0"
        ":250@0.5,0,0:250@0.0,0,0",
        help="'<steps>@vx,vy,wz' stages linearly blended; 250 steps = 5 s at 50 Hz",
    )
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--out", default="replay_out/supermimic_deploy_mujoco.mp4")
    ap.add_argument("--no-overlay", action="store_true")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    policy_dir = repo / args.policy_dir
    cfg = yaml.safe_load((policy_dir / "deploy.yaml").read_text(encoding="utf-8"))
    session = ort.InferenceSession(str(policy_dir / cfg["policy"][args.task]),
                                   providers=["CPUExecutionProvider"])
    print(f"[replay] policy {cfg['policy'][args.task]}  obs_dim={cfg['observation_dim'][args.task]}")

    kp = np.array(cfg["kp"], dtype=np.float64)
    kd = np.array(cfg["kd"], dtype=np.float64)
    effort = np.array(cfg["effort_limit"], dtype=np.float64)
    default_q = np.array(cfg["default_joint_pos"], dtype=np.float64)
    lower = np.array(cfg["joint_lower"], dtype=np.float64)
    upper = np.array(cfg["joint_upper"], dtype=np.float64)
    action_scale = float(cfg["action_scale"])
    clip = cfg["safety"]["velocity_command_clip"]
    step_dt = float(cfg["step_dt"])

    model = build_model(repo / g1.G1_XML, kp, kd, effort)
    data = mujoco.MjData(model)
    substeps = int(round(step_dt / model.opt.timestep))
    print(f"[replay] nq={model.nq} nv={model.nv} nu={model.nu} "
          f"dt={model.opt.timestep} substeps={substeps}")

    # start from a stand-still reference pose so the policy is inside its training set
    stand = None
    for cand in sorted((repo / "datasets/walk_only/npz").glob("*.npz")):
        d = np.load(cand)
        if d["joint_pos"].shape[0] > 100 and np.abs(d["root_lin_vel_w"]).mean() < 0.05:
            stand = d
            break
    q0 = stand["joint_pos"][0] if stand is not None else default_q
    z0 = place_on_floor(model, data, q0, float(stand["root_pos_w"][0, 2]) if stand is not None else 0.75)
    print(f"[replay] spawn z={z0:.3f} (soles on floor)")

    phases = parse_phases(args.phases)
    renderer = mujoco.Renderer(model, args.height, args.width)
    camera = -1
    for i in range(model.ncam):
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i) == "tracking":
            camera = i

    frames: list[np.ndarray] = []
    rows: list[dict] = []
    prev_cmd = (0.0, 0.0, 0.0)
    step_index = 0

    for phase_index, (steps, target) in enumerate(phases):
        positions = []
        for i in range(steps):
            frac = (i + 1) / steps
            cmd = tuple(prev_cmd[j] + (target[j] - prev_cmd[j]) * frac for j in range(3))
            cmd = (
                float(np.clip(cmd[0], *clip["vx"])),
                float(np.clip(cmd[1], *clip["vy"])),
                float(np.clip(cmd[2], *clip["wz"])),
            )

            rot = quat_mat(data.qpos[3:7])
            v_b = rot.T @ data.qvel[0:3]
            w_b = data.qvel[3:6].copy()  # free joint: already body-frame
            g_b = rot.T @ np.array([0.0, 0.0, -1.0])
            rep = np.concatenate([
                [data.qpos[2]], g_b, v_b, w_b, data.qpos[7:36], data.qvel[6:35],
            ]).astype(np.float32)
            obs = np.concatenate([np.asarray(cmd, dtype=np.float32), rep])[None, :]
            action = session.run(["actions"], {"obs": obs})[0][0]

            data.ctrl[:] = np.clip(default_q + action_scale * action, lower + 0.01, upper - 0.01)
            for _ in range(substeps):
                mujoco.mj_step(model, data)

            step_index += 1
            positions.append(data.qpos[:2].copy())
            renderer.update_scene(data, camera=camera)
            frame = renderer.render()
            if not args.no_overlay:
                tilt = np.degrees(np.arccos(np.clip(
                    1 - 2 * (data.qpos[4] ** 2 + data.qpos[5] ** 2), -1.0, 1.0)))
                frame = overlay(frame, [
                    f"CMD  vx {cmd[0]:+.2f}  vy {cmd[1]:+.2f}  wz {cmd[2]:+.2f}",
                    f"ACT  vx {v_b[0]:+.2f}  vy {v_b[1]:+.2f}  wz {w_b[2]:+.2f}",
                    f"phase {phase_index + 1}/{len(phases)}   t {step_index * step_dt:5.1f}s   "
                    f"z {data.qpos[2]:.2f} m   tilt {tilt:4.1f}deg",
                ])
            frames.append(frame)

        arr = np.stack(positions)
        deltas = np.linalg.norm(np.diff(arr, axis=0), axis=1) if len(arr) > 1 else np.zeros(1)
        duration = max((len(arr) - 1) * step_dt, 1e-6)
        mean_speed = float(deltas.sum() / duration)
        print(f"[replay] phase{phase_index:02d} cmd={target[0]:+.2f},{target[1]:+.2f},"
              f"{target[2]:+.2f} steps={steps} mean_speed={mean_speed:.3f} m/s "
              f"disp={np.linalg.norm(arr[-1] - arr[0]):.2f} m z={data.qpos[2]:.2f} m",
              flush=True)
        rows.append({
            "phase": phase_index + 1, "steps": steps,
            "cmd": f"{target[0]:+.2f},{target[1]:+.2f},{target[2]:+.2f}",
            "mean_speed_mps": f"{mean_speed:.3f}",
            "displacement_m": f"{np.linalg.norm(arr[-1] - arr[0]):.2f}",
            "final_height_m": f"{data.qpos[2]:.2f}",
        })
        prev_cmd = target

    renderer.close()
    out = repo / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    import imageio.v2 as imageio

    imageio.mimwrite(str(out), frames, fps=int(round(1.0 / step_dt)), macro_block_size=1)
    csv_path = out.with_name(out.stem + "_phases.csv")
    csv_path.write_text(
        "\n".join([",".join(rows[0])] + [",".join(map(str, r.values())) for r in rows]) + "\n",
        encoding="utf-8",
    )
    print(f"[replay] wrote {out} ({len(frames)} frames = {len(frames) * step_dt:.1f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
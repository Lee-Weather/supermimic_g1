#!/usr/bin/env python3
"""Package a SuperMimic checkpoint as a deployable ONNX policy (+ deploy.yaml + manifest).

Needs no torch: the checkpoint is read straight out of the ``torch.save`` zip with numpy
(see ``tools/supermimic_ckpt.py``), the MLP is re-emitted as ONNX ops, and the two
empirical normalizers are folded into the graph so the C++ side sees raw inputs.

Contract produced here (matches what ``deploy/include/supermimic/package.h`` validates):

    input  ``obs``     float32 [1, goal_dim + 68]   = [goal(task), rep_state]
    output ``actions`` float32 [1, 29]

with ``rep_state = [z, g_b(3), v_b(3), w_b(3), qj(29), vj(29)]`` and
``q_des = default_joint_pos + 0.5 * action``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import yaml
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g1_robot_constants as g1  # noqa: E402
from supermimic_ckpt import (  # noqa: E402
    NORMALIZER_EPS,
    forward_numpy,
    load_state_dict,
    strip_index_suffix,
)

REP_STATE_DIM = 68
GOAL_DIMS = {"velocity": 3, "motion": 640}
ACTION_DIM = 29
ACTION_SCALE = 0.5
LAYER_NORM_EPS = 1e-5  # torch.nn.LayerNorm default
BEYONDMIMIC_YAML = (
    "deploy/G1_deploy/deploy/robots/g1_29dof/config/policy/beyondmimic/"
    "dance2_subject3_30s/deploy.yaml"
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_mlp_nodes(graph, weights, prefix, x_name, out_dim, in_dim):
    """Append ``Linear -> LayerNorm -> GELU`` blocks for one ``build_mlp`` Sequential.

    ``build_mlp`` emits ``Linear(0), LayerNorm(1), GELU, Linear(3), LayerNorm(4), ...``
    (the GELU is stateless, so torch's Sequential skips an index) — walk the parameters
    that actually exist instead of assuming a fixed stride.
    """
    indices = sorted(
        int(k[len(prefix) + 1 : -len(".weight")])
        for k in weights
        if k.startswith(prefix + ".") and k.endswith(".weight")
    )
    node_out = x_name
    for i in indices:
        w_raw = weights[f"{prefix}.{i}.weight"]
        b_raw = weights[f"{prefix}.{i}.bias"].reshape(-1)
        if w_raw.ndim == 2:  # Linear: MatMul(x, W^T) + b (Add broadcasts a 1-D bias)
            w = numpy_helper.from_array(
                np.ascontiguousarray(w_raw.T.astype(np.float32)), name=f"{prefix}_w{i}"
            )
            b = numpy_helper.from_array(b_raw.astype(np.float32), name=f"{prefix}_b{i}")
            graph.initializer.extend([w, b])
            mm_out = f"{prefix}_mm{i}"
            graph.node.append(
                helper.make_node("MatMul", [node_out, w.name], [mm_out],
                                 name=f"{prefix}_mm{i}")
            )
            lin_out = f"{prefix}_lin{i}"
            graph.node.append(
                helper.make_node("Add", [mm_out, b.name], [lin_out],
                                 name=f"{prefix}_lin{i}")
            )
            node_out = lin_out
            continue
        # LayerNorm: weight/bias are 1-D gains
        g = numpy_helper.from_array(w_raw.reshape(-1).astype(np.float32), name=f"{prefix}_g{i}")
        beta = numpy_helper.from_array(b_raw.astype(np.float32), name=f"{prefix}_lnb{i}")
        graph.initializer.extend([g, beta])
        ln_out = f"{prefix}_ln{i}"
        graph.node.append(
            helper.make_node(
                "LayerNormalization", [node_out, g.name, beta.name], [ln_out],
                name=f"{prefix}_ln{i}", axis=-1, epsilon=LAYER_NORM_EPS,
            )
        )
        # exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
        for op, inputs, out in (
            ("Div", [ln_out, "sqrt2"], f"{prefix}_d{i}"),
            ("Erf", [f"{prefix}_d{i}"], f"{prefix}_e{i}"),
            ("Add", [f"{prefix}_e{i}", "one"], f"{prefix}_a{i}"),
            ("Mul", [ln_out, "half"], f"{prefix}_h{i}"),
            ("Mul", [f"{prefix}_h{i}", f"{prefix}_a{i}"], f"{prefix}_gelu{i}"),
        ):
            graph.node.append(helper.make_node(op, inputs, [out], name=out))
        node_out = f"{prefix}_gelu{i}"
    return node_out


def export(task: str, weights: dict[str, np.ndarray], out_dir: Path) -> dict:
    goal_dim = GOAL_DIMS[task]
    obs_dim = goal_dim + REP_STATE_DIM
    out_path = out_dir / f"policy_{task}.onnx"

    graph = helper.make_graph(
        [],
        f"supermimic_{task}",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, obs_dim])],
        [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, ACTION_DIM])],
    )
    for const_name, value in (("sqrt2", np.sqrt(2.0)), ("one", 1.0), ("half", 0.5)):
        graph.initializer.append(
            numpy_helper.from_array(np.array(value, dtype=np.float32), name=const_name)
        )

    # 1) split [goal, rep_state]; opset 17 takes the split sizes as an input tensor
    graph.initializer.append(
        numpy_helper.from_array(
            np.array([goal_dim, REP_STATE_DIM], dtype=np.int64), name="split_sizes"
        )
    )
    graph.node.append(
        helper.make_node(
            "Split", ["obs", "split_sizes"], ["goal_raw", "rep_raw"], name="split", axis=1
        )
    )
    # 2) fold both empirical normalizers: (x - mean) / (std + eps)
    for tag, raw_name, prefix in (
        (task, "goal_raw", f"goal_normalizers.{task}"),
        ("obs", "rep_raw", "obs_normalizer"),
    ):
        mean = weights[f"{prefix}._mean"].reshape(-1).astype(np.float32)
        std = (weights[f"{prefix}._std"].reshape(-1) + NORMALIZER_EPS).astype(np.float32)
        graph.initializer.append(numpy_helper.from_array(mean, name=f"{tag}_mean"))
        graph.initializer.append(numpy_helper.from_array(std, name=f"{tag}_std"))
        graph.node.append(
            helper.make_node("Sub", [raw_name, f"{tag}_mean"], [f"{tag}_sub"], name=f"{tag}_sub")
        )
        graph.node.append(
            helper.make_node("Div", [f"{tag}_sub", f"{tag}_std"], [f"{tag}_norm"], name=f"{tag}_norm")
        )

    # 3) encoder -> tanh latent
    enc_out = build_mlp_nodes(graph, weights, f"encoders.{task}", f"{task}_norm",
                              weights[f"encoders.{task}.12.bias"].size, goal_dim)
    graph.node.append(helper.make_node("Tanh", [enc_out], ["latent"], name="latent"))

    # 4) decoder over [latent, rep_state_normalized]
    graph.node.append(
        helper.make_node("Concat", ["latent", "obs_norm"], ["dec_in"], name="concat", axis=1)
    )
    dec_out = build_mlp_nodes(graph, weights, "decoder", "dec_in",
                              ACTION_DIM, weights["decoder.0.weight"].shape[1])
    graph.node.append(helper.make_node("Identity", [dec_out], ["actions"], name="actions"))

    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)],
        producer_name="supermimic_export",
    )
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, out_path)

    # parity: independent numpy reference vs onnxruntime
    rng = np.random.default_rng(20261127)
    session = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    max_err = 0.0
    for _ in range(64):
        obs = np.zeros(obs_dim, dtype=np.float32)
        obs[0:goal_dim] = rng.uniform(-1.0, 1.0, goal_dim)
        obs[goal_dim + 0] = rng.uniform(0.60, 0.90)          # z
        obs[goal_dim + 1:goal_dim + 4] = rng.normal(0, 0.3, 3)   # g_b
        obs[goal_dim + 4:goal_dim + 7] = rng.normal(0, 0.4, 3)   # v_b
        obs[goal_dim + 7:goal_dim + 10] = rng.normal(0, 0.6, 3)  # w_b
        obs[goal_dim + 10:goal_dim + 39] = rng.normal(0, 0.5, ACTION_DIM)  # qj
        obs[goal_dim + 39:goal_dim + 68] = rng.normal(0, 1.0, ACTION_DIM)  # vj
        got = session.run(["actions"], {"obs": obs[None, :]})[0][0]
        ref = forward_numpy(weights, task, obs)
        max_err = max(max_err, float(np.abs(got - ref).max()))
    print(f"  {task}: obs_dim={obs_dim} parity max_abs_err={max_err:.3e}")
    return {
        "task": task,
        "policy": out_path.name,
        "obs_dim": obs_dim,
        "goal_dim": goal_dim,
        "parity_max_abs_err": max_err,
        "policy_sha256": sha256(out_path),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=".")
    ap.add_argument("--model-dir", default="models/g1_walk_velocity_30k_20261126")
    ap.add_argument("--tasks", nargs="+", default=["velocity"], choices=["velocity", "motion"])
    ap.add_argument(
        "--output",
        default="deploy/G1_deploy/deploy/robots/g1_29dof/config/policy/supermimic/velocity_v1",
    )
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    model_dir = repo / args.model_dir
    out_dir = repo / args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    weights = strip_index_suffix(load_state_dict(str(model_dir / "actor.pt")))
    print(f"loaded {len(weights)} tensors from {model_dir/'actor.pt'}")

    exported = [export(task, weights, out_dir) for task in args.tasks]

    table = g1.actuator_table()
    default_q = g1.default_joint_pos()
    lower, upper = g1.joint_limits(repo / g1.G1_XML)

    beyond = yaml.safe_load((repo / BEYONDMIMIC_YAML).read_text(encoding="utf-8"))
    config = {
        "format": "supermimic_g1_gencoord_v1",
        "policy": {e["task"]: e["policy"] for e in exported},
        "step_dt": 0.02,
        "action_dim": ACTION_DIM,
        "action_scale_mode": "fixed",
        "action_scale": ACTION_SCALE,
        "rep_state_dim": REP_STATE_DIM,
        "rep_state_layout": ["z", "g_b", "v_b", "w_b", "qj", "vj"],
        "observation_dim": {e["task"]: e["obs_dim"] for e in exported},
        "goal_dim": {e["task"]: e["goal_dim"] for e in exported},
        "goal_layout": {
            "velocity": ["vx_b", "vy_b", "wz_b"],
            "motion": "10 frames x stride 5 of [qj(29), vj(29)] then 10 x rot6d(60)",
        },
        "anchor_body_name": "pelvis",
        "joint_names": g1.JOINT_ORDER,
        "sdk_indices": g1.sdk_indices(),
        "default_joint_pos": [round(v, 6) for v in default_q],
        "kp": [round(table[j]["kp"], 6) for j in g1.JOINT_ORDER],
        "kd": [round(table[j]["kd"], 6) for j in g1.JOINT_ORDER],
        "effort_limit": [table[j]["effort"] for j in g1.JOINT_ORDER],
        "velocity_limit": [table[j]["velocity"] for j in g1.JOINT_ORDER],
        "joint_lower": lower,
        "joint_upper": upper,
        "safety": {
            "max_tilt_rad": 0.85,
            "max_anchor_error_m": 0.35,
            "max_target_speed_rad_s": 8.0,
            "velocity_command_clip": {"vx": [-0.1, 1.0], "vy": [-0.1, 0.1], "wz": [-1.5, 1.5]},
        },
        "kinematics": beyond["kinematics"],
        "normalizer": {"embedded": True, "epsilon": NORMALIZER_EPS},
    }
    (out_dir / "deploy.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    manifest = {
        "checkpoint": "actor.pt",
        "checkpoint_path": str(model_dir / "actor.pt"),
        "checkpoint_sha256": sha256(model_dir / "actor.pt"),
        "tasks": exported,
        "joint_order": "policy == MJCF == G1 SDK motor order (identity sdk_indices)",
        "action_scale": ACTION_SCALE,
        "normalizer_epsilon": NORMALIZER_EPS,
        "deploy_yaml_sha256": sha256(out_dir / "deploy.yaml"),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out_dir}/{{deploy.yaml,manifest.json}}")
    print(json.dumps({e['task']: f"{e['policy_sha256'][:12]} obs={e['obs_dim']}" for e in exported},
                     indent=2))
    worst = max(e["parity_max_abs_err"] for e in exported)
    if worst > 2e-4:
        print(f"[FAIL] parity {worst:.3e} exceeds 2e-4")
        return 1
    print(f"[PASS] parity {worst:.3e} <= 2e-4")
    return 0


if __name__ == "__main__":
    sys.exit(main())
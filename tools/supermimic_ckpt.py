"""Read a SuperMimic ``.pt`` checkpoint into numpy arrays, without torch.

A ``torch.save`` file is a zip: ``archive/data.pkl`` holds the pickled state_dict and each
tensor's raw bytes live in ``archive/data/<storage_key>``. We walk the pickle to recover
``name -> (storage_key, offset, shape, stride)`` and then rebuild the arrays with numpy, so
the exporter can run on a machine that has no torch (and no GPU).

Shared by ``export_supermimic.py`` and the MuJoCo replay harness.
"""

from __future__ import annotations

import pickle
import zipfile
from typing import Any

import numpy as np

NORMALIZER_EPS = 1e-2  # rsl_rl EmpiricalNormalization default; forward is (x-mean)/(std+eps)


class _Storage:
    __slots__ = ("key", "numel")

    def __init__(self, key: str, numel: int) -> None:
        self.key = key
        self.numel = numel


class _Rebuild:
    """Placeholder for torch's ``_rebuild_tensor_v2``; keeps its args."""

    __slots__ = ("args",)

    def __init__(self, *args: Any) -> None:
        self.args = args


class _Unpickler(pickle.Unpickler):
    def persistent_load(self, pid):  # noqa: D102
        # ('storage', storage_type, key, location, numel)
        return _Storage(str(pid[2]), int(pid[4]))

    def find_class(self, module, name):  # noqa: D102
        if module.startswith("torch"):
            return _Rebuild
        return super().find_class(module, name)


def _archive_prefix(zf: zipfile.ZipFile) -> str:
    for name in zf.namelist():
        if name.endswith("/data.pkl"):
            return name[: -len("data.pkl")]
    raise ValueError("not a torch zip archive (no */data.pkl)")


def load_state_dict(path: str) -> dict[str, np.ndarray]:
    """``name -> ndarray`` for every tensor and buffer in the checkpoint."""
    with zipfile.ZipFile(path) as zf:
        prefix = _archive_prefix(zf)
        with zf.open(prefix + "data.pkl") as fh:
            obj = _Unpickler(fh).load()

        out: dict[str, np.ndarray] = {}
        cache: dict[str, np.ndarray] = {}

        def array_for(storage: _Storage, offset: int, shape, stride):
            if storage.key not in cache:
                raw = zf.read(f"{prefix}data/{storage.key}")
                itemsize = len(raw) // storage.numel if storage.numel else 4
                dtype = {4: np.float32, 8: np.float64}[itemsize]
                cache[storage.key] = np.frombuffer(raw, dtype=dtype)
            flat = cache[storage.key]
            shape = tuple(int(s) for s in shape)
            if not shape:
                return flat[offset : offset + 1].reshape(())
            if tuple(int(s) for s in stride) == tuple(
                int(np.prod(shape[i + 1 :], dtype=np.int64)) for i in range(len(shape))
            ):
                # contiguous: ignore the offset view trick below
                pass
            index = np.lib.stride_tricks.as_strided(
                flat[offset:],
                shape=shape,
                strides=tuple(int(s) * flat.itemsize for s in stride),
                writeable=False,
            )
            return np.array(index)

        def walk(node: Any, prefix_name: str) -> None:
            if isinstance(node, dict):
                for key, value in node.items():
                    child = f"{prefix_name}.{key}" if prefix_name else str(key)
                    walk(value, child)
                return
            if isinstance(node, _Rebuild):
                args = node.args
                storage, offset, shape, stride = args[0], args[1], args[2], args[3]
                if isinstance(storage, _Storage):
                    out[prefix_name] = array_for(storage, int(offset), shape, stride)

        walk(obj, "")
        return out


def strip_index_suffix(weights: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Our quick-and-dirty unpickler tags rebuild args with ``<i>``; drop that noise."""

    def clean(name: str) -> str:
        while "<" in name and name.endswith(">"):
            name = name[: name.rindex("<")]
        return name

    return {clean(k): v for k, v in weights.items()}


def normalize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """rsl_rl EmpiricalNormalization forward: ``(x - mean) / (std + eps)``."""
    return (x - mean) / (std + NORMALIZER_EPS)


def mlp_layers(weights: dict[str, np.ndarray], prefix: str):
    """Yield ``(kind, index, params)`` for a ``build_mlp`` Sequential (Linear/LN/GELU)."""
    layers = []
    index = 0
    while f"{prefix}.{index}.weight" in weights:
        w = weights[f"{prefix}.{index}.weight"]
        b = weights[f"{prefix}.{index}.bias"]
        layers.append(("linear", index, (w, b)))
        index += 1
        if f"{prefix}.{index}.weight" in weights:
            g = weights[f"{prefix}.{index}.weight"]
            beta = weights[f"{prefix}.{index}.bias"]
            layers.append(("layernorm", index, (g, beta)))
            index += 1
            layers.append(("gelu", index, None))
            index += 1
    return layers


def gelu(x: np.ndarray) -> np.ndarray:
    """Exact GELU (torch default): 0.5 * x * (1 + erf(x / sqrt(2)))."""
    from math import erf, sqrt

    return 0.5 * x * (1.0 + np.vectorize(erf)(x / sqrt(2.0)))


def layer_norm(x: np.ndarray, g: np.ndarray, beta: np.ndarray, eps: float = 1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return ((x - mu) / np.sqrt(var + eps)) * g + beta


def forward_numpy(weights: dict[str, np.ndarray], task: str, obs: np.ndarray) -> np.ndarray:
    """Reference implementation of ``MultiEncoderAE.forward`` for one sample.

    ``obs`` = ``[goal(task_dims), rep_state(68)]``.
    """
    goal_dim = {"velocity": 3, "motion": 640}[task]
    goal = obs[None, :goal_dim]
    rep = obs[None, goal_dim:]
    goal_n = normalize(goal, weights[f"goal_normalizers.{task}._mean"],
                       weights[f"goal_normalizers.{task}._std"])
    rep_n = normalize(rep, weights["obs_normalizer._mean"], weights["obs_normalizer._std"])

    x = goal_n
    for kind, _, params in mlp_layers(weights, f"encoders.{task}"):
        if kind == "linear":
            x = x @ params[0].T + params[1]
        elif kind == "layernorm":
            x = layer_norm(x, params[0], params[1])
        else:
            x = gelu(x)
    latent = np.tanh(x)  # MultiEncoderAE applies tanh to the encoder output

    x = np.concatenate([latent, rep_n], axis=-1)
    for kind, _, params in mlp_layers(weights, "decoder"):
        if kind == "linear":
            x = x @ params[0].T + params[1]
        elif kind == "layernorm":
            x = layer_norm(x, params[0], params[1])
        else:
            x = gelu(x)
    return x[0]
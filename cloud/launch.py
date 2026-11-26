"""Cloud launcher shim for the Gradmotion/flux platform.

Why a Python shim: the platform's SDK takes the task's ``startScript``, strips the leading
``gm-run`` and executes the result as ``<image-python> <second-token> <rest...>``. A shell
script (``bash``) as the second token therefore fails with
``can't open file '/workspace/.../bash'``. This file is a real Python script, so it is what
the platform expects; it then does the heavy lifting: run ``cloud/bootstrap.sh`` once, and
launch training with the venv python that bootstrap created (Python 3.13 + torch).

The venv deliberately lives **outside** the workspace (``$SUPERMIMIC_VENV``, default
``~/.cache/supermimic-venv``): the platform's SDK watches the workspace and uploads every new
file, and a 30k-file venv inside it floods the upload queue and the console log.

Standalone use (also handy locally)::

    python cloud/launch.py --probe
    python cloud/launch.py --num-envs 1024 --iters 100000 --save-interval 2000
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV_DIR = Path(
  os.environ.get("SUPERMIMIC_VENV", str(Path.home() / ".cache/supermimic-venv"))
)
VENV_PY = VENV_DIR / "bin/python"
TRAIN = ROOT / "supermimic/src/supermimic/scripts/train.py"
BOOTSTRAP = ROOT / "cloud/bootstrap.sh"
TASK_ID = "Supermimic-GenCoord-G1".replace("Supermimic", "SuperMimic")

# The pod's shell is polluted by the image's Isaac Sim / conda environment: PYTHONHOME and
# PYTHONPATH point at IsaacLab's bundled stdlib, which makes any other interpreter abort with
# "AssertionError: SRE module mismatch" (3.13 binary loading a 3.11 stdlib). Strip them for
# every child process we spawn.
POLLUTED_KEYS = (
  "PYTHONHOME",
  "PYTHONPATH",
  "PYTHONSTARTUP",
  "PYTHONEXECUTABLE",
  "PYTHONUSERBASE",
  "PYTHONWARNINGS",
  "CONDA_PREFIX",
  "CONDA_DEFAULT_ENV",
  "CONDA_PROMPT_MODIFIER",
  "VIRTUAL_ENV",
  "LD_LIBRARY_PATH",
)


# Extra environment handed to the trainer, filled from the CLI in main().
TASK_ENV: dict[str, str] = {}


def clean_env() -> dict[str, str]:
  env = {k: v for k, v in os.environ.items() if k not in POLLUTED_KEYS}
  env["PYTHONNOUSERSITE"] = "1"
  env["PYTHONUNBUFFERED"] = "1"
  env["SUPERMIMIC_VENV"] = str(VENV_DIR)
  env.setdefault("WANDB_MODE", "offline")
  env.update(TASK_ENV)
  return env


def default_motion() -> str:
  """mjlab treats a directory as a *packed* motion dir (needs motion_ends.npy); per-clip
  .npz files must be passed as a glob. Prefer the packed dir when bootstrap built it."""
  packed = ROOT / "datasets/walk_only/packed"
  if packed.is_dir():
    return str(packed)
  return str(ROOT / "datasets/walk_only/npz/*.npz")


def probe() -> int:
  print("=== launch.py probe ===", flush=True)
  print("cwd        :", os.getcwd())
  print("repo root  :", ROOT)
  print("python     :", sys.version.split()[0], sys.executable)
  print("venv dir   :", VENV_DIR, "python exists:", VENV_PY.exists())
  print("train.py   :", TRAIN, "exists:", TRAIN.exists())
  print("PYTHONHOME :", os.environ.get("PYTHONHOME", "<unset>"))
  print("PYTHONPATH :", os.environ.get("PYTHONPATH", "<unset>"))
  print("--- repo root listing ---", flush=True)
  for p in sorted(ROOT.iterdir())[:20]:
    print("  ", p.name, "(dir)" if p.is_dir() else p.stat().st_size)
  print("--- tools ---", flush=True)
  for tool in ("python3", "python", "bash", "git", "curl", "wget", "nvidia-smi", "uv"):
    found = subprocess.run(
      ["bash", "-lc", f"command -v {tool}"], capture_output=True, text=True
    ).stdout.strip()
    print(f"  {tool:<12} -> {found or 'MISSING'}")
  print("=== probe done ===", flush=True)
  return 0


def run(cmd: list[str]) -> int:
  print("+ " + " ".join(cmd), flush=True)
  return subprocess.call(cmd, cwd=str(ROOT), env=clean_env())


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--probe", action="store_true")
  ap.add_argument("--num-envs", type=int, default=1024)
  ap.add_argument("--iters", type=int, default=100000)
  ap.add_argument("--save-interval", type=int, default=2000)
  ap.add_argument("--motion-file", default=default_motion())
  ap.add_argument("--skip-bootstrap", action="store_true")
  # Task-slot mix: default is motion-only. --velocity-mix 0.5 trains the
  # downstream velocity-command head alongside motion tracking.
  ap.add_argument("--motion-mix", type=float, default=None)
  ap.add_argument("--velocity-mix", type=float, default=None)
  # "conservative" narrows the velocity command ranges to what walk-only data covers.
  ap.add_argument("--vel-range", choices=("wide", "conservative"), default=None)
  args, extra = ap.parse_known_args()

  if args.probe:
    return probe()

  if args.velocity_mix is not None:
    motion_mix = args.motion_mix if args.motion_mix is not None else 1.0 - args.velocity_mix
    TASK_ENV["SM_MIX_MOTION"] = str(motion_mix)
    TASK_ENV["SM_MIX_VELOCITY"] = str(args.velocity_mix)
    print(f"[launch] slot_mix motion={motion_mix} velocity={args.velocity_mix}", flush=True)
  if args.vel_range is not None:
    TASK_ENV["SM_VEL_RANGE"] = args.vel_range
    print(f"[launch] velocity command range = {args.vel_range}", flush=True)

  if not VENV_PY.exists() and not args.skip_bootstrap:
    print(f"[launch] venv missing at {VENV_DIR} -> bootstrap (one-time, ~10 min)", flush=True)
    code = run(["bash", str(BOOTSTRAP)])
    if code != 0:
      print(f"[launch] bootstrap failed with {code}", flush=True)
      return code

  if not VENV_PY.exists():
    print("[launch] no venv python; abort", flush=True)
    return 2

  cmd = [
    str(VENV_PY), "-u", str(TRAIN), TASK_ID,
    "--num-envs", str(args.num_envs),
    "--motion-file", args.motion_file,
    "--runner.max-iterations", str(args.iters),
    "--runner.save-interval", str(args.save_interval),
    "--runner.headless", "True",
    *extra,
  ]
  return run(cmd)


if __name__ == "__main__":
  sys.exit(main())
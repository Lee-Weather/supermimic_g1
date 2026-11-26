#!/usr/bin/env bash
# Cloud bootstrap for SuperMimic on a fresh Gradmotion/flux pod.
#
# Why this script exists: the official images top out at Python 3.11 while mjlab needs 3.13,
# and the CUDA stack (mujoco-warp / warp-lang / torch-cu126) must match the pod driver.
# Everything is installed into a user-space venv, no sudo required.
#
# Usage (from the repo root, after `git clone`):
#   bash cloud/bootstrap.sh [--cu126|--cu128|--cpu]
#
# Exits non-zero on the first failed check, so the platform marks the task failed instead of
# burning GPU hours on a broken environment.

set -euo pipefail

BACKEND="${1:---cu126}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/supermimic/.venv"
export WANDB_MODE="${WANDB_MODE:-offline}"
export DEBIAN_FRONTEND=noninteractive
export PYTHONUNBUFFERED=1

ts() { date '+%H:%M:%S'; }
say() { echo "[$(ts)] $*"; }

say "=== 0. host info ==="
nvidia-smi || { echo "[FATAL] no GPU / no driver"; exit 1; }
df -h "$ROOT" | tail -1

say "=== 0b. basic tooling ==="
for bin in curl git; do
  if ! command -v "$bin" >/dev/null 2>&1; then
    say "missing $bin -> trying apt-get"
    if command -v apt-get >/dev/null 2>&1 && [ "$(id -u)" = "0" ]; then
      apt-get update -qq && apt-get install -y -qq "$bin"
    else
      echo "[FATAL] $bin missing and cannot install (no root / no apt)"; exit 1
    fi
  fi
done

say "=== 1. install uv (user space) ==="
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

say "=== 2. python 3.13 venv ==="
# only-managed: never reuse the pod's 3.11 (Isaac Sim) interpreter as "3.13".
uv venv --python 3.13 --python-preference only-managed "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -c "import sys; print('python', sys.version.split()[0], sys.executable)"

say "=== 3. torch (>=2.14 required by mjlab) from a wheel index that works ==="
install_torch() {
  local idx="$1"
  say "trying torch from $idx"
  uv pip install --python "$VENV" torch --index-url "$idx" || return 1
  "$VENV/bin/python" - <<'PY'
import torch
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available(), flush=True)
assert torch.__version__ >= "2.14", "torch too old for mjlab"
PY
}
case "$BACKEND" in
  --cu126) CANDIDATES=("https://download.pytorch.org/whl/cu126" "https://download.pytorch.org/whl/cu128") ;;
  --cu128) CANDIDATES=("https://download.pytorch.org/whl/cu128" "https://download.pytorch.org/whl/cu126") ;;
  --cpu)   CANDIDATES=("https://download.pytorch.org/whl/cpu") ;;
  *) echo "[FATAL] unknown backend $BACKEND"; exit 2 ;;
esac
for IDX in "${CANDIDATES[@]}"; do
  if install_torch "$IDX"; then TORCH_OK=1; break; fi
  say "torch install from $IDX failed, trying next"
done
[ "${TORCH_OK:-0}" = "1" ] || { echo "[FATAL] no torch wheel worked"; exit 3; }

say "=== 4. mjlab + supermimic (editable; nvidia index for mujoco-warp) ==="
uv pip install --python "$VENV" \
  --extra-index-url https://pypi.nvidia.com \
  -e "$ROOT/mjlab_research" \
  -e "$ROOT/supermimic"

say "=== 5. verify the CUDA stack ==="
python -u - <<'PY'
import torch, warp, mujoco
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available(),
      "cuda", torch.version.cuda, "gpu", torch.cuda.get_device_name(0), flush=True)
print("warp", warp.__version__, flush=True)
print("mujoco", mujoco.__version__, flush=True)
import mjlab.tasks  # noqa: F401
import supermimic.tasks  # noqa: F401
print("mjlab + supermimic import OK", flush=True)
PY

say "=== 6. pack the motion dataset (npz -> packed, ~20 s) ==="
if [ -d "$ROOT/datasets/walk_only/npz" ] && [ ! -d "$ROOT/datasets/walk_only/packed" ]; then
  python -u "$ROOT/mjlab_research/src/mjlab/scripts/pack_motions.py" \
    --input "$ROOT/datasets/walk_only/npz/*.npz" \
    --output "$ROOT/datasets/walk_only/packed"
fi

say "=== 7. smoke: 20 iterations, 64 envs ==="
cd "$ROOT"
python -u supermimic/src/supermimic/scripts/train.py SuperMimic-GenCoord-G1 \
  --num-envs 64 \
  --motion-file "$ROOT/datasets/walk_only/npz" \
  --runner.max-iterations 20 \
  --runner.headless \
  || { echo "[FATAL] smoke training failed"; exit 1; }

say "=== bootstrap OK ==="
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
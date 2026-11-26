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

echo "=== 0. host info ==="
nvidia-smi || { echo "[FATAL] no GPU / no driver"; exit 1; }
df -h "$ROOT" | tail -1

echo "=== 1. install uv (user space) ==="
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

echo "=== 2. python 3.13 venv ==="
uv venv --python 3.13 "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -c "import sys; print('python', sys.version.split()[0])"

echo "=== 3. torch ($BACKEND) from the matching wheel index ==="
case "$BACKEND" in
  --cu126) IDX="https://download.pytorch.org/whl/cu126" ;;
  --cu128) IDX="https://download.pytorch.org/whl/cu128" ;;
  --cpu)   IDX="https://download.pytorch.org/whl/cpu"   ;;
  *) echo "[FATAL] unknown backend $BACKEND"; exit 2 ;;
esac
uv pip install --python "$VENV" torch --index-url "$IDX"

echo "=== 4. mjlab + supermimic (editable; nvidia index for mujoco-warp) ==="
uv pip install --python "$VENV" \
  --extra-index-url https://pypi.nvidia.com \
  -e "$ROOT/mjlab_research" \
  -e "$ROOT/supermimic"

echo "=== 5. verify the CUDA stack ==="
python - <<'PY'
import torch, warp, mujoco
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available(),
      "cuda", torch.version.cuda, "gpu", torch.cuda.get_device_name(0))
print("warp", warp.__version__)
print("mujoco", mujoco.__version__)
import mjlab.tasks  # noqa: F401
import supermimic.tasks  # noqa: F401
print("mjlab + supermimic import OK")
PY

echo "=== 6. pack the motion dataset (npz -> packed, ~20 s) ==="
if [ -d "$ROOT/datasets/walk_only/npz" ] && [ ! -d "$ROOT/datasets/walk_only/packed" ]; then
  python "$ROOT/mjlab_research/src/mjlab/scripts/pack_motions.py" \
    --input "$ROOT/datasets/walk_only/npz/*.npz" \
    --output "$ROOT/datasets/walk_only/packed"
fi

echo "=== 7. smoke: 20 iterations, 64 envs ==="
cd "$ROOT"
python supermimic/src/supermimic/scripts/train.py SuperMimic-GenCoord-G1 \
  --num-envs 64 \
  --motion-file "$ROOT/datasets/walk_only/npz" \
  --runner.max-iterations 20 \
  --runner.headless \
  || { echo "[FATAL] smoke training failed"; exit 1; }

echo "=== bootstrap OK ==="
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
#!/usr/bin/env bash
# Training entry for the Gradmotion/flux platform.
#
# The platform requires the task's startScript to begin with `gm-run`, so point it at this file:
#   gm-run cloud/start_train.sh --num-envs 1024 --iters 120000
#
# Only assumes the venv created by cloud/bootstrap.sh exists; installs nothing, so a retry
# after a crash starts in seconds.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/supermimic/.venv"
NUM_ENVS=1024
ITERS=120000
MOTION="${MOTION_FILE:-$ROOT/datasets/walk_only/npz}"
SAVE_INTERVAL=2000
export WANDB_MODE="${WANDB_MODE:-offline}"
EXTRA_ARGS=()

while [ $# -gt 0 ]; do
  case "$1" in
    --num-envs)      NUM_ENVS="$2"; shift 2 ;;
    --iters)         ITERS="$2"; shift 2 ;;
    --motion-file)   MOTION="$2"; shift 2 ;;
    --save-interval) SAVE_INTERVAL="$2"; shift 2 ;;
    *)               EXTRA_ARGS+=("$1"); shift ;;
  esac
done

if [ ! -x "$VENV/bin/python" ]; then
  echo "[info] venv missing -> running cloud/bootstrap.sh first (one-time, ~10 min)"
  bash "$ROOT/cloud/bootstrap.sh" "${BOOTSTRAP_BACKEND:---cu126}"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo "=== training ==="
echo "num_envs     : $NUM_ENVS"
echo "iterations   : $ITERS"
echo "motion       : $MOTION"
echo "save every   : $SAVE_INTERVAL iters"
echo "wandb mode   : $WANDB_MODE"
nvidia-smi --query-gpu=memory.total,driver_version --format=csv,noheader

python "$ROOT/supermimic/src/supermimic/scripts/train.py" SuperMimic-GenCoord-G1 \
  --num-envs "$NUM_ENVS" \
  --motion-file "$MOTION" \
  --runner.max-iterations "$ITERS" \
  --runner.save-interval "$SAVE_INTERVAL" \
  --runner.headless \
  "${EXTRA_ARGS[@]}"
#!/usr/bin/env bash
# Training entry for the Gradmotion/flux platform.
#
# The platform requires the task's startScript to begin with `gm-run`. Use the bash form so a
# missing exec bit cannot break the launch:
#   gm-run bash supermimic_g1/cloud/start_train.sh --num-envs 1024 --iters 100000
#
# Everything is tee'd to $LOG so the run can be inspected even if the platform's console
# stream is empty:  tail -f /tmp/supermimic_train.log

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/supermimic/.venv"
LOG="${TRAIN_LOG:-/tmp/supermimic_train.log}"
NUM_ENVS=1024
ITERS=100000
MOTION="${MOTION_FILE:-$ROOT/datasets/walk_only/npz}"
SAVE_INTERVAL=2000
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONUNBUFFERED=1
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

mkdir -p "$(dirname "$LOG")"
# Mirror everything to the platform console and to a file that survives log-stream problems.
exec > >(tee -a "$LOG") 2>&1
set -x
trap 'echo "[TRAP] exit=$? at $(date +%H:%M:%S); log is $LOG"; tail -n 40 "$LOG" || true' EXIT

echo "[$(date +%H:%M:%S)] === start_train.sh ==="
echo "ROOT=$ROOT"
echo "pwd=$(pwd)"
echo "num_envs=$NUM_ENVS iters=$ITERS motion=$MOTION save_interval=$SAVE_INTERVAL"
ls -la "$ROOT" | head -20

if [ ! -x "$VENV/bin/python" ]; then
  echo "[info] venv missing -> running cloud/bootstrap.sh first (one-time, ~10 min)"
  bash "$ROOT/cloud/bootstrap.sh" "${BOOTSTRAP_BACKEND:---cu126}"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo "[$(date +%H:%M:%S)] === training ==="
nvidia-smi --query-gpu=memory.total,driver_version --format=csv,noheader || true

set +e
python -u "$ROOT/supermimic/src/supermimic/scripts/train.py" SuperMimic-GenCoord-G1 \
  --num-envs "$NUM_ENVS" \
  --motion-file "$MOTION" \
  --runner.max-iterations "$ITERS" \
  --runner.save-interval "$SAVE_INTERVAL" \
  --runner.headless \
  "${EXTRA_ARGS[@]}"
STATUS=$?
set -e

echo "[$(date +%H:%M:%S)] training exited with $STATUS"
echo "=== checkpoints ==="
find "$ROOT" -name '*.pt' -newermt '-24 hours' 2>/dev/null | head -20 || true
exit "$STATUS"
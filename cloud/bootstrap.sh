#!/usr/bin/env bash
# Cloud bootstrap for SuperMimic on a fresh Gradmotion/flux pod.
#
# Why this script exists: the official images top out at Python 3.11 while mjlab needs 3.13,
# and the CUDA stack (mujoco-warp / warp-lang / torch) must match *the pod's driver*.
#
# The venv lives OUTSIDE the workspace (``$SUPERMIMIC_VENV``, default ``~/.cache/supermimic-venv``):
# the platform SDK watches the workspace and uploads every new file, so a 30k-file venv
# inside it floods the upload queue and the console log.
#
# Usage (from the repo root, after `git clone`):
#   bash cloud/bootstrap.sh [--cu126|--cu128|--cpu]

set -euo pipefail

BACKEND="${1:---cu126}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${SUPERMIMIC_VENV:-$HOME/.cache/supermimic-venv}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export DEBIAN_FRONTEND=noninteractive
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
# uv: be patient and retry, the pod's link to PyPI/file hosts is flaky.
export UV_HTTP_TIMEOUT=120
export UV_CONCURRENT_DOWNLOADS=4
# The pod exports PYTHONHOME/PYTHONPATH for Isaac Sim; they break any other interpreter.
unset PYTHONHOME PYTHONPATH PYTHONSTARTUP PYTHONWARNINGS || true

ts() { date '+%H:%M:%S'; }
say() { echo "[$(ts)] $*"; }

# retry <attempts> <cmd...>: uv installs hit transient PyPI/metadata failures.
retry() {
  local n="$1"; shift
  local i=1
  until "$@"; do
    if [ "$i" -ge "$n" ]; then
      say "command failed after $n attempts: $*"
      return 1
    fi
    say "attempt $i failed, retrying in 10 s: $*"
    i=$((i + 1))
    sleep 10
  done
}

say "=== 0. host info ==="
nvidia-smi || { echo "[FATAL] no GPU / no driver"; exit 1; }
say "driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
df -h "$HOME" | tail -1

say "=== 0b. basic tooling ==="
for bin in curl git; do
  if ! command -v "$bin" >/dev/null 2>&1; then
    say "missing $bin -> trying apt-get"
    if command -v apt-get >/dev/null 2>&1 && [ "$(id -u)" = "0" ]; then
      apt-get update -qq && apt-get install -y -qq "$bin"
    else
      echo "[FATAL] $bin missing and cannot install"; exit 1
    fi
  fi
done

say "=== 1. install uv (user space) ==="
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

say "=== 2. python 3.13 venv at $VENV ==="
mkdir -p "$(dirname "$VENV")"
# only-managed: never reuse the pod's 3.11 (Isaac Sim) interpreter as "3.13".
uv venv --python 3.13 --python-preference only-managed "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -c "import sys; print('python', sys.version.split()[0], sys.executable)"

say "=== 3. torch (>=2.14 required by mjlab) ==="
install_torch() {
  local idx="$1"
  say "trying torch from $idx"
  retry 3 uv pip install --python "$VENV" torch --index-url "$idx" || return 1
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
TORCH_OK=0
for IDX in "${CANDIDATES[@]}"; do
  if install_torch "$IDX"; then TORCH_OK=1; break; fi
  say "torch install from $IDX failed, trying next"
done
[ "$TORCH_OK" = "1" ] || { echo "[FATAL] no torch wheel worked"; exit 3; }

say "=== 4. mjlab + supermimic (editable; nvidia index for mujoco-warp) ==="
retry 4 uv pip install --python "$VENV" \
  --extra-index-url https://pypi.nvidia.com \
  -e "$ROOT/mjlab_research" \
  -e "$ROOT/supermimic"

say "=== 4a. pillow (replay video text overlay) ==="
# Replay draws "target vs actual velocity" on the frames; there is no other text renderer in
# the venv, so pillow is a hard replay dependency. Failure here is not fatal for training.
retry 3 uv pip install --python "$VENV" pillow || say "pillow install failed (overlay will be skipped)"
# onnxruntime: the replay can drive the exported deploy ONNX instead of the torch actor
retry 3 uv pip install --python "$VENV" onnxruntime || say "onnxruntime install failed (deploy-onnx replay unavailable)"

say "=== 4b. pin a warp-lang build the driver can actually use ==="
# warp 1.19/1.18 wheels are built with CUDA 13.4 and refuse to run on a 12.x driver
# ("Insufficient CUDA driver version ... 12.2"), which leaves zero CUDA devices and then fails
# with "Invalid device identifier: cuda:0" inside mujoco-warp. 1.17/1.16/1.15 are built with
# CUDA 12.9 and do see the 4090 on this pod (verified in the log). Probe with a sentinel line,
# because warp prints its init banner to stdout and would otherwise corrupt the count.
WARP_OK=0
for V in ${WARP_VERSIONS:-1.17.0 1.16.0 1.15.0}; do
  say "trying warp-lang==$V"
  if retry 2 uv pip install --python "$VENV" "warp-lang==$V" >/dev/null 2>&1; then
    OUT=$("$VENV/bin/python" - <<'PY' 2>&1 || true
import warnings
warnings.filterwarnings("ignore")
import warp as wp
wp.init()
n = len([d for d in wp.get_devices() if d.alias.startswith("cuda")])
print(f"CUDA_DEVICES={n}")
PY
)
    say "warp $V probe: $(echo "$OUT" | tr '\n' ' ')"
    if echo "$OUT" | grep -qE 'CUDA_DEVICES=[1-9]'; then
      WARP_OK=1
      say "warp $V works"
      break
    fi
  else
    say "warp $V not installable"
  fi
done
if [ "$WARP_OK" != "1" ]; then
  echo "[FATAL] no warp-lang build sees the GPU on this driver"
  nvidia-smi --query-gpu=driver_version --format=csv,noheader || true
  exit 4
fi

say "=== 5. verify the CUDA stack ==="
"$VENV/bin/python" -u - <<'PY'
import torch, warp, mujoco
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available(),
      "cuda", torch.version.cuda, "gpu", torch.cuda.get_device_name(0), flush=True)
print("warp", warp.__version__, "devices", [d.alias for d in warp.get_devices()], flush=True)
print("mujoco", mujoco.__version__, flush=True)
import mjlab.tasks  # noqa: F401
import supermimic.tasks  # noqa: F401
print("mjlab + supermimic import OK", flush=True)
PY

say "=== 6. pack the motion dataset (npz -> packed, ~20 s) ==="
if [ -d "$ROOT/datasets/walk_only/npz" ] && [ ! -d "$ROOT/datasets/walk_only/packed" ]; then
  "$VENV/bin/python" -u "$ROOT/mjlab_research/src/mjlab/scripts/pack_motions.py" \
    --input "$ROOT/datasets/walk_only/npz/*.npz" \
    --output "$ROOT/datasets/walk_only/packed"
fi

say "=== 7. smoke: 20 iterations, 64 envs ==="
cd "$ROOT"
# tyro bool flags need an explicit value: `--runner.headless True`, not `--runner.headless`.
"$VENV/bin/python" -u supermimic/src/supermimic/scripts/train.py SuperMimic-GenCoord-G1 \
  --num-envs 64 \
  --motion-file "$ROOT/datasets/walk_only/packed" \
  --runner.max-iterations 20 \
  --runner.headless True \
  || { echo "[FATAL] smoke training failed"; exit 1; }

say "=== bootstrap OK ==="
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
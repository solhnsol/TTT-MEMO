#!/bin/bash
# RunPod one-time setup (NOT a queue job, no GPU work except listing the device). Creates the Python 3.12 venv (e2e/pyproject.toml needs python>=3.12 and jax[cuda12]<0.6; a py3.11 pod image is fine because uv installs 3.12), installs the pinned package list, and writes the environment file that jobs 170-174 source.
# Usage (from the repo root on the pod, repo already cloned and the commit fetched):  bash scripts/exp07_write_ladder/jobs_proposed/rp_setup.sh
# Parameters (env): REQ = pinned package list (default: $RP_REQ_DEFAULT below = /workspace/rp_requirements.txt if it exists, else env/venv.freeze.txt of the repo; the lab server list is /tmp/claude-1147/rp_requirements.txt, 107 packages: python 3.12.3, jax/jaxlib 0.5.3, equinox 0.12.1, optax 0.2.4, orbax-checkpoint 0.11.13, numpy 2.2.5), VENV (default $PWD/.venv), RP_ENV (default /workspace/rp_env.sh), OUT (default /workspace/exp07_out: results go here; EXP07_OUT).
# Needs network (PyPI) once. The venv, the HF/JAX caches and OUT should be under /workspace (the network volume), not on the container disk. NO sudo. Not run anywhere yet (written without a pod): the first run is the first test; a failure here costs setup time only.
set -e
cd "$(dirname "$0")/../../.." 2>/dev/null || true
ROOT=$(pwd)
VENV="${VENV:-$ROOT/.venv}"
RP_ENV="${RP_ENV:-/workspace/rp_env.sh}"
OUT="${OUT:-/workspace/exp07_out}"
if [ -z "$REQ" ]; then if [ -f /workspace/rp_requirements.txt ]; then REQ=/workspace/rp_requirements.txt; else REQ="$ROOT/env/venv.freeze.txt"; fi; fi
[ -f "$REQ" ] || { echo "package list $REQ missing (exit 3)"; exit 3; }
command -v uv >/dev/null 2>&1 || python3 -m pip install --user uv
export PATH="$HOME/.local/bin:$PATH"
[ -d "$VENV" ] || uv venv "$VENV" --python 3.12
uv pip install --python "$VENV/bin/python" -r "$REQ"
"$VENV/bin/python" - <<'PY'
import sys;assert sys.version_info[:2]==(3,12),sys.version
import jax,numpy;print('python',sys.version.split()[0],'jax',jax.__version__,'numpy',numpy.__version__,'devices',jax.devices())
PY
NVRTC=$(ls -d "$VENV"/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib 2>/dev/null || true)
[ -n "$NVRTC" ] || echo "WARNING: nvidia/cuda_nvrtc/lib not in the venv (the list should contain nvidia-cuda-nvrtc-cu12==12.8.93); cuDNN attention needs it"
mkdir -p "$OUT/states" "$(dirname "$RP_ENV")"
cat > "$RP_ENV" <<ENVEOF
# written by rp_setup.sh; sourced by jobs 170-174 (RP_ENV)
export PY="$VENV/bin/python"
export OUT="$OUT"
export EXP07_OUT="$OUT"
export XLA_FLAGS="--xla_gpu_deterministic_ops=true"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export LD_LIBRARY_PATH="$NVRTC\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
export HF_HOME="\${HF_HOME:-/workspace/hf}"
ENVEOF
echo "wrote $RP_ENV:"; cat "$RP_ENV"
for P in checkpoints/3b_ttt_e2e_finetune_books_128k_3x_cc/1949 data/tokenizer/llama3/tokenizer.json data/dclm_val_subset/val_c0_200x8193.u32le.bin; do [ -e "$P" ] && echo "ok   $P" || echo "MISSING $P (copy it from the lab bundle)"; done

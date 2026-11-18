#!/bin/bash
# JOB 1319 = FictionalQA W0 pre-screen on Qwen3-4B-Base (GPU forward only; the Llama-3 screen of 1312 does not transfer, Qwen3-token worlds / golds / reader): scripts/exp09_qwen_ttt/fq_qwen_screen.py over the 8 Qwen3 worlds (labels 211..218, experiments/exp07_write_ladder/fq/worlds_qwen3 = branch commit 733799e worlds_qwen3), 24 facts each; ~150-300 s per world (tiny-model CPU selftest 294 s incl. model build; the 4B GPU read is expected faster): est. 20-30 min. Output experiments/exp07_write_ladder/fq/w0screen_qwen/. NOT queued until Hans' sentence.
set -eo pipefail
export PYTHONUNBUFFERED=1
unset LD_LIBRARY_PATH XLA_PYTHON_CLIENT_ALLOCATOR XLA_FLAGS EXP07_OUT
cd "$(dirname "$0")/../../.." 2>/dev/null || true
PYQ=.venv-exp02a/bin/python; E=scripts/exp09_qwen_ttt; FQ=experiments/exp07_write_ladder/fq
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 17"; exit 17; }
sha256sum $E/fq_qwen_screen.py scripts/exp07_write_ladder/fq_screen_run.py scripts/exp07_write_ladder/fq_world.py $FQ/worlds_qwen3/*.json
$PYQ $E/fq_qwen_screen.py --fq-worlds $FQ/worlds_qwen3 --out $FQ/w0screen_qwen --fast fullmlp --lr 1.0

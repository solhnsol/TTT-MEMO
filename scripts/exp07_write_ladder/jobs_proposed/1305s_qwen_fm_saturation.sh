#!/bin/bash
# JOB 1305s = QWEN3-4B fullmlp SATURATION CHECK (technical; EXPLORATORY; GPU one job at a time; approved by Hans 11/17 02:50Z "포화 여부 실험 먼저 진행하자 승인할게"): the 1304s sweep (seed 161, NATS) saw the out-of-window gain saturate near +1.2..+1.3 nat and the fullmlp update norm equal the lr (clip always binds); fullmlp lr above 1e-2 was not tested. Two runs, same arguments as 1304s: fullmlp lr 3e-2 and 1e-1 (world seed 161, arm NATS, reads at exposures 64,128, clip 1.0). Each prints a SWEEP line; the table lr_sweep/summary.json is rewritten over all configs in the directory. ESTIMATE ~4.5 min per run, ~9 min total.
set -eo pipefail
export PYTHONUNBUFFERED=1
unset LD_LIBRARY_PATH XLA_PYTHON_CLIENT_ALLOCATOR XLA_FLAGS EXP07_OUT   # the torch env of exp09 (the job runner sets the JAX environment)
cd "$(dirname "$0")/../../.." 2>/dev/null || true
PYQ=.venv-exp02a/bin/python; E=scripts/exp09_qwen_ttt; O=experiments/exp09_qwen_ttt/lr_sweep
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
for t in test_bandmask.py test_bandmask_vs_jax.py; do CUDA_VISIBLE_DEVICES= $PYQ $E/$t | tail -1; done
CUDA_VISIBLE_DEVICES= $PYQ $E/cband_sweep_summary.py --selftest | tail -1
mkdir -p $O
run() { # cfg fast lr
  [ -f $O/$1/cband_NATS_s161.json ] && [ -f $O/$1/cband_W0_s161.json ] || $PYQ $E/cell_band_ladder.py --stage run --world-seed 161 --arms NATS --reads 64,128 --fast $2 --lr $3 --clip 1.0 --out $O/$1
  $PYQ $E/cband_sweep_summary.py --dir $O --cfgs $1
}
run fm_lr3e-2 fullmlp 3e-2
run fm_lr1e-1 fullmlp 1e-1
$PYQ $E/cband_sweep_summary.py --dir $O --json $O/summary.json
echo "outputs: $O/summary.json"

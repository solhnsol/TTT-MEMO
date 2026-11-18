#!/bin/bash
# JOB 1318 = E59 Qwen seed addition: world seeds 163 / 164, 6% (T = 0.30426) only (EXPLORATORY; same code as 1315: e59_qwen_run.py; pre-set rule 'add seeds when the sign of a contrast differs across the first two seeds': in 1315 NAT-NATF, FACT-ISO and I differ between seeds 161 and 162 (6%); NOT queued until Hans' sentence; est. ~1 h). W0 reads reused from experiments/exp09_qwen_ttt/cband_main (seeds 163 / 164 exist, stream sha asserted). Output experiments/exp09_qwen_ttt/e59q/p6 (same directory as seeds 161 / 162).
set -eo pipefail
export PYTHONUNBUFFERED=1
unset LD_LIBRARY_PATH XLA_PYTHON_CLIENT_ALLOCATOR XLA_FLAGS EXP07_OUT
cd "$(dirname "$0")/../../.." 2>/dev/null || true
PYQ=.venv-exp02a/bin/python; E=scripts/exp09_qwen_ttt; O=experiments/exp09_qwen_ttt/e59q/p6; REF=experiments/exp09_qwen_ttt/cband_main
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 16"; exit 16; }
mkdir -p $O
sha256sum $E/e59_qwen_run.py scripts/exp07_write_ladder/e59_srv_core.py $E/cell_band_ladder.py $E/inner.py $E/qwen_runtime.py $E/world_text.py
CUDA_VISIBLE_DEVICES= $PYQ $E/e59_qwen_run.py --selftest | tail -1
for s in 163 164; do
  set +e
  $PYQ $E/e59_qwen_run.py --world-seed $s --step-norm 0.30426 --arms NAT,NATF,CHK,CHKF,FACT,ISO --fast fullmlp --lr 1.0 --clip 1.0 --reads 16,32,64,128 --ref-dir $REF --out $O
  echo "RUN e59q seed $s rc=$?"
  set -e
done

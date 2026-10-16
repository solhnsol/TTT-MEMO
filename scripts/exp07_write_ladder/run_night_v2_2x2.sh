#!/bin/bash
# overnight A100 chain (Hans 00:36 KST: keep the GPU busy): after the selector diagnostic -> label-free v2 screen (s101-103) ->
# 2x2 main cells (registered spec v2 + amendment) at the selected kappa.
cd /home/solhnsol/workspace/TTT-MEMO
OUT=experiments/exp07_write_ladder;S=scripts/exp07_write_ladder;PY=.venv/bin/python;lg(){ echo "$(date -u +%FT%TZ) $*" >> $OUT/queue.log; }
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib" XLA_PYTHON_CLIENT_ALLOCATOR=platform XLA_FLAGS="--xla_gpu_deterministic_ops=true" EXP07_OUT="$PWD/$OUT"
until grep -qE "Z SELDIAG_DONE$" $OUT/queue.log; do sleep 60; done
if [ -f $OUT/v2_selector_diagnostic.json ]; then
  for sd in 101 102 103; do lg START lfv2 s$sd; SEED=$sd ONESTEP2=1 STAGE=lfv2 $PY $S/grad_components.py > $OUT/onestep2_lfv2_s$sd.log 2>&1 && lg DONE lfv2 s$sd || lg FAILED lfv2 s$sd; done
  $PY $S/analyze_onestep2.py lfv2 > $OUT/onestep2_lfv2.log 2>&1; lg "ANALYZED lfv2 $(grep -o 'V2: [A-Z(). ,a-z0-9>-]*' $OUT/onestep2_lfv2.md | head -1)"
else lg "SKIP lfv2: no selector diagnostic"; fi
K=$(grep -o "S2X2 KAPPA SELECTED [0-9.]*" $OUT/queue.log | tail -1 | awk '{print $4}')
if [ -z "$K" ]; then lg "SKIP 2x2 main: no kappa selected"; lg NIGHT_DONE; exit 0; fi
c(){ cell=$1;k=$2;sd=$3;for w in A B; do lg START s2x2 $cell k$k s$sd $w
  $PY $S/stream2x2.py --cell $cell --kappa $k --world-seed $sd --worlds $w --mode save > $OUT/n_s2x2_${cell}_k${k}_s${sd}_$w.log 2>&1 && \
  $PY $S/stream2x2.py --cell $cell --kappa $k --world-seed $sd --worlds $w --mode eval > $OUT/n_s2x2e_${cell}_k${k}_s${sd}_$w.log 2>&1 && lg DONE s2x2 $cell k$k s$sd $w || lg FAILED s2x2 $cell k$k s$sd $w; done; }
if grep -q "S2X2 BITID FAIL" $OUT/queue.log; then for sd in 101 102 103; do c nat 0 $sd; c msk 0 $sd; done; fi   # registered: re-run the reused cells
for sd in 102 103; do c natc2 $K $sd; c mskc2 $K $sd; done
for sd in 101 102 103; do c mskgp $K $sd; done
c mskrep $K 101
lg NIGHT_DONE

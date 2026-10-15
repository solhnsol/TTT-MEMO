#!/bin/bash
# Reordered per reviewer priority GO (2026-10-15 13:35 KST): after the running grad_components -> seed 102/103
# replication (L1 ck, L0mp alt, L0fs) -> K sweep K=1,4. Rules: PROTOCOL_stage01.md.
cd /home/solhnsol/workspace/TTT-MEMO
OUT=experiments/exp07_write_ladder
lg(){ echo "$(date -u +%FT%TZ) $*" >> $OUT/queue.log; }
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib" XLA_PYTHON_CLIENT_ALLOCATOR=platform XLA_FLAGS="--xla_gpu_deterministic_ops=true" EXP07_OUT="$PWD/$OUT"
PY=.venv/bin/python; S=scripts/exp07_write_ladder
run1(){ lg START "$@"; $PY $S/ladder.py "$@" >> $OUT/q3_$(echo "$*" | tr ' -' '__').log 2>&1 && lg DONE "$@" || lg FAILED "$@"; }
two(){ for mode in save eval; do lg START $mode "$@"
  if $PY $S/ladder.py --mode $mode "$@" >> $OUT/q3_$(echo "$mode $*" | tr ' -' '__').log 2>&1; then lg DONE $mode "$@"; else lg FAILED $mode "$@"; return; fi; done; }
lg START grad_checks; if GRAD_CHECK=1 $PY $S/grad_components.py > $OUT/st2_grad_checks.log 2>&1; then lg "PASS grad_checks sha256 $(sha256sum $S/grad_components.py | cut -c1-64)"
  lg START grad_components; $PY $S/grad_components.py > $OUT/st2_grad_components.log 2>&1 && lg DONE grad_components || lg FAILED grad_components
else lg "FAIL grad_checks (screening not run)"; fi
for ws in 102 103; do
  run1 --level L1 --world-seed $ws --epochs 32 --tag ck
  two --level L0mp --world-seed $ws --passes 8 --tag alt
  two --level L0fs --world-seed $ws --passes 8
  $PY $S/analyze_stage01.py $ws > $OUT/analyze_stage01_s$ws.log 2>&1; lg ANALYZED repl $ws
done
lg REPL_DONE
for K in 1 4; do two --level L0k --world-seed 101 --k $K --tag k$K; done
$PY $S/analyze_ksweep.py 101 > $OUT/analyze_ksweep_s101.log 2>&1; lg STAGE23_DONE
# 125M native L0 s101 (small; run after the 3B queue because it OOMs next to the 3B fp32 gradient job)
O8=experiments/exp08_125m; rm -f $O8/L0_main_s101_*.saved.json; rm -rf $O8/states/L0_main_s101
EXP07_OUT=$PWD/$O8 $PY scripts/exp08_125m/run125.py ladder --mode save --level L0 --world-seed 101 > $O8/L0_save.log 2>&1 && EXP07_OUT=$PWD/$O8 $PY scripts/exp08_125m/run125.py ladder --mode eval --level L0 --world-seed 101 > $O8/L0_eval.log 2>&1 && lg DONE 125M L0 || lg FAILED 125M L0

#!/bin/bash
# Overnight branch after the shadowing controls (PROTOCOL_stage01.md, 22:10 KST): L0mw pilot if L1-irr is not a strong
# reduction; then L0or replication on s102/s103.
cd /home/solhnsol/workspace/TTT-MEMO
OUT=experiments/exp07_write_ladder;S=scripts/exp07_write_ladder;PY=.venv/bin/python;lg(){ echo "$(date -u +%FT%TZ) $*" >> $OUT/queue.log; }
until grep -q SHADOW_DONE $OUT/queue.log; do sleep 60; done
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib" XLA_PYTHON_CLIENT_ALLOCATOR=platform XLA_FLAGS="--xla_gpu_deterministic_ops=true" EXP07_OUT="$PWD/$OUT"
two(){ for mode in save eval; do lg START $mode "$@"
  if $PY $S/ladder.py --mode $mode "$@" >> $OUT/nb_$(echo "$mode $*" | tr ' -' '__').log 2>&1; then lg DONE $mode "$@"; else lg FAILED $mode "$@"; return; fi; done; }
R=$($PY -c "
import re;t=open('$OUT/l1variants_report_s101.md').read();m=re.search(r'\| L1-irr \|.*?\| ([0-9.]+) \[',t);print(m.group(1) if m else 'NA')")
lg "BRANCH L1-irr s101 R=$R"
if [ "$R" != "NA" ] && $PY -c "import sys;sys.exit(0 if float('$R')>0.30 else 1)"; then
  lg "BRANCH -> run L0mw pilot"; two --level L0mw --world-seed 101 --passes 8
  $PY $S/analyze_stage01.py 101 > $OUT/analyze_stage01_s101_mw.log 2>&1; lg ANALYZED mw
else lg "BRANCH -> L0mw skipped (L1-irr strong reduction or missing)"; fi
for ws in 102 103; do two --level L0or --world-seed $ws --passes 8; $PY $S/analyze_stage01.py $ws > $OUT/analyze_stage01_s${ws}_or.log 2>&1; lg ANALYZED or $ws; done
if grep -q "BRANCH -> run L0mw pilot" $OUT/queue.log; then
  for ws in 102 103; do two --level L0mw --world-seed $ws --passes 8; $PY $S/analyze_stage01.py $ws > $OUT/analyze_stage01_s${ws}_mw.log 2>&1; lg ANALYZED mw $ws; done
fi
lg NIGHT_DONE

#!/bin/bash
# Night queue v2 (PROTOCOL_stage01.md, 23:10 KST): replaces run_em / run_shadow / run_night_branch.
cd /home/solhnsol/workspace/TTT-MEMO
OUT=experiments/exp07_write_ladder;S=scripts/exp07_write_ladder;PY=.venv/bin/python;O8=experiments/exp08_125m;lg(){ echo "$(date -u +%FT%TZ) $*" >> $OUT/queue.log; }
until grep -q NSWEEP_QUEUE_DONE $OUT/queue.log; do sleep 60; done
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib" XLA_PYTHON_CLIENT_ALLOCATOR=platform XLA_FLAGS="--xla_gpu_deterministic_ops=true" EXP07_OUT="$PWD/$OUT"
run1(){ lg START "$@"; $PY $S/ladder.py "$@" >> $OUT/n2_$(echo "$*" | tr ' -' '__').log 2>&1 && lg DONE "$@" || lg FAILED "$@"; }
two(){ for mode in save eval; do lg START $mode "$@"
  if $PY $S/ladder.py --mode $mode "$@" >> $OUT/n2_$(echo "$mode $*" | tr ' -' '__').log 2>&1; then lg DONE $mode "$@"; else lg FAILED $mode "$@"; return 1; fi; done; }
two125(){ for mode in save eval; do lg START 125M $mode "$@"
  if EXP07_OUT=$PWD/$O8 $PY scripts/exp08_125m/run125.py ladder --mode $mode "$@" >> $O8/n2_$(echo "$mode $*" | tr ' -' '__').log 2>&1; then lg DONE 125M $mode "$@"; else lg FAILED 125M $mode "$@"; return 1; fi; done; }
# 0. smoke of the new masked-write path on 125M (1 pass, world A) before any 3B pilot
lg START smoke L0mw 125M; EXP07_OUT=$PWD/$O8 $PY scripts/exp08_125m/run125.py ladder --level L0mw --world-seed 101 --passes 1 --block 256 --scale 1 --worlds A --tag smoke > $O8/smoke_mw.log 2>&1 && lg DONE smoke L0mw || lg FAILED smoke L0mw
run1 --level L1 --world-seed 101 --epochs 8 --tag irr --l1-variant irr --save-epochs 8
$PY $S/analyze_l1variants.py 101 > $OUT/analyze_l1variants_s101.log 2>&1
lg START ctx3; CTX3=1 $PY $S/grad_components.py > $OUT/ctx3.log 2>&1 && lg DONE ctx3 || lg FAILED ctx3
R=$($PY -c "
import re;t=open('$OUT/l1variants_report_s101.md').read();m=re.search(r'\| L1-irr \|.*?\| ([0-9.]+) \[',t);print(m.group(1) if m else 'NA')")
lg "BRANCH L1-irr s101 R=$R"; PILOT=0
if [ "$R" != "NA" ] && $PY -c "import sys;sys.exit(0 if float('$R')>0.30 else 1)" && grep -q "DONE smoke L0mw" $OUT/queue.log; then
  PILOT=1; lg "BRANCH -> 3B L0mw pilot"
  for sc in 1 2; do two --level L0mw --world-seed 101 --passes 8 --block 1024 --scale $sc --tag x$sc; done
  $PY $S/analyze_pilot.py 101 > $OUT/analyze_pilot_s101.log 2>&1; lg ANALYZED pilot 101
else lg "BRANCH -> pilot skipped"; fi
for ws in 102 103; do run1 --level L1 --world-seed $ws --epochs 8 --tag ctx --l1-variant ctx; run1 --level L1 --world-seed $ws --epochs 8 --tag irr --l1-variant irr; $PY $S/analyze_l1variants.py $ws > $OUT/analyze_l1variants_s$ws.log 2>&1; done
lg SHADOW_SEEDS_DONE
if [ $PILOT = 1 ] && grep -q "POSITIVE" $OUT/pilot_report_s101.md 2>/dev/null; then
  SEL=$(grep -o "selected lr x[0-9]*" $OUT/pilot_report_s101.md | head -1 | grep -o "[0-9]*$")
  two --level L0mw --world-seed 102 --passes 8 --block 1024 --scale ${SEL:-1} --tag x${SEL:-1}; $PY $S/analyze_pilot.py 102 > $OUT/analyze_pilot_s102.log 2>&1
fi
for b in 1024 256; do for sc in 1 2 4; do two125 --level L0mw --world-seed 101 --passes 8 --block $b --scale $sc --tag b${b}x$sc; done; done
EXP07_OUT=$PWD/$O8 $PY $S/analyze_pilot.py 101 125m > $O8/analyze_pilot125.log 2>&1; lg ANALYZED 125M grid
lg START em L1ck; $PY $S/em_eval.py L1ck > $OUT/em_L1ck.log 2>&1 && lg DONE em L1ck || lg FAILED em L1ck
lg NIGHT2_DONE

#!/bin/bash
# Takes over run_day3b.sh (Hans chose "넣기" 10:34 KST, strategy thread): after Q3 s101 lr selection, the W0 one-step
# replication on s102/s103 (gate) and the suppression-signal screen (s101-103) run BEFORE Q3 s102/s103.
# $1 = pid of the running python child of run_day3b.sh (longctx_nll 101 L0mw x2k), not restarted.
cd /home/solhnsol/workspace/TTT-MEMO
OUT=experiments/exp07_write_ladder;S=scripts/exp07_write_ladder;PY=.venv/bin/python;lg(){ echo "$(date -u +%FT%TZ) $*" >> $OUT/queue.log; }
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib" XLA_PYTHON_CLIENT_ALLOCATOR=platform XLA_FLAGS="--xla_gpu_deterministic_ops=true" EXP07_OUT="$PWD/$OUT"
one(){ mode=$1;shift;lg START $mode "$@"
  if $PY $S/ladder.py --mode $mode "$@" >> $OUT/n3_$(echo "$mode $*" | tr ' -' '__').log 2>&1; then lg DONE $mode "$@"; else lg FAILED $mode "$@"; return 1; fi; }
job(){ name=$1;shift;lg START $name; if "$@" > $OUT/n3_$name.log 2>&1; then lg DONE $name; else lg FAILED $name; return 1; fi; }
lnll(){ job longnll_$2_$3_s$1 $PY $S/longctx_nll.py $1 $2 $3; }
em(){ EM_POINTS=1,2 job em_$2_$3_s$1 $PY $S/em_eval.py saved $2 $3 $1; }
drop(){ rm -rf $OUT/states/$2_$3_s$1; lg "DELETED states $2_$3_s$1"; }
mwfs(){ sd=$1;t=$2;sc=$3;P="--level L0mwfs --world-seed $sd --passes 8 --block 1024 --scale $sc --tag $t"
  one save $P || return 1
  if [ "$sd" = 101 ] && [ "$t" != x05 ]; then
    if $PY $S/check_hashes.py 101 L0mw $t L0mwfs $t 1 > $OUT/f3_$t.log 2>&1; then lg "F3 PASS $t"; else lg "F3 FAIL $t"; return 1; fi; fi
  one eval $P --keep && lnll $sd L0mwfs $t; drop $sd L0mwfs $t; $PY $S/analyze_mwfs.py $sd $t > $OUT/analyze_mwfs_${t}_s$sd.log 2>&1; lg ANALYZED mwfs $t s$sd; }
lg "QUEUE 3d takes over run_day3b (waiting for pid $1)"
while kill -0 $1 2>/dev/null; do sleep 30; done
[ -f $OUT/longnll_L0mw_x2k_s101.json ] && lg "DONE longnll_L0mw_x2k_s101 (observed by 3d)" || lg "FAILED? longnll_L0mw_x2k_s101 (observed by 3d)"
em 101 L0mw x2k; drop 101 L0mw x2k
mwfs 101 x2 2; mwfs 101 x1 1
$PY $S/analyze_mwfs.py select > $OUT/mwfs_select.log 2>&1
if grep -q "SELECTED NONE" $OUT/mwfs_select.md; then mwfs 101 x05 0.5; $PY $S/analyze_mwfs.py select > $OUT/mwfs_select.log 2>&1; fi
SEL=$(grep -o "SELECTED [a-z0-9]*" $OUT/mwfs_select.md | awk '{print $2}');lg "MWFS SELECTED $SEL"
# ---- inserted: one-step replication (gate) and suppression screen; code must be marked ready after its checks ----
until [ -f $OUT/ONESTEP2_READY ]; do sleep 60; done
for sd in 102 103 101; do lg START onestep2 s$sd; SEED=$sd ONESTEP2=1 STAGE=rep $PY $S/grad_components.py > $OUT/onestep2_rep_s$sd.log 2>&1 && lg DONE onestep2 rep s$sd || lg FAILED onestep2 rep s$sd; done
$PY $S/analyze_onestep2.py gate > $OUT/onestep2_gate.log 2>&1; lg "ONESTEP2 GATE $(grep -o 'GATE [A-Z]*' $OUT/onestep2_gate.md | head -1)"
if grep -q "GATE PASS" $OUT/onestep2_gate.md; then
  for sd in 101 102 103; do lg START screen s$sd; SEED=$sd ONESTEP2=1 STAGE=screen $PY $S/grad_components.py > $OUT/onestep2_screen_s$sd.log 2>&1 && lg DONE screen s$sd || lg FAILED screen s$sd; done
  $PY $S/analyze_onestep2.py screen > $OUT/onestep2_screen.log 2>&1; lg ANALYZED screen
fi
if [ "$SEL" != NONE ]; then SC=$(echo $SEL | sed 's/x05/0.5/;s/x//'); for sd in 102 103; do mwfs $sd $SEL $SC; done; fi
X2K2="--level L0mw --world-seed 102 --passes 8 --block 1024 --scale 2 --tag x2k"
one save $X2K2 && { $PY $S/check_hashes.py 102 L0mw x2 L0mw x2k > $OUT/f1_s102.log 2>&1 && lg "F1 PASS s102" || lg "F1 FAIL s102"; lnll 102 L0mw x2k; em 102 L0mw x2k; drop 102 L0mw x2k; }
lg DAY3B_DONE

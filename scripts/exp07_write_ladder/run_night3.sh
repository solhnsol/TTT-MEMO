#!/bin/bash
# Night queue v3 (strict reviewer verdict, 2026-10-16 00:35 KST): takes over run_night2.sh after the running pilot x1 save
# (pid given as $1, not restarted); inserts the W0 one-step realization test right after the pilot.
cd /home/solhnsol/workspace/TTT-MEMO
OUT=experiments/exp07_write_ladder;S=scripts/exp07_write_ladder;PY=.venv/bin/python;O8=experiments/exp08_125m;lg(){ echo "$(date -u +%FT%TZ) $*" >> $OUT/queue.log; }
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib" XLA_PYTHON_CLIENT_ALLOCATOR=platform XLA_FLAGS="--xla_gpu_deterministic_ops=true" EXP07_OUT="$PWD/$OUT"
run1(){ lg START "$@"; $PY $S/ladder.py "$@" >> $OUT/n2_$(echo "$*" | tr ' -' '__').log 2>&1 && lg DONE "$@" || lg FAILED "$@"; }
one(){ mode=$1;shift;lg START $mode "$@"
  if $PY $S/ladder.py --mode $mode "$@" >> $OUT/n2_$(echo "$mode $*" | tr ' -' '__').log 2>&1; then lg DONE $mode "$@"; else lg FAILED $mode "$@"; return 1; fi; }
two(){ one save "$@" && one eval "$@"; }
two125(){ for mode in save eval; do lg START 125M $mode "$@"
  if EXP07_OUT=$PWD/$O8 $PY scripts/exp08_125m/run125.py ladder --mode $mode "$@" >> $O8/n2_$(echo "$mode $*" | tr ' -' '__').log 2>&1; then lg DONE 125M $mode "$@"; else lg FAILED 125M $mode "$@"; return 1; fi; done; }
lg "QUEUE v3 takes over (waiting for pilot x1 save pid $1)"
while kill -0 $1 2>/dev/null; do sleep 30; done
X1="--level L0mw --world-seed 101 --passes 8 --block 1024 --scale 1 --tag x1"
if [ -f $OUT/L0mw_x1_s101_A.saved.json ] && [ -f $OUT/L0mw_x1_s101_B.saved.json ]; then lg DONE save $X1 "(observed by v3)"; one eval $X1
else lg "FAILED save $X1 (observed by v3: saved files missing)"; two $X1; fi
two --level L0mw --world-seed 101 --passes 8 --block 1024 --scale 2 --tag x2
$PY $S/analyze_pilot.py 101 > $OUT/analyze_pilot_s101.log 2>&1; lg ANALYZED pilot 101
# W0 one-step realization test (mandatory, first after the pilot): grad checks rerun, then ONESTEP (own checks O1-O4)
lg START gradcheck; GRAD_CHECK=1 $PY $S/grad_components.py > $OUT/gradcheck_v3.log 2>&1 && lg DONE gradcheck || lg FAILED gradcheck
if grep -q "DONE gradcheck" <(tail -1 $OUT/queue.log); then
  lg START onestep; ONESTEP=1 $PY $S/grad_components.py > $OUT/onestep.log 2>&1 && lg DONE onestep || lg FAILED onestep
  $PY $S/analyze_onestep.py > $OUT/analyze_onestep.log 2>&1; lg ANALYZED onestep
fi
for ws in 102 103; do run1 --level L1 --world-seed $ws --epochs 8 --tag ctx --l1-variant ctx; run1 --level L1 --world-seed $ws --epochs 8 --tag irr --l1-variant irr; $PY $S/analyze_l1variants.py $ws > $OUT/analyze_l1variants_s$ws.log 2>&1; done
lg SHADOW_SEEDS_DONE
if grep -q "amended verdict.*POSITIVE" $OUT/pilot_report_s101.md 2>/dev/null; then
  SEL=$(grep -o "selected lr x[0-9]*" $OUT/pilot_report_s101.md | head -1 | grep -o "[0-9]*$")
  two --level L0mw --world-seed 102 --passes 8 --block 1024 --scale ${SEL:-1} --tag x${SEL:-1}; $PY $S/analyze_pilot.py 102 > $OUT/analyze_pilot_s102.log 2>&1
fi
for b in 1024 256; do for sc in 1 2 4; do two125 --level L0mw --world-seed 101 --passes 8 --block $b --scale $sc --tag b${b}x$sc; done; done
EXP07_OUT=$PWD/$O8 $PY $S/analyze_pilot.py 101 125m > $O8/analyze_pilot125.log 2>&1; lg ANALYZED 125M grid
lg START em L1ck; $PY $S/em_eval.py L1ck > $OUT/em_L1ck.log 2>&1 && lg DONE em L1ck || lg FAILED em L1ck
lg NIGHT3_DONE

#!/bin/bash
# kappa search, faster procedure (design decision 03:15 KST; rule unchanged: halve until both suppression cells have pass-1 canary <= 0.10)
cd /home/solhnsol/workspace/TTT-MEMO
OUT=experiments/exp07_write_ladder;S=scripts/exp07_write_ladder;PY=.venv/bin/python;lg(){ echo "$(date -u +%FT%TZ) $*" >> $OUT/queue.log; }
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib" XLA_PYTHON_CLIENT_ALLOCATOR=platform XLA_FLAGS="--xla_gpu_deterministic_ops=true" EXP07_OUT="$PWD/$OUT"
q(){ cell=$1;k=$2;w=$3; $PY $S/stream2x2.py --cell $cell --kappa $k --world-seed 101 --worlds $w --mode save --canary-only > $OUT/n_s2x2cq_${cell}_k${k}_$w.log 2>&1 || lg FAILED s2x2cq $cell k$k $w; }
rd(){ $PY - "$@" <<'PY'
import json,sys;k=sys.argv[1];ws=sys.argv[2];o='experiments/exp07_write_ladder/';worst=0;msg=[]
for cell in ('natc2','mskc2'):
    for w in ws:
        L=json.load(open(o+f'S2X2_{cell}_k{k}_cq_s101_{w}.saved.json'))['log'];m0,m1=L[0]['margins'],L[1]['margins']
        r=max(m1[c]-m0[c] for c in ('_gen_nll','_tail_nll'));worst=max(worst,r)
        msg.append(f"{cell}{w} canary {r:+.4f} clip {L[1]['clip_bind_frac']:.3f} c {L[1]['mean_coef']:.3f}")
print(('yes' if worst<=0.10 else 'no'),round(worst,4),'|',' ; '.join(msg))
PY
}
SEL="";BOTH=0
for k in 1 0.5 0.25 0.125 0.0625 0.03125 0.015625; do
  if [ $BOTH = 0 ]; then
    q natc2 $k A; q mskc2 $k A; r=$(rd $k A); lg "S2X2CQ kappa $k worldA $r"
    case "$r" in yes*) q natc2 $k B; q mskc2 $k B; rb=$(rd $k B); lg "S2X2CQ kappa $k worldB $rb"; case "$rb" in yes*) SEL=$k;; *) BOTH=1;; esac;; esac
  else
    for c in natc2 mskc2; do for w in A B; do q $c $k $w; done; done; r=$(rd $k AB); lg "S2X2CQ kappa $k worldsAB $r"; case "$r" in yes*) SEL=$k;; esac
  fi
  [ -n "$SEL" ] && { lg "S2X2 KAPPA SELECTED $SEL"; break; }
done
[ -z "$SEL" ] && lg "S2X2 KAPPA NONE down to 1/64 (both suppression cells failed through damage; descriptive mskgp needs the design thread's ok)"
lg S2X2CHECK_DONE

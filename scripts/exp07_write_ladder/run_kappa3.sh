#!/bin/bash
# kappa search continuation with the kappa/sqrt2 robustness rule (design 03:55 KST): kappa* = first kappa on the halving grid where natc2 and
# mskc2 pass the world-A pass-1 canary at kappa AND at kappa/sqrt2; world B checked at kappa only; the main cells use kappa.
cd /home/solhnsol/workspace/TTT-MEMO
OUT=experiments/exp07_write_ladder;S=scripts/exp07_write_ladder;PY=.venv/bin/python;lg(){ echo "$(date -u +%FT%TZ) $*" >> $OUT/queue.log; }
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib" XLA_PYTHON_CLIENT_ALLOCATOR=platform XLA_FLAGS="--xla_gpu_deterministic_ops=true" EXP07_OUT="$PWD/$OUT"
while pgrep -f "stream2x2.py --cell" >/dev/null; do sleep 10; done   # let the running kappa=0.25 world-A job finish
q(){ local cell=$1 k=$2 w=$3; [ -f $OUT/S2X2_${cell}_k${k}_cq_s101_$w.saved.json ] && return 0; $PY $S/stream2x2.py --cell $cell --kappa $k --world-seed 101 --worlds $w --mode save --canary-only > $OUT/n_s2x2cq_${cell}_k${k}_$w.log 2>&1 || lg FAILED s2x2cq $cell k$k $w; }
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
s2(){ $PY -c "import sys;print(f'{float(sys.argv[1])/2**0.5:.6g}')" $1; }
SEL=""
for k in 0.25 0.125 0.0625 0.03125 0.015625; do
  q natc2 $k A; q mskc2 $k A; r=$(rd $k A); lg "S2X2CQ kappa $k worldA $r"
  case "$r" in yes*)
    k2=$(s2 $k); q natc2 $k2 A; q mskc2 $k2 A; r2=$(rd $k2 A); lg "S2X2CQ kappa $k2 (kappa/sqrt2) worldA $r2"
    case "$r2" in yes*) q natc2 $k B; q mskc2 $k B; rb=$(rd $k B); lg "S2X2CQ kappa $k worldB $rb"; case "$rb" in yes*) SEL=$k;; esac;; esac;; esac
  [ -n "$SEL" ] && { lg "S2X2 KAPPA SELECTED $SEL"; break; }
done
[ -z "$SEL" ] && lg "S2X2 KAPPA NONE down to 1/64 (both suppression cells failed through damage; descriptive mskgp needs the design thread's ok)"
lg S2X2CHECK_DONE

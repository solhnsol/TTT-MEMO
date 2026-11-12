#!/bin/bash
# Re-orchestration after the 06:30 incidents (kappa bug + my concurrent run that OOM'd the selector diagnostic). Order (design):
# kappa search continuation -> selector diagnostic -> v2 screen -> KEYSTAT R1/R2/R3 -> rho + flag -> 2x2 main -> NIGHT_DONE (gradfactors).
cd /home/solhnsol/workspace/TTT-MEMO
OUT=experiments/exp07_write_ladder;S=scripts/exp07_write_ladder;PY=.venv/bin/python;lg(){ echo "$(date -u +%FT%TZ) $*" >> $OUT/queue.log; }
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib" XLA_PYTHON_CLIENT_ALLOCATOR=platform XLA_FLAGS="--xla_gpu_deterministic_ops=true" EXP07_OUT="$PWD/$OUT"
while pgrep -f "exp07_write_ladder/keystat_C.py" >/dev/null; do sleep 20; done
[ -f /home/solhnsol/workspace/TTT-MEMO-dumps/keystat_C.npz ] && lg "DONE keystat_C (observed)" || { lg START keystat_C retry; $PY $S/keystat_C.py > $OUT/keystat_C_retry.log 2>&1 && lg DONE keystat_C || lg FAILED keystat_C; }
# ---- kappa search continuation (registered rule + kappa/sqrt2 + world B at both points) ----
q(){ local cell=$1 k=$2 w=$3; [ -f $OUT/S2X2_${cell}_k${k}_cq_s101_$w.saved.json ] && return 0; $PY $S/stream2x2.py --cell $cell --kappa $k --world-seed 101 --worlds $w --mode save --canary-only > $OUT/n_s2x2cq_${cell}_k${k}_$w.log 2>&1 || lg FAILED s2x2cq $cell k$k $w; }
rd(){ $PY - "$@" <<'PY'
import json,sys;k=sys.argv[1];ws=sys.argv[2];o='experiments/exp07_write_ladder/';worst=0;msg=[]
for cell in ('natc2','mskc2'):
    for w in ws:
        try:L=json.load(open(o+f'S2X2_{cell}_k{k}_cq_s101_{w}.saved.json'))['log']
        except Exception:print('no missing',cell,w);sys.exit()
        m0,m1=L[0]['margins'],L[1]['margins'];r=max(m1[c]-m0[c] for c in ('_gen_nll','_tail_nll'));worst=max(worst,r)
        msg.append(f"{cell}{w} canary {r:+.5f} clip {L[1]['clip_bind_frac']:.3f} c {L[1]['mean_coef']:.3f}")
print(('yes' if worst<=0.10 else 'no'),round(worst,5),'|',' ; '.join(msg))
PY
}
s2(){ $PY -c "import sys;print(f'{float(sys.argv[1])/2**0.5:.6g}')" $1; }
K=""
for kk in 0.0625 0.03125 0.015625; do
  q natc2 $kk A; q mskc2 $kk A; r=$(rd $kk A); lg "S2X2CQ kappa $kk worldA $r"
  case "$r" in yes*) k2=$(s2 $kk); q natc2 $k2 A; q mskc2 $k2 A; r2=$(rd $k2 A); lg "S2X2CQ kappa $k2 (kappa/sqrt2) worldA $r2"
    case "$r2" in yes*) q natc2 $kk B; q mskc2 $kk B; rb=$(rd $kk B); lg "S2X2CQ kappa $kk worldB $rb"
      case "$rb" in yes*) q natc2 $k2 B; q mskc2 $k2 B; rb2=$(rd $k2 B); lg "S2X2CQ kappa $k2 (kappa/sqrt2) worldB $rb2"; case "$rb2" in yes*) K=$kk;; esac;; esac;; esac;; esac
  [ -n "$K" ] && { lg "S2X2 KAPPA SELECTED (corrected search) $K"; break; }
done
[ -z "$K" ] && lg "S2X2 KAPPA NONE down to 1/64 (corrected search; both suppression cells failed through damage; descriptive mskgp needs the design thread's ok)"
# ---- selector diagnostic and v2 screen ----
lg START seldiag retry; $PY $S/sel_diag.py > $OUT/v2_selector_diagnostic.log 2>&1 && lg "DONE seldiag $(grep -o 'SELECTOR [a-z]*' $OUT/v2_selector_diagnostic.log)" || lg FAILED seldiag
if [ -f $OUT/v2_selector_diagnostic.json ]; then
  for sd in 101 102 103; do lg START lfv2 s$sd; SEED=$sd ONESTEP2=1 STAGE=lfv2 $PY $S/grad_components.py > $OUT/onestep2_lfv2_s$sd.log 2>&1 && lg DONE lfv2 s$sd || lg FAILED lfv2 s$sd; done
  $PY $S/analyze_onestep2.py lfv2 > $OUT/onestep2_lfv2.log 2>&1; lg "ANALYZED lfv2 $(grep -o 'V2: [A-Z(). ,a-z0-9>-]*' $OUT/onestep2_lfv2.md | head -1)"
fi
# ---- KEYSTAT stage 1 ----
if [ -f /home/solhnsol/workspace/TTT-MEMO-dumps/keystat_C.npz ]; then
  for R in "R1 c0,a,a_s,b1,b4,b16,wr0h,wr0" "R2 c_hi,c_lo,wr1,p05,p10" "R3 muon,topk8"; do set -- $R
    lg START keystat s101 $1; SEED=101 ONESTEP2=1 STAGE=keystat KS_CELLS=$2 $PY $S/grad_components.py > $OUT/onestep2_keystat_s101_$1.log 2>&1 && lg DONE keystat s101 $1 || lg FAILED keystat s101 $1
    cp $OUT/onestep2_keystat_checks_s101.json $OUT/onestep2_keystat_checks_s101_$1.json 2>/dev/null; done
  $PY $S/analyze_onestep2.py keystat 101 > $OUT/onestep2_keystat.log 2>&1; lg ANALYZED keystat stage1
fi
# ---- 2x2 ----
if [ -n "$K" ]; then
  lg START rho2x2; $PY $S/rho2x2.py 101 102 103 > $OUT/rho2x2.log 2>&1 && lg DONE rho2x2 || lg FAILED rho2x2
  $PY $S/analyze_rho2x2.py $K > $OUT/rho2x2_flag.log 2>&1 && lg "DONE rho2x2 flag" || lg "FAILED rho2x2 flag"
  c(){ local cell=$1 k=$2 sd=$3 w; for w in A B; do lg START s2x2 $cell k$k s$sd $w
    $PY $S/stream2x2.py --cell $cell --kappa $k --world-seed $sd --worlds $w --mode save > $OUT/n_s2x2_${cell}_k${k}_s${sd}_$w.log 2>&1 && \
    $PY $S/stream2x2.py --cell $cell --kappa $k --world-seed $sd --worlds $w --mode eval > $OUT/n_s2x2e_${cell}_k${k}_s${sd}_$w.log 2>&1 && lg DONE s2x2 $cell k$k s$sd $w || lg FAILED s2x2 $cell k$k s$sd $w; done; }
  for sd in 101 102 103; do c natc2 $K $sd; c mskc2 $K $sd; done
  for sd in 101 102 103; do c mskgp $K $sd; done
  c mskrep $K 101
else lg "SKIP 2x2 main: no kappa selected"; fi
lg NIGHT_DONE

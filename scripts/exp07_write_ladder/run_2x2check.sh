#!/bin/bash
# 2x2 setup checks (PROTOCOL "3B 2x2 stream (spec v2)" + amendment; Hans's queue order 00:23 KST): after the LF screen,
# (1) kappa=0 bit-identity on s101 world A for the native and masked code paths (pass-1 state hash vs the reused runs),
# (2) kappa search on s101 (A and B): kappa = 4, 2, 1, 0.5, 0.25 until both natc2 and mskc2 have pass-1 canary rise <= 0.10.
cd /home/solhnsol/workspace/TTT-MEMO
OUT=experiments/exp07_write_ladder;S=scripts/exp07_write_ladder;PY=.venv/bin/python;lg(){ echo "$(date -u +%FT%TZ) $*" >> $OUT/queue.log; }
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib" XLA_PYTHON_CLIENT_ALLOCATOR=platform XLA_FLAGS="--xla_gpu_deterministic_ops=true" EXP07_OUT="$PWD/$OUT"
until grep -qE "Z LFSCREEN_DONE$" $OUT/queue.log; do sleep 60; done
c(){ cell=$1;k=$2;w=$3;lg START s2x2 $cell k$k $w; $PY $S/stream2x2.py --cell $cell --kappa $k --world-seed 101 --worlds $w --mode save > $OUT/n_s2x2_${cell}_k${k}_$w.log 2>&1 && lg DONE s2x2 save $cell k$k $w || { lg FAILED s2x2 save $cell k$k $w; return 1; }; }
e(){ cell=$1;k=$2;w=$3; $PY $S/stream2x2.py --cell $cell --kappa $k --world-seed 101 --worlds $w --mode eval > $OUT/n_s2x2e_${cell}_k${k}_$w.log 2>&1 && lg DONE s2x2 eval $cell k$k $w || { lg FAILED s2x2 eval $cell k$k $w; return 1; }; }
c nat 0 A; c msk 0 A
$PY - <<'PY' >> $OUT/s2x2_bitid.log 2>&1
import json
o='experiments/exp07_write_ladder/'
r={}
for cell,ref in (('nat','L0mp_main_s101_A.json'),('msk','L0mwfs_x1_s101_A.json')):
    new=json.load(open(o+f'S2X2_{cell}_k0_s101_A.saved.json'))['log'];old=json.load(open(o+ref))['log']
    h_new=[e['margins']['_hash'] for e in new if 'margins' in e];h_old=[e.get('hash') for e in old if 'margins' in e]
    r[cell]={'W0_equal':h_new[0]==h_old[0],'pass1_equal':h_new[1]==h_old[1],'new':h_new[:2],'ref':h_old[:2]}
print(json.dumps(r));print('BITID','PASS' if all(v['pass1_equal'] for v in r.values()) else 'FAIL')
PY
lg "S2X2 $(grep -o 'BITID [A-Z]*' $OUT/s2x2_bitid.log | tail -1)"
for k in 4 2 1 0.5 0.25; do
  for cell in natc2 mskc2; do for w in A B; do c $cell $k $w; e $cell $k $w; done; done
  ok=$($PY - $k <<'PY'
import json,sys;k=sys.argv[1];o='experiments/exp07_write_ladder/';worst=0
for cell in ('natc2','mskc2'):
    for w in 'AB':
        L=[e for e in json.load(open(o+f'S2X2_{cell}_k{k}_s101_{w}.json'))['log'] if 'margins' in e]
        worst=max(worst,max(L[1]['margins'][c]-L[0]['margins'][c] for c in ('_gen_nll','_tail_nll')))
print('yes' if worst<=0.10 else 'no',round(worst,4))
PY
)
  lg "S2X2 kappa $k canary $ok"
  case "$ok" in yes*) lg "S2X2 KAPPA SELECTED $k"; break;; esac
done
lg S2X2CHECK_DONE

#!/bin/bash
# JOB 800 = E57 SYNTHETIC SEED 64 CURVE OF THE BRANCH-TOKEN RANK OVER THE CYCLE (thread 'ALLQ보다 나은 가림 방법 찾기', Hans '사이클 안에서의 w=8 곡선' / '합성 + 뉴스'): the writes of the arms M3W1N05 (= ALLQ256, bit-identical to the original ALLQ256, job 755) and M3W8N05 are repeated with the same code and seed and the state is saved at 5 points of the cycle (--save-apps 16,32,64,96,128 = appearance 16 / 32 / 64 / 96 / 128; canary at every save point); the end-state hash is compared with the original state; per-token reads of all 10 states; curve of the branch-token / first-token rank ratio. No existing e57_*.py edited, no git checkout.
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder; S=$D
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
for f in e57_syn_mrun.py e57_syn_tokread.py e57_curve_report.py e57_qband_run.py e57_closure_masks.py e55_gpu.py ladder.py; do echo "sha256 $(sha256sum $D/$f | cut -c1-12) $f"; done
$PY $S/e57_syn_mrun.py --selftest | tail -1; $PY $S/e57_syn_tokread.py --selftest | tail -1; $PY $S/e57_curve_report.py --selftest | tail -1
SQ=$OUT/e57_synq; TT=${TT:-$OUT/e57_syn_curve_s64}; mkdir -p "$TT"
for f in $SQ/syn_s64_m30_masks.npz $OUT/E55_v3a_NAT_s64.saved.json; do [ -f "$f" ] || { echo "input missing: $f (exit 7)"; exit 7; }; done
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
[ -f "$TT/e57_curve_s64.json" ] && [ -z "$RERUN" ] && { echo "curve json exists (exit 8; RERUN=1 to redo)"; exit 8; }
$PY $S/e57_syn_mrun.py --seed 64  --out $SQ --tag synqc --m30 $SQ/syn_s64_m30_masks.npz --ntg-nat $OUT/E55_v3a_NAT_s64.saved.json --arms M3W1N05,M3W8N05 --save-apps 16,32,64,96,128
$PY - <<'PYE'
import json,os,pathlib
O=pathlib.Path(os.environ['EXP07_OUT']);res={}
for a in ('M3W1N05','M3W8N05'):
    n=json.load(open(O/f'e57_synq/E55_synqc_{a}_s64.saved.json'))['reads'];o=json.load(open(O/f'e57_synq/E55_synq_{a}_s64.saved.json'))['reads'];ch=str(max(int(k) for k in o));res[a]=dict(end_chunk=int(ch),end_state_hash_new=n[ch]['_hash'],end_state_hash_original=o[ch]['_hash'],equal=n[ch]['_hash']==o[ch]['_hash'])
print('end-state hash check',res);json.dump(res,open(''.replace('$O','') or '/dev/null','w')) if False else None
(pathlib.Path(os.environ.get('TT_DIR','.'))/'x').parent
PYE
[ -f "$TT/e57_syn_tokread_s64.json" ] || $PY $S/e57_syn_tokread.py --stage read --seed 64 --out "$TT" --tag synqc
$PY $S/e57_curve_report.py --kind syn --seed 64 --tokread $TT/e57_syn_tokread_s64.json --out $TT/e57_curve_s64.json
echo "outputs: $TT/e57_curve_s64.json $TT/e57_syn_tokread_s64.json $SQ/E55_synqc_{M3W1N05,M3W8N05}_s64.{saved,read}.json"

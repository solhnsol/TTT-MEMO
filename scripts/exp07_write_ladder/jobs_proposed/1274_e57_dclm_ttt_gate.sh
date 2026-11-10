#!/bin/bash
# JOB 1274 = E57 DCLM PRESERVATION GATE (Council-fixed spec 11/10; Hans 11/10 'dclm은 ... 마스킹이 기본 모델보다 DCLM 성능을 낮추는지', 'ttt하면서 dclm nll'): online TTT over DCLM validation docs 0-63 as 4 streams of 128 chunks, arms W0 (frozen), NATS (original NAT), NATMASK (ALLQ256 from chunk 8), NATMASKC; prequential NLL of every chunk before its write, natural full context. A 10-chunk smoke (stream 0; checks finite NLL, NATS nll == ce, mask path active at chunks 8-9) runs first and aborts the job on failure. ESTIMATE ~100 min (+ ~10 min smoke).
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
cd "$(dirname "$0")/../../.." 2>/dev/null || true
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
D=scripts/exp07_write_ladder
$PY $D/e57_dclm_ttt_run.py --selftest | tail -1
SM=$OUT/e57_dclm_ttt_smoke
$PY $D/e57_dclm_ttt_run.py --stage run --out $SM --streams 0 --smoke 10 --arms W0,NATS,NATMASK,NATMASKC
$PY - $SM <<'PYE'
import json,sys,math
o=sys.argv[1];R={a:json.load(open(f'{o}/E57_dclmttt_{a}_st0.json')) for a in ('W0','NATS','NATMASK','NATMASKC')}
for a,r in R.items():
    assert all(math.isfinite(x) for x in r['nll_pre']),('non-finite',a)
n=R['NATS']['nll_pre'];c=R['NATS']['ce'];assert max(abs(x-y) for x,y in zip(n,c))<2e-3,('NATS nll vs ce',max(abs(x-y) for x,y in zip(n,c)))
w=R['W0']['nll_pre'];assert abs(w[0]-n[0])<1e-5,('chunk 0 W0 vs NATS',w[0],n[0])        # before any update both equal
assert R['NATMASK']['nll_pre'][:8]==R['NATS']['nll_pre'][:8] or max(abs(x-y) for x,y in zip(R['NATMASK']['nll_pre'][:8],n[:8]))<1e-5,'mask active before chunk 8'
assert R['NATMASK']['ce'][8]!=R['NATS']['ce'][8],'mask path not active at chunk 8'
print('smoke ok: mean nll chunks 8-9',{a:round(sum(r['nll_pre'][8:10])/2,4) for a,r in R.items()})
PYE
$PY $D/e57_dclm_ttt_run.py --stage run --out $OUT/e57_dclm_ttt --streams 0,1,2,3 --arms W0,NATS,NATMASK,NATMASKC
$PY $D/e57_dclm_ttt_run.py --stage analyze --out $OUT/e57_dclm_ttt --json $OUT/e57_dclm_ttt_gate.json
echo "outputs: $OUT/e57_dclm_ttt_gate.json"

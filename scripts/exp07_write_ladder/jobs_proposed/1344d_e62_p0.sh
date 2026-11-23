#!/bin/bash
# JOB 1344d = E62 P0a (E62 P0 request v1 via the coordinator 11/23; Hans: 'make the E62 P0 code after 1344k'): invented facts (6 attribute types, names + 3-token answers, never in the 24 selected PG-19 books) planted into 3 PG-19 books (Coningsby, Old Mortality, The Ragged Trousered Philanthropists; first 180 chunks), 14 cells per book (n = 2, 4, 8, 16, 32 mentions x gap kinds close / mixed / far / edge), 426 mentions from a hand-written pool of 384 templates (checked: 5-gram overlap 0), arms NAT / SAME (earlier planted mentions of the same fact hidden from the rows of the later mention, window AND same chunk) / RAND (matched placebo), frozen-weight reads with 3 question frames and 16 candidates at W0 and after the 1st / 2nd / 4th / last mention and the book end, positive control at W0. Code: e62_templates.py e62_facts.py e62_plant.py e62_run.py e62_analyze.py (new). W0 screen: facts with margin > 0 are redrawn. EXPLORATORY; no threshold. Est. GPU ~1-1.5 h (UNMEASURED). P0b (auxiliary records, name swap, normal end read) is not part of this job.
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
cd "$(dirname "$0")/../../.." 2>/dev/null || true
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 29"; exit 29; }
D=scripts/exp07_write_ladder;O=$OUT/e62_p0
mkdir -p $O/smoke
sha256sum $D/e62_templates.py $D/e62_facts.py $D/e62_plant.py $D/e62_run.py $D/e62_analyze.py $D/e59_nat_write.py $D/e55_core.py
$PY $D/e62_facts.py | tail -1
$PY $D/e62_plant.py --selftest | tail -2 | cut -c1-200
$PY $D/e62_run.py --out $O/smoke --books 1 --max-chunks 40 2>&1 | grep -v "Missing metrics\|metrics not found" | tee $O/smoke/run.log
$PY - $O/smoke <<'PYE'
import json,sys,math,numpy as np
r=json.load(open(sys.argv[1]+'/e62_p0a.json'));B=r['books'][0];A=B['arms'];assert set(A)=={'NAT','SAME','RAND'},list(A)
for ar,x in A.items():
    st=x['steps'];assert len(st)==B['nch'] and all(math.isfinite(s['gnorm']) and s['gnorm']>0 for s in st),ar;assert len(x['reads'])==len(B['hooks']),(len(x['reads']),len(B['hooks']))
    print(ar,'sec',round(x['sec']),'masked pairs',sum(s['masked_pairs'] for s in st),'hidden key tokens',sum(s['hidden_key_tokens'] for s in st))
assert sum(s['masked_pairs'] for s in A['SAME']['steps'])>0 and sum(s['masked_pairs'] for s in A['RAND']['steps'])>0
print('E62 SMOKE PASSED')
PYE
rm -rf $O/smoke/states
$PY $D/e62_run.py --out $O 2>&1 | grep -v "Missing metrics\|metrics not found" | tee $O/run.log
rm -rf $O/states
$PY $D/e62_analyze.py --json $O/e62_p0a.json | tee $O/e62_p0a_summary.txt

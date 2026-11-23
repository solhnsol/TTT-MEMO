#!/bin/bash
# JOB 1344z = E61 InfiniteBench natural-novel pilot, SMOKE on the shortest pilot book (Hans 11/23 'run only the smoke on the shortest book and give the estimate for 16 books'; code e61_book_run.py (new; derived from e59_nat_write.py); data data/natural-text/infbench/ (longbook_qa_eng.jsonl from HF xinrongzhang2022/InfiniteBench, mask spans and distractors from the project library files); book aa0310e147, 91,136 tokens = 89 whole chunks, 1 pilot question (qid 142, answer Malaysia, 19 evidence sentences): arms NAT, SAME (later evidence sentences lose the keys of the earlier evidence sentences of the same question inside the 8-chunk window), RAND (placebo, same rows, same number of hidden sentences of matched length and chunk); reads = frozen weights, no book context, summed log-probability of [correct, 3 distractors] at W0, after every chunk with an evidence token and after the last chunk. Prints per-arm seconds for the 16-book estimate. EXPLORATORY. Est. 10-20 min.
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
cd "$(dirname "$0")/../../.." 2>/dev/null || true
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 28"; exit 28; }
D=scripts/exp07_write_ladder;O=$OUT/e61_smoke
mkdir -p $O
sha256sum $D/e61_book_run.py $D/e59_stage0_run.py $D/e59_nat_write.py data/natural-text/infbench/longbook_qa_eng.jsonl data/natural-text/infbench/infbench_pilot_mask_compact.tsv data/natural-text/infbench/infbench_pilot_distractors.tsv
$PY $D/e61_book_run.py --selftest | tail -3
$PY $D/e61_book_run.py --book aa0310e147 --out $O 2>&1 | tee $O/run.log | grep -v "Missing metrics\|metrics not found"
$PY - $O <<'PYE'
import json,sys,math,numpy as np
r=json.load(open(sys.argv[1]+'/e61_aa0310e147.json'));A=r['arms'];nch=r['nch']
assert set(A)=={'NAT','SAME','RAND'},list(A)
for ar,x in A.items():
    st=x['steps'];assert len(st)==nch and all(math.isfinite(s['gnorm']) and s['gnorm']>0 for s in st),ar
    assert len(x['reads'])==len(r['hook_chunks']),(ar,len(x['reads']),len(r['hook_chunks']))
    print(ar,'sec',round(x['sec']),'sec/chunk',round(x['sec']/nch,2),'masked pairs total',sum(s['masked_pairs'] for s in st),'reads',len(x['reads']))
assert sum(s['masked_pairs'] for s in A['SAME']['steps'])>0 and sum(s['masked_pairs'] for s in A['RAND']['steps'])>0
for ar in ('SAME','RAND'):print('check',ar,'max |d gnorm| vs NAT before the first masked chunk',A[ar].get('pre_first_mask_max_dgnorm_vs_NAT'))
W0=np.asarray(r['W0']);print('W0 margin (correct - mean distractor):',W0[:,0]-W0[:,1:].mean(1))
for ar in A:
    ks=sorted(int(k) for k in A[ar]['reads']);m=[ (lambda v:(v[0][0]-np.mean(v[0][1:])))(np.asarray(A[ar]['reads'][str(k)])) for k in ks];print(ar,'margin at read points',dict(zip(ks,np.round(m,2).tolist())))
print('E61 SMOKE PASSED')
PYE

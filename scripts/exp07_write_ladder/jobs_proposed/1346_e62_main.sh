#!/bin/bash
# JOB 1346 = E62 MAIN RUN (design 1 'as recommended', option D: 180-chunk units, 7 PG-19 test + 5 PG-19 train novels; NOT QUEUED: Hans decides): 12 DISTINCT novels (PG-19 test: Carmen Ariza, Ragged Trousered, Old Mortality, Fair Maid of Perth, The Crisis, Coningsby, Nature's Serial Story; PG-19 train: Mill on the Floss, Ralph the Heir, Night and Morning, Perpetual Curate, Dred) x 14 cells (n = 2..32 mentions x gap kinds close / mixed / far / edge), the 6 n-groups carry the 6 attribute types rotated from unit to unit, candidates = gold + 1-2 EXPOSED same-type decoys of the same book (same number of exposures) + unexposed words; arms NAT and SAME; ONE question frame; reads at W0 / 1st / 2nd / 4th / last mention / end; pre-write answer-token NLL; positive control at W0. Code: e62m_run.py (+ e62_plant.py e62_templates.py e62_facts.py), e62m_analyze.py, e62_frames.py (frame check on P0a). EXPLORATORY; no threshold. Est. GPU ~9.5 h (UNMEASURED for 1 frame; 2 arms x ~12 min per unit + ~15 min setup, W0 screening, smoke).
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
cd "$(dirname "$0")/../../.." 2>/dev/null || true
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 32"; exit 32; }
D=scripts/exp07_write_ladder;O=$OUT/e62_main
mkdir -p $O/smoke
sha256sum $D/e62m_run.py $D/e62m_analyze.py $D/e62_plant.py $D/e62_templates.py $D/e62_facts.py $D/e59_nat_write.py $D/e55_core.py data/natural-text/pg19_test/metadata.csv
$PY $D/e62_facts.py | tail -1
$PY $D/e62m_run.py --selftest | tail -3 | cut -c1-240
$PY $D/e62m_run.py --out $O/smoke --books 1 --max-chunks 40 2>&1 | grep -v "Missing metrics\|metrics not found" | tee $O/smoke/run.log
$PY - $O/smoke <<'PYE'
import json,sys,math,numpy as np
r=json.load(open(sys.argv[1]+'/e62m.json'));B=r['books'][0];A=B['arms'];assert set(A)=={'NAT','SAME'},list(A)
for ar,x in A.items():
    st=x['steps'];assert len(st)==B['nch'] and all(math.isfinite(s['gnorm']) and s['gnorm']>0 for s in st),ar;assert len(x['reads'])==len(B['hooks']),(len(x['reads']),len(B['hooks']))
    print(ar,'sec',round(x['sec']),'masked pairs',sum(s['masked_pairs'] for s in st),'hidden key tokens',sum(s['hidden_key_tokens'] for s in st))
assert sum(s['masked_pairs'] for s in A['SAME']['steps'])>0
S=np.asarray(B['W0']);assert S.shape[1]==1 and S.shape[2]==16;assert all(1<=len(f['exposed'])<=2 for f in B['facts'])
print('E62 MAIN SMOKE PASSED')
PYE
rm -rf $O/smoke/states
$PY $D/e62m_run.py --out $O 2>&1 | grep -v "Missing metrics\|metrics not found" | tee $O/run.log
rm -rf $O/states
$PY $D/e62m_analyze.py --json $O/e62m.json | tee $O/e62m_summary.txt

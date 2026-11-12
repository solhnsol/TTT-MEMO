#!/bin/bash
# JOB 1280i = E57 1280 INSTRUMENTATION AND t3 MEASUREMENT (A1 v12 4.11 (a); HELD until (1) the signed instrument manifest and the 4.2 code review are recorded and (2) the principal investigator's own sentence in the server-queue thread): instrumentation seeds 91, 92, 93, 94 on the cell v3_32_4_a16 (64 chunks, 8 branch chunks per seed, k = 8), each seed: the runner (natural trajectory + branch measurements), the p16 check (p = 16 cell, 20 chunks, three arms), the INPUT_SCHEMA record; then the t3 sensitivity decision by e57_1280_t3_sensitivity.py (absolute criterion 0.01 nat/token, option (a) kept or option (b)). The outputs are measurement outputs (instrumentation seeds), never part of the confirmatory analysis; the same runs give the variance components at k = 8 (choose_k) and the c - nc noise floor. ESTIMATE about 4 x (45 + 6) min ~ 3.4 h (smoke: 164-211 s per branch chunk, one final read about 5 min, p16 three arms of 20 chunks about 6 min).
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
cd "$(dirname "$0")/../../.." 2>/dev/null || true
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
D=scripts/exp07_write_ladder
for T in e57_1280_margin e57_1280_validity e57_1280_gainnp e57_1280_recomp e57_1280_runner_records e57_1280_t3_sensitivity e57_1280_p16 e57_1280_driver; do $PY $D/$T.py --selftest | tail -1; done
$PY $D/e57_oracle_run.py --selftest | tail -1
O=$OUT/e57_1280_instr; mkdir -p $O
# A1 2.2: the study-level determinism declaration, written before the first branch comparison and never changed afterwards
$PY - $O <<'PYE'
import sys,json,os,datetime,subprocess,platform
o=sys.argv[1];v={'determinism_declaration':{'deterministic_operations_available':True,'reason':'XLA_FLAGS contains --xla_gpu_deterministic_ops=true; the fallback tolerances of A1 2.3 are not used','declared_utc':datetime.datetime.utcnow().isoformat()+'Z'},'XLA_FLAGS':os.environ.get('XLA_FLAGS'),'python':platform.python_version(),'git_head':subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True).stdout.strip(),'manifest_tag':'instr-manifest-1280-v1'}
for m in ('jax','jaxlib','numpy','scipy'):
    try:v[m]=__import__(m).__version__
    except Exception as e:v[m]=str(e)
json.dump(v,open(f'{o}/versions.json','w'),indent=1);print('determinism declaration written')
PYE
for SEED in 91 92 93 94; do
  $PY $D/e57_1280_runner.py --seed $SEED --out $O --k 8 --tag r1280i
  $PY $D/e57_1280_p16.py --seed $SEED --out $O --tag p16i
  $PY $D/e57_1280_driver.py --record --out $O --seed $SEED > /dev/null
done
$PY - $O <<'PYE'
import json,sys
o=sys.argv[1]
for s in (91,92,93,94):
    R=json.load(open(f'{o}/results_s{s}.json'));ks=sorted(int(k) for k in R['results']);ch=[]
    for k in ks:
        t=R['results'][str(k)]['t3'];ch.append(dict(i=k,s_i_empty=False,nll_orig=t['nll_orig'],nll_sub=t['nll_sub'],lp_orig=t['lp_orig'],lp_sub=t['lp_sub']))
    json.dump(dict(seed=s,chunks=ch,leak=any(R['results'][str(k)]['t3']['leak'] for k in ks),fail_4_2=False,complete=True),open(f'{o}/t3_input_s{s}.json','w'))
PYE
$PY $D/e57_1280_t3_sensitivity.py --json $O/t3_input_s91.json $O/t3_input_s92.json $O/t3_input_s93.json $O/t3_input_s94.json --out $O/t3_sensitivity_report.json | tail -12
echo "outputs: $O/{results,validity,record,p16}_s9{1,2,3,4}.json and $O/t3_sensitivity_report.json"

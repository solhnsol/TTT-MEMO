#!/bin/bash
# JOB 1280c = E57 1280 CONFIRMATORY RUN (seeds 101..130; HELD until freeze B (the instrument manifest of the confirmatory state, the t3 measurement of job 1280i, option (a) or (b) and the A1b value of c_nc_tol, signed and published) and the principal investigator's sentence). Per seed: the runner (natural trajectory + 8 branch chunks), the p16 check, nothing else; the runner's result-bearing log text is FILTERED (A1 4.11 (d): logs contain only validity statuses and progress). VALIDITY-FIRST STAGING: after all seeds, the validity records and the p16 records are hashed and judged BEFORE any record is built or evaluated; if any t2/t3/p16 is false or any t3 status is not PASS the job stops (exit 20) and an amendment A2 must be written before anyone opens a result (A1 4.8, 4.11 (d)). Only if every validity flag is true: the INPUT_SCHEMA records are built and the registered analysis is run. ESTIMATE about 30 x 52 min = 26 h (smoke: 164-211 s per branch chunk; 1280i measures the exact value).
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
cd "$(dirname "$0")/../../.." 2>/dev/null || true
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
: "${SEEDS:=101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130}"
: "${ANALYZER:=e57_1280_analyze_a1b.py}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
D=scripts/exp07_write_ladder; O=$OUT/e57_1280_main; mkdir -p $O
for T in e57_1280_margin e57_1280_validity e57_1280_gainnp e57_1280_recomp e57_1280_runner_records e57_1280_p16 e57_1280_driver; do $PY $D/$T.py --selftest | tail -1; done
[ -f $O/versions.json ] || $PY - $O <<'PYE'
import sys,json,os,datetime,subprocess,platform
o=sys.argv[1];v={'determinism_declaration':{'deterministic_operations_available':True,'reason':'XLA_FLAGS contains --xla_gpu_deterministic_ops=true','declared_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},'XLA_FLAGS':os.environ.get('XLA_FLAGS'),'git_head':subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True).stdout.strip()}
json.dump(v,open(f'{o}/versions.json','w'),indent=1)
PYE
for SEED in $SEEDS; do
  if [ ! -f $O/p16_s$SEED.json ]; then
    $PY $D/e57_1280_runner.py --seed $SEED --out $O --k 8 --tag r1280 2>&1 | sed -E '/1280 branch chunk/ s/: n_F.*/ (result-bearing details filtered)/'
    $PY $D/e57_1280_p16.py --seed $SEED --out $O --tag p16
  fi
done
# validity-first staging: hash and judge the validity records before any record is built
( cd $O && sha256sum validity_s*.json p16_s*.json > validity_hashes.txt )
$PY - $O $SEEDS <<'PYE'
import sys,json
o=sys.argv[1];bad=[]
for s in sys.argv[2:]:
    V=json.load(open(f'{o}/validity_s{s}.json'));P=json.load(open(f'{o}/p16_s{s}.json'))
    if not (V['t2_ok'] and V['t3_ok'] and P['p16_ok'] and V['t3_status']=='PASS' and V.get('isolation_ok',True)):bad.append((s,V['t2_ok'],V['t3_status'],P['p16_ok']))
if bad:print('VALIDITY NOT CONFIRMED (seed, t2, t3 status, p16):',bad);print('STOP: write the amendment A2 before any result field is opened (A1 4.8, 4.11 (d))');sys.exit(20)
print('all validity flags true for',len(sys.argv)-2,'seeds; hashes in validity_hashes.txt')
PYE
for SEED in $SEEDS; do $PY $D/e57_1280_driver.py --record --out $O --seed $SEED > /dev/null; done
$PY - $O $SEEDS <<'PYE'
import sys,json
o=sys.argv[1];json.dump([json.load(open(f'{o}/record_s{s}.json')) for s in sys.argv[2:]],open(f'{o}/records.json','w'))
PYE
$PY $D/$ANALYZER --json $O/records.json > $O/analysis_1280.json
echo "outputs: $O/analysis_1280.json (records.json, validity_hashes.txt)"

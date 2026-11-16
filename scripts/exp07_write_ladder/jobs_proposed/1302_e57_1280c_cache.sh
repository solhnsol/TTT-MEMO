#!/bin/bash
# JOB 1302 = E57 1280-CACHE: CACHE-PATH ONE-STEP MARGINS at the 1280c branch chunks (EXPLORATORY; rules fixed in advance (TOST of A vs NATMASK, delta 0.21 on both margins, Hans 11/16); not queued until the freeze tag is pushed; GPU one job at a time; registration = docstring of e57_1280c_cache_analyze.py, its sha256 is frozen and a signed tag is pushed to the private registry BEFORE the first seed is run). Seeds 101..108 of job 1280c: for each seed the natural NATS trajectory is re-run with exactly the arguments of 1280c (e57_1280c_cache_runner.py imports the unchanged e57_1280_runner.py) and at the SAME branch chunks (read from $OUT/e57_1280_main/results_s<seed>.json) one update step is taken from W_t with the update-path masks NATS, NATMASK (ALLQ256), CACHE_A (cached oracle), CACHE_AP (cached placebo) on the cached window K,V and the one-step margins are read (primary logsumexp, secondary largest-wrong). After every seed a CPU identity check against the stored 1280c results (isolation gnorm / coef, NATS and NATMASK margins and NC / A / A_P NLL arrays EXACTLY equal) runs: a mismatch aborts (exit 13). ESTIMATE: per seed ~11 min trajectory + 8 branch chunks x ~30-40 s (recompute path, t3, gain_noprior and p16 are not repeated) = ~16-20 min, about 2.2-2.7 h for 8 seeds. Output $OUT/e57_1280_cache/ (results_cache_s<seed>.json), result $OUT/e57_1280_cache_result.json. Run through the job runner (XLA allocator and nvrtc LD_LIBRARY_PATH are set there).
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
cd "$(dirname "$0")/../../.." 2>/dev/null || true
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
D=scripts/exp07_write_ladder; O=$OUT/e57_1280_cache; REF=$OUT/e57_1280_main; mkdir -p $O
cat > $OUT/e57_1280_cache_pinned_sha256.txt <<'PIN'
f05caa0a48114758a38a3834365dc70224ef3b5aee66a0b16f033da638027c48  scripts/exp07_write_ladder/e55_core.py
24f1836b5fe2b1a5b436ba21c0b7a31335f0e578ef73e2f5d8e78d93e6eba0da  scripts/exp07_write_ladder/e57_natmask_run.py
a0f07082a03392f6dc67cd9b2f16ea51436ec4632132bc2ab439f2b92f00e371  scripts/exp07_write_ladder/e57_oracle_masks.py
85a8fdb8711e9b6426ab6815e5b3854fa970dc96b27a33b90ffd3fb510db2fea  scripts/exp07_write_ladder/e57_oracle_run.py
714b08ea3b90fb951f4d1cb15635df9a5964004938c20c99805363bffba74ad4  scripts/exp07_write_ladder/ladder.py
a123948130e487fc8a43eec07bdc9b0daf2ea2200bdfeb06c6d4b7aa28f4cb8d  scripts/exp07_write_ladder/s3.py
c48a57e093ed8c4d652e74e4009f43a8fc5ac0e6a02e75c64feafd830a5e651e  scripts/exp07_write_ladder/e57_qband_run.py
091506dbe7b8a565ccd53594f2a0f9651d80ae07d86448047766cb811cfd5847  scripts/exp07_write_ladder/e57_closure_masks.py
bf34cfd17662a142db5756f199b34d71268e4f37ef6f955428f177430f1b8d9b  scripts/exp07_write_ladder/e57_1280_runner.py
0ffb23948bc3761f13a797b692c1b484aa356ef9769799f2675322c58d6be18a  scripts/exp07_write_ladder/e57_1280_margin.py
93a5ef0d2b485b55b30eb6136770db532e72dae00137a6f89b2e09f1390a8ad6  scripts/exp07_write_ladder/e57_1280_gainnp.py
07642582cad12b6f47990472b2e3fc3044f411fb6783e48890e741ba0fe8266a  scripts/exp07_write_ladder/e57_1280_validity.py
08d6575775cd4b1b156318c182b3eb13bb888541d35b36c2b3386152546cfac6  scripts/exp07_write_ladder/e57_1280_recomp.py
da853fe6a2e0f855eef787511a9afcc9fab2936d5308578be5cd9f022c95f6db  scripts/exp07_write_ladder/e57_1280_runner_records.py
142c5918cbb5cb5e26fb70ec417fa97d9c97e5c4a2fc76d5547c500f87c13039  scripts/exp07_write_ladder/e57_1280c_cache_runner.py
3ec68756c1b461f60acd9613f840562d557b0f43e8ce832f8c0e6ab9e8bac437  scripts/exp07_write_ladder/e57_1280c_cache_analyze.py
PIN
sha256sum -c $OUT/e57_1280_cache_pinned_sha256.txt || { echo "REUSE REFUSED: a pinned file differs"; exit 12; }
SEEDS="101 102 103 104 105 106 107 108"
for s in $SEEDS; do [ -f $REF/results_s$s.json ] || { echo "missing $REF/results_s$s.json (job 1280c output): exit 14"; exit 14; }; done
$PY $D/e57_1280c_cache_runner.py --selftest | tail -1; $PY $D/e57_1280c_cache_analyze.py --selftest | tail -1
for s in $SEEDS; do
  BR=$($PY -c "import json,sys;print(','.join(str(k) for k in json.load(open('$REF/results_s$s.json'))['branches']))")
  $PY $D/e57_1280c_cache_runner.py --seed $s --out $O --branches $BR --tag cache1280
  $PY $D/e57_1280c_cache_analyze.py --check --dir $O --ref $REF --seeds $s | tail -3
done
$PY $D/e57_1280c_cache_analyze.py --check --dir $O --ref $REF --seeds 101,102,103,104,105,106,107,108 > /dev/null
$PY $D/e57_1280c_cache_analyze.py --analyze --dir $O --seeds 101,102,103,104,105,106,107,108 --json $OUT/e57_1280_cache_result.json > /dev/null
echo "outputs: $OUT/e57_1280_cache_result.json"

#!/bin/bash
# JOB 1297b = COMPLETION OF JOB 1297 (repair, Hans 11/15: authority to fix and rerun so that the experiment runs to the end). Job 1297 finished all 8 synthetic seeds (141..148, 3 arms) and the DCLM raw runs W0 / NATS / NATMASK_256 for the 9 fresh streams 16..24, then stopped (rc=1) in e57_preservation_analyze.py (g2 reads st0..st15 from index 0; the job's streams are st16..st24: FileNotFoundError, a script bug, not a data problem). Nothing registered was changed. This job (1) runs the remaining DCLM raw run NATMASK_1024 on the same 9 streams with the same command as job 1297, (2) computes the preservation statistics with the unchanged g2(D, 9) through the new wrapper e57_band1024_dclm_idx.py (links st16..st24 to st0..st8), (3) runs the frozen analyzer e57_band1024_confirm_analyze.py exactly as in job 1297. ESTIMATE about 25 min. Runs through the job runner (XLA allocator and nvrtc LD_LIBRARY_PATH set there).
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
cd "$(dirname "$0")/../../.." 2>/dev/null || true
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
D=scripts/exp07_write_ladder; O=$OUT/e57_band1024_conf; DC=$O/dclm
sha256sum -c <<PIN
a912fcc260e62742f445b00d9f44a2c317fcbb868d4cb7d80b413fc9f83da8d0  $D/e57_band1024_confirm_analyze.py
PIN
mkdir -p $DC/w1024
for f in $DC/E57_dclmttt_W0_st*.json $DC/E57_dclmttt_NATS_st*.json; do ln -sf "$(realpath $f)" $DC/w1024/; done
$PY $D/e57_dclm_band_run.py --band 1024 --stage run --out $DC/w1024 --k0 0 --n-streams 25 --docs-per-stream 8 --streams 16,17,18,19,20,21,22,23,24 --arms NATMASK
for W in 256 1024; do $PY $D/e57_band1024_dclm_idx.py --dclm $DC --band $W --json $OUT/e57_band1024_conf_preservation_$W.json; done
$PY $D/e57_band1024_confirm_analyze.py --dir $O --seeds 141,142,143,144,145,146,147,148 --dclm-json $OUT/e57_band1024_conf_preservation_1024.json --json $OUT/e57_band1024_conf_result.json > /dev/null
echo "outputs: $OUT/e57_band1024_conf_result.json, $OUT/e57_band1024_conf_preservation_{256,1024}.json"

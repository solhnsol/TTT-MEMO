#!/bin/bash
# JOB 1303 = E55 ISO 8-SEED EXTENSION (EXPLORATORY; not queued until Hans' sentence; GPU one job at a time; same margin definition as the existing E55 rows): cell v3_16_1, seeds 61..68. (1) ISO-STD / ISO-MOM / ISO-ADAM (e55_run.py q1a, tag q1a8, cumulative update norm matched to the NAT run of the same seed, reads at exposures 16 / 32 / 64 / 128, out-of-window) are RERUN for seeds 61, 62, 63 and must be bit identical to the stored runs of job 192 (tag q1a): the first seed (61) is the smoke test and the identity gate (every read value, W0 value and logged step value exactly equal; a difference stops the job, exit 13, nothing further runs); (2) ISO for seeds 64, 65 (NAT runs exist, stored seeds 61..65); (3) NAT for seeds 66, 67, 68 (the command of jobs 238 / 239: run --tag v3a --arms NAT --plan "SD=16,32,64,128|128"); (4) ISO for seeds 66, 67, 68; (5) analysis e55_iso8_check.py --analyze (mean and two-sided 95% t interval per exposure for NAT, ISO-STD, ISO-MOM, ISO-ADAM, and ISO - NAT paired by seed). ESTIMATE ~80 min: 4 model loads for ISO / 3 for NAT (~3-5 min each) + ISO 3 arms x ~2 min per seed x 8 seeds ~48 min + NAT 3 x ~6.6 min. Output $OUT/E55_q1a8_<arm>_s<seed>.{saved,read}.json, $OUT/E55_v3a_NAT_s66..68, $OUT/e55_iso8_result.json. Restart-safe (finished arms / seeds are reused). Run through the job runner (XLA allocator and nvrtc LD_LIBRARY_PATH are set there).
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
cd "$(dirname "$0")/../../.." 2>/dev/null || true
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
D=scripts/exp07_write_ladder; D4=scripts/exp04_what_ttt_retains; mkdir -p $OUT
cat > $OUT/e55_iso8_pinned_sha256.txt <<'PIN'
71f0fc416f7446c5adcf371ba97244da770c0f4427a3a1f506db6f6ee60b0b2d  scripts/exp07_write_ladder/e55_run.py
f05caa0a48114758a38a3834365dc70224ef3b5aee66a0b16f033da638027c48  scripts/exp07_write_ladder/e55_core.py
c4c8a0d4e386b38a3c3b4846c6d97c5fc272571f8f5136e1c3101a7654ef8169  scripts/exp07_write_ladder/e55_iso.py
9827e34dcde64b13a477ee876bac87d8d5ba5588e8d22d22c826bb53b41169b6  scripts/exp07_write_ladder/e55_gpu.py
b82e7ab635c2cf811205cf4b7f8d845c731179ddf9a73ab258a2a04ee8a943bd  scripts/exp07_write_ladder/e55_swap.py
a6d5069490711f73ebc71bb7996925f8d2b9114cb7eaddffa0a81f76061e4c97  scripts/exp07_write_ladder/e55_read.py
b72c0290d4219de62659c0506248871d4fcf4a345401d5c63910dde1fd1197cc  scripts/exp07_write_ladder/e55_iso8_check.py
a123948130e487fc8a43eec07bdc9b0daf2ea2200bdfeb06c6d4b7aa28f4cb8d  scripts/exp07_write_ladder/s3.py
714b08ea3b90fb951f4d1cb15635df9a5964004938c20c99805363bffba74ad4  scripts/exp07_write_ladder/ladder.py
b867b6f680f31d1d76b4e204b435122fab9a7eb52f3cbbd196f3203bb5be98f7  scripts/exp04_what_ttt_retains/gen_facts.py
1cf0caa825534a4f961088555a463da4385ce302c4a0dbc83e41f0c14bb07025  scripts/exp04_what_ttt_retains/gen_world_ctl.py
eeaaa91359078b8a64021c6224c2e80386059af2548d3d437269cdfa6be16a64  scripts/exp04_what_ttt_retains/frame_bank.py
866eb5989467202750f225b47df60327e0b6a976e40f77d26089e5498b1702b1  scripts/exp04_what_ttt_retains/frames300.txt
PIN
sha256sum -c $OUT/e55_iso8_pinned_sha256.txt || { echo "REUSE REFUSED: a pinned file differs"; exit 12; }
for s in 61 62 63 64 65; do for f in E55_v3a_NAT_s$s.saved.json E55_v3a_NAT_s$s.read.json E55_q1a_ISO-STD_s$s.read.json; do [ -f $OUT/$f ] || [ $s -gt 63 -a "${f#E55_q1a}" != "$f" ] || { echo "missing $OUT/$f: exit 14"; exit 14; }; done; done
$PY $D4/gen_facts.py --selftest | tail -1; $PY $D/e55_read.py --selftest | tail -1; $PY $D/e55_iso8_check.py --selftest | tail -1
Q="$PY $D/e55_run.py q1a --cell v3_16_1 --tag q1a8 --bg pool --out $OUT --read-apps 16,32,64,128 --norm-files $OUT/E55_v3a_NAT_s{seed}.saved.json --q1-arms ISO-STD,ISO-MOM,ISO-ADAM"
$Q --seeds 61
$PY $D/e55_iso8_check.py --check --dir $OUT --seeds 61 | tail -3
$Q --seeds 62,63
$PY $D/e55_iso8_check.py --check --dir $OUT --seeds 62,63 | tail -3
$Q --seeds 64,65
for SD in 66 67 68; do
  [ -f $OUT/E55_v3a_NAT_s$SD.saved.json ] || $PY $D/e55_run.py run --cell v3_16_1 --bg pool --tag v3a --out $OUT --decoy names --save-apps 128 --canary-apps 128 --seeds $SD --arms NAT --plan "$SD=16,32,64,128|128"
done
$Q --seeds 66,67,68
$PY $D/e55_iso8_check.py --analyze --dir $OUT --seeds 61,62,63,64,65,66,67,68 --json $OUT/e55_iso8_result.json > /dev/null
echo "outputs: $OUT/e55_iso8_result.json"

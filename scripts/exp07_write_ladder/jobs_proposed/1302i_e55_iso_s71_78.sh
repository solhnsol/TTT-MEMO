#!/bin/bash
# JOB 1302i = E55 ISO-STD ON THE 1283 SEEDS 71..78 (EXPLORATORY; not queued until Hans' sentence; GPU one job at a time; Hans 11/16 option A): ISO-STD (e55_run.py q1a, tag q1a71, offline isolated-row training, cumulative update norm matched to the norm source of the seed, reads at exposures 16 / 32 / 64 / 128, out-of-window) on cell v3_16_1, seeds 71..78 = the seeds of job 1283 (NATS, NATMASK256) and 1293 (NATMASK1024). NATS-NORM ASSUMPTION CHECK first (e55_iso71_nchk.py): the E57 NATS trajectory (e57_oracle_run.py --arms NATS, tag nchk) rerun on seed 61 must equal the stored E55 NAT run of seed 61 (E55_v3a_NAT_s61: every step value update_norm / ce / gnorm / coef and every out read value at 16 / 32 / 64 / 128 exactly equal). EQUAL -> the norms of the 1283 NATS runs (seeds 71..78) are used as the matching source; DIFFERENT -> the job runs the E55 NAT arm (e55_run.py run, tag v3a, the command of jobs 238 / 239) on seeds 71..78 and uses its norms; the source used and the differences are logged and stored in $OUT/e55_iso71_nchk.json. Then ISO-STD on seeds 71..78 and the analysis e57_align71_analyze.py (shared table of the exposure curve on seeds 71..78; arms of job 1302o are listed as missing if that job has not run) -> $OUT/e57_align71_result.json. ESTIMATE: check ~15 min + ISO 8 seeds x ~2 min + model load ~5 min = ~40 min (EQUAL) or ~1.6 h (DIFFERENT: + 8 NAT runs x ~6.6 min). Run through the job runner (XLA allocator and nvrtc LD_LIBRARY_PATH are set there).
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
cd "$(dirname "$0")/../../.." 2>/dev/null || true
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
D=scripts/exp07_write_ladder; D4=scripts/exp04_what_ttt_retains; mkdir -p $OUT
cat > $OUT/e55_iso71_pinned_sha256.txt <<'PIN'
71f0fc416f7446c5adcf371ba97244da770c0f4427a3a1f506db6f6ee60b0b2d  scripts/exp07_write_ladder/e55_run.py
f05caa0a48114758a38a3834365dc70224ef3b5aee66a0b16f033da638027c48  scripts/exp07_write_ladder/e55_core.py
c4c8a0d4e386b38a3c3b4846c6d97c5fc272571f8f5136e1c3101a7654ef8169  scripts/exp07_write_ladder/e55_iso.py
9827e34dcde64b13a477ee876bac87d8d5ba5588e8d22d22c826bb53b41169b6  scripts/exp07_write_ladder/e55_gpu.py
b82e7ab635c2cf811205cf4b7f8d845c731179ddf9a73ab258a2a04ee8a943bd  scripts/exp07_write_ladder/e55_swap.py
a6d5069490711f73ebc71bb7996925f8d2b9114cb7eaddffa0a81f76061e4c97  scripts/exp07_write_ladder/e55_read.py
a123948130e487fc8a43eec07bdc9b0daf2ea2200bdfeb06c6d4b7aa28f4cb8d  scripts/exp07_write_ladder/s3.py
714b08ea3b90fb951f4d1cb15635df9a5964004938c20c99805363bffba74ad4  scripts/exp07_write_ladder/ladder.py
85a8fdb8711e9b6426ab6815e5b3854fa970dc96b27a33b90ffd3fb510db2fea  scripts/exp07_write_ladder/e57_oracle_run.py
24f1836b5fe2b1a5b436ba21c0b7a31335f0e578ef73e2f5d8e78d93e6eba0da  scripts/exp07_write_ladder/e57_natmask_run.py
a0f07082a03392f6dc67cd9b2f16ea51436ec4632132bc2ab439f2b92f00e371  scripts/exp07_write_ladder/e57_oracle_masks.py
c48a57e093ed8c4d652e74e4009f43a8fc5ac0e6a02e75c64feafd830a5e651e  scripts/exp07_write_ladder/e57_qband_run.py
653c7c4ef4feee0b8b0dc1a651b260a0fe034f5cc21f6afa8bc097afc0edb627  scripts/exp07_write_ladder/e55_iso71_nchk.py
cc4b67286f17c04d5e2b3853c13a8536c99059dc0ee06ede3411272c22f36123  scripts/exp07_write_ladder/e57_align71_analyze.py
b867b6f680f31d1d76b4e204b435122fab9a7eb52f3cbbd196f3203bb5be98f7  scripts/exp04_what_ttt_retains/gen_facts.py
1cf0caa825534a4f961088555a463da4385ce302c4a0dbc83e41f0c14bb07025  scripts/exp04_what_ttt_retains/gen_world_ctl.py
eeaaa91359078b8a64021c6224c2e80386059af2548d3d437269cdfa6be16a64  scripts/exp04_what_ttt_retains/frame_bank.py
866eb5989467202750f225b47df60327e0b6a976e40f77d26089e5498b1702b1  scripts/exp04_what_ttt_retains/frames300.txt
PIN
sha256sum -c $OUT/e55_iso71_pinned_sha256.txt || { echo "REUSE REFUSED: a pinned file differs"; exit 12; }
for f in E55_v3a_NAT_s61.saved.json E55_v3a_NAT_s61.read.json; do [ -f $OUT/$f ] || { echo "missing $OUT/$f: exit 14"; exit 14; }; done
for SD in 71 72 73 74 75 76 77 78; do [ -f $OUT/e57_k0conf16/E55_k0c16_NATS_s$SD.saved.json ] || { echo "missing 1283 NATS s$SD: exit 14"; exit 14; }; done
$PY $D4/gen_facts.py --selftest | tail -1; $PY $D/e55_read.py --selftest | tail -1; $PY $D/e55_iso71_nchk.py --selftest | tail -1; $PY $D/e57_align71_analyze.py --selftest | tail -1
O1=$OUT/e55_iso71_nchk; mkdir -p $O1
$PY $D/e57_oracle_run.py --seed 61 --out $O1 --masks none --mask-arm ALLQ256 --arms NATS --save-apps 16,32,64,128 --nat-loss-every 8 --tag nchk
set +e; $PY $D/e55_iso71_nchk.py --new $O1 --ref $OUT --seed 61 --json $OUT/e55_iso71_nchk.json; RC=$?; set -e
if [ $RC -eq 0 ]; then
  NORM="$OUT/e57_k0conf16/E55_k0c16_NATS_s{seed}.saved.json"; echo "NORM SOURCE: NATS of job 1283 (check EQUAL)"
else
  echo "NATS != NAT on seed 61 (check DIFFERENT, rc $RC): running the NAT arm on seeds 71..78 for the norm matching"
  for SD in 71 72 73 74 75 76 77 78; do
    [ -f $OUT/E55_v3a_NAT_s$SD.saved.json ] || $PY $D/e55_run.py run --cell v3_16_1 --bg pool --tag v3a --out $OUT --decoy names --save-apps 128 --canary-apps 128 --seeds $SD --arms NAT --plan "$SD=16,32,64,128|128"
  done
  NORM="$OUT/E55_v3a_NAT_s{seed}.saved.json"; echo "NORM SOURCE: E55 NAT seeds 71..78"
fi
$PY $D/e55_run.py q1a --cell v3_16_1 --tag q1a71 --bg pool --out $OUT --read-apps 16,32,64,128 --norm-files "$NORM" --q1-arms ISO-STD --seeds 71,72,73,74,75,76,77,78
$PY $D/e57_align71_analyze.py --dir $OUT --seeds 71,72,73,74,75,76,77,78 --json $OUT/e57_align71_result.json > /dev/null
echo "outputs: $OUT/E55_q1a71_ISO-STD_s<seed>.{saved,read}.json, $OUT/e55_iso71_nchk.json, $OUT/e57_align71_result.json"

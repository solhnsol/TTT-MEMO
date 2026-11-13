#!/bin/bash
# JOB 1295 = E57 BLOCK-STREAM DILUTION EXPERIMENT (exploratory; HELD until the principal investigator's own sentence, after jobs 1293 / 1294): 4 blocks of 16 facts streamed one after the other (block j in chunks 16 j .. 16 j + 15, every fact of the block once per chunk = 16 exposures in 16 different frames, 64 chunks, k = 64, ids 0..63), NATS vs NATMASK (--mask-arm ALLQ256), K0 = 0, token diag every 8 chunks as in job 1283, reads (out and in-window, all items, 16 same-block candidates) after every block (appearances 16, 32, 48, 64 = lag 0..3 blocks). Wrapper e57_blockstream_run.py of the unchanged e57_oracle_run.py (pinned SHA-256 below, abort 12). SEEDS (new worlds, none used before): default 81 = smoke / effect-size run (2 runs, ESTIMATE about 12-15 min incl. load; per-run time is a guess scaled from job 1283: 8.6 min per 128-chunk run with 4 reads; here 64 chunks but 4 read points of 68 items out + in-window); the main run is SEEDS="82 83 84 85 86 87 88" (14 runs, about 1.5 h). Output $OUT/e57_blockstream/ (tag bs); analysis: e57_blockstream_analyze.py --dir $OUT/e57_blockstream --tag bs --seeds 81,...,88.
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
cd "$(dirname "$0")/../../.." 2>/dev/null || true
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
: "${SEEDS:=81}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
D=scripts/exp07_write_ladder; O=$OUT/e57_blockstream; mkdir -p $O
$PY $D/e57_oracle_run.py --selftest | tail -1
$PY $D/e57_blockstream_run.py --selftest | tail -1
$PY $D/e57_blockstream_analyze.py --selftest | tail -1
cat > $O/pinned_sha256.txt <<'PIN'
85a8fdb8711e9b6426ab6815e5b3854fa970dc96b27a33b90ffd3fb510db2fea  scripts/exp07_write_ladder/e57_oracle_run.py
24f1836b5fe2b1a5b436ba21c0b7a31335f0e578ef73e2f5d8e78d93e6eba0da  scripts/exp07_write_ladder/e57_natmask_run.py
a0f07082a03392f6dc67cd9b2f16ea51436ec4632132bc2ab439f2b92f00e371  scripts/exp07_write_ladder/e57_oracle_masks.py
f05caa0a48114758a38a3834365dc70224ef3b5aee66a0b16f033da638027c48  scripts/exp07_write_ladder/e55_core.py
714b08ea3b90fb951f4d1cb15635df9a5964004938c20c99805363bffba74ad4  scripts/exp07_write_ladder/ladder.py
a123948130e487fc8a43eec07bdc9b0daf2ea2200bdfeb06c6d4b7aa28f4cb8d  scripts/exp07_write_ladder/s3.py
PIN
sha256sum -c $O/pinned_sha256.txt || { echo "REUSE REFUSED: a runner file differs from the pinned hash"; exit 12; }
for SEED in $SEEDS; do
  $PY $D/e57_blockstream_run.py --cell blk16x4_c16 --seed $SEED --out $O --masks none --mask-arm ALLQ256 --arms NATS,NATMASK --save-apps 16,32,48,64 --nat-loss-every 8 --tag bs
done
echo "outputs: $O/E55_bs_{NATS,NATMASK}_s<seed>.{saved,read}.json (analysis: e57_blockstream_analyze.py --dir $O --tag bs --seeds ...)"

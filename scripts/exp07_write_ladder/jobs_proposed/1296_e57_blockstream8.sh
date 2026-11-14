#!/bin/bash
# JOB 1296 = E57 BLOCK-STREAM DILUTION, MAIN EXPERIMENT (exploratory; K0 = 0 only; HELD until the principal investigator's own sentence; GPU: one job at a time): 8 blocks of 16 facts streamed one after the other (blk16x8_c16: block j in chunks 16 j .. 16 j + 15, 16 exposures in 16 different frames per fact, 128 chunks, k = 128), arms NATS vs NATMASK with band ALLQ1024 (--mask-arm ALLQ1024). The ASSIGNMENT of the 128 fact ids to the 8 blocks is a seed-specific random permutation (--shuffle-blocks; perm = random.Random("E57bs|<seed>|blockperm").shuffle, recorded per seed in $O/E55_bs8_perm_s<seed>.json). Reads after every block (appearances 16 .. 128 = lag 0..7, lag -1.. = untrained control): 'out' (neutral 8192 tail) for all items, in-window reads only at lag 0 (--in-lag0-only: facts of the block just finished + control items). Primary result: NATMASK1024 - NATS per lag with t 95% CI over seeds and the difference of the dilution slopes (lag 0..7, 'out'); per-arm slopes are secondary. Seeds 82 83 84 85 (new worlds; 81 = job 1295 smoke, 86-88 unused reserve). Saved states: last read point only (128). ESTIMATE ~100 min per run (8 runs, ~13 h; scaled from job 1295: 4.6 s per item-read, 1056 out + 160 in-window item-reads per run + 128 chunks x 3 s; to be calibrated on the first run). DCLM preservation of ALLQ1024 is job 1294 (w1024, already run, PASS): not repeated here.
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
cd "$(dirname "$0")/../../.." 2>/dev/null || true
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
: "${SEEDS:=82 83 84 85}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
D=scripts/exp07_write_ladder; O=$OUT/e57_blockstream8; mkdir -p $O
$PY $D/e57_oracle_run.py --selftest | tail -1
$PY $D/e57_blockstream8_run.py --selftest | tail -2
$PY $D/e57_blockstream8_analyze.py --selftest | tail -1
cat > $O/pinned_sha256.txt <<'PIN'
f05caa0a48114758a38a3834365dc70224ef3b5aee66a0b16f033da638027c48  scripts/exp07_write_ladder/e55_core.py
38dfcdb642cb744601ddc73a1c8c0f6906fe89900dc1f46880bbec8782260310  scripts/exp07_write_ladder/e57_blockstream8_analyze.py
4b8ab161224bc31e4bc5be6f547e74c82afb656acf2ebc41d9e664cd65b30c62  scripts/exp07_write_ladder/e57_blockstream8_run.py
24f1836b5fe2b1a5b436ba21c0b7a31335f0e578ef73e2f5d8e78d93e6eba0da  scripts/exp07_write_ladder/e57_natmask_run.py
a0f07082a03392f6dc67cd9b2f16ea51436ec4632132bc2ab439f2b92f00e371  scripts/exp07_write_ladder/e57_oracle_masks.py
85a8fdb8711e9b6426ab6815e5b3854fa970dc96b27a33b90ffd3fb510db2fea  scripts/exp07_write_ladder/e57_oracle_run.py
714b08ea3b90fb951f4d1cb15635df9a5964004938c20c99805363bffba74ad4  scripts/exp07_write_ladder/ladder.py
a123948130e487fc8a43eec07bdc9b0daf2ea2200bdfeb06c6d4b7aa28f4cb8d  scripts/exp07_write_ladder/s3.py
PIN
sha256sum -c $O/pinned_sha256.txt || { echo "REUSE REFUSED: a runner file differs from the pinned hash"; exit 12; }
for SEED in $SEEDS; do
  $PY $D/e57_blockstream8_run.py --cell blk16x8_c16 --seed $SEED --out $O --masks none --mask-arm ALLQ1024 --arms NATS,NATMASK --save-apps 128 --nat-loss-every 8 --tag bs8 --shuffle-blocks --in-lag0-only
done
echo "outputs: $O/E55_bs8_{NATS,NATMASK}_s<seed>.{saved,read}.json, $O/E55_bs8_perm_s<seed>.json (analysis: $PY $D/e57_blockstream8_analyze.py --dir $O --tag bs8 --seeds 82,83,84,85 --nb 8 --facts 16 --json $OUT/e57_blockstream8_result.json)"

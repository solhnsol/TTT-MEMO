#!/bin/bash
# JOB 1300 = E57 QA BLOCK-STREAM, BAND EXTENSION (EXPLORATORY; K0 = 0 only; GPU one job at a time; Hans 11/16 'recommended' via the core-claims thread): the job-1299 design (same plans, same stream ids, 6 bundles of 8 paragraphs, seeds 101..104, same reads: OUT-tail reads of all items after every bundle in conditions A / B and the in-window lag-0 read) with the arms NATMASK ALLQ256 and NATMASK ALLQ2048 added; NATS and NATMASK ALLQ1024 are reused from job 1299 (qabs_s<seed>_reads.json), W0 and the item meta of the new reads are asserted equal to those of 1299 in the analysis. Code: e57_qabs_band.py imports the unchanged e57_qabs_run.py (sha adefe1cf...) and applies counted textual substitutions (arm list, file names). Analysis: bands 256 / 1024 / 2048 minus NATS per lag -1..5 and in0 for first-token dA / dAB and margins S / N (e57_qabs_margin definitions). ESTIMATE: write 2 arms x 4 seeds x ~4.5 min = ~36 min, reads ~4 x ~7 min = ~28 min, about 1.1 h in total. Output $OUT/e57_qabs/ (qabs_s<seed>_writes_band.json, _reads_band.json), result $OUT/e57_qabs_band_result.json.
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
cd "$(dirname "$0")/../../.." 2>/dev/null || true
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
D=scripts/exp07_write_ladder; Q=$D/e57_qabs_band.py; O=$OUT/e57_qabs
cat > $OUT/qabs_band_pinned_sha256.txt <<'PIN'
f05caa0a48114758a38a3834365dc70224ef3b5aee66a0b16f033da638027c48  scripts/exp07_write_ladder/e55_core.py
24f1836b5fe2b1a5b436ba21c0b7a31335f0e578ef73e2f5d8e78d93e6eba0da  scripts/exp07_write_ladder/e57_natmask_run.py
a0f07082a03392f6dc67cd9b2f16ea51436ec4632132bc2ab439f2b92f00e371  scripts/exp07_write_ladder/e57_oracle_masks.py
85a8fdb8711e9b6426ab6815e5b3854fa970dc96b27a33b90ffd3fb510db2fea  scripts/exp07_write_ladder/e57_oracle_run.py
714b08ea3b90fb951f4d1cb15635df9a5964004938c20c99805363bffba74ad4  scripts/exp07_write_ladder/ladder.py
a123948130e487fc8a43eec07bdc9b0daf2ea2200bdfeb06c6d4b7aa28f4cb8d  scripts/exp07_write_ladder/s3.py
c48a57e093ed8c4d652e74e4009f43a8fc5ac0e6a02e75c64feafd830a5e651e  scripts/exp07_write_ladder/e57_qband_run.py
2c39d805224b167c43810423c513fdb36f20f4ecfcc5358f38f4567affaed42e  scripts/exp07_write_ladder/e57_pqa12_run.py
bb7a8f150af850b86a544f0a68b3e3e12cd8599c4f8aa2a7c51e2823f5c12292  scripts/exp07_write_ladder/e57_pqa6_run.py
b19e986904bada8c0e49d3e97c84c18fa4ea13edd08dd8b1d62cca87f4c9c885  scripts/exp07_write_ladder/e57_reads.py
adefe1cfd9a56f5a301c0e2233b30f0f41932628afe4ce1f567dd1257b9e8e87  scripts/exp07_write_ladder/e57_qabs_run.py
5a7b7058aebb7c75170fe6a995b71f220a18cc5923e398f9e0bfcd8c236f64a9  scripts/exp07_write_ladder/e57_qabs_margin.py
fcc9ed686cdf3d20f02c382272c0855db8bad6a2ac48e21e95c3be9cde0b46cf  scripts/exp07_write_ladder/e57_qabs_band.py
cc13e5089a426e5c98f5821bcb6f89038bcefc9863a84f49974562d8b98092c6  scripts/exp07_write_ladder/e57_pqa2_stream.py
6ec96f3b7d734e7a4df295c0dc28776ca2b41f23063109e985882c5866624e99  scripts/exp07_write_ladder/e57_pqa3_stream.py
9c00d674562abb9f4a9e6ba0a595b0f1f5c2f053d851d40831e9f4a8a9b32158  data/public-qa/rewrite128/paragraphs128.json
934cbec55b4fc2b280acd3eb609b2acf7a44c7f77060eb9a627bd0400f798ce7  data/public-qa/rewrite128/rewrites_16_qa128.json
PIN
sha256sum -c $OUT/qabs_band_pinned_sha256.txt || { echo "REUSE REFUSED: a pinned file differs"; exit 12; }
for s in 101 102 103 104; do for f in qabs_s${s}_plan.json qabs_s${s}_ids.npy qabs_s${s}_reads.json; do [ -f $O/$f ] || { echo "missing $O/$f (job 1299 output): exit 14"; exit 14; }; done; done
$PY $Q --selftest | tail -1
$PY $Q --stage write --out $O --seeds 101,102,103,104 --nb 6
$PY $Q --stage read --out $O --seeds 101,102,103,104 --nb 6
$PY $Q --stage analyze --out $O --seeds 101,102,103,104 --nb 6 --json $OUT/e57_qabs_band_result.json
echo "outputs: $OUT/e57_qabs_band_result.json"

#!/bin/bash
# JOB 1298s = E57 QA BLOCK-STREAM SMOKE (technical only, not a result; K0 = 0; GPU one job at a time; runs right after job 1297): 1 seed (101), 2 bundles of 8 paragraphs (32 chunks of 1024 tokens, 16 exposures per paragraph), arms NATS and NATMASK ALLQ1024 (clipped SGD, mask from chunk 0), reads of all bundles after each bundle (OUT tail, conditions A and B), in-window lag-0 reads (tail = real last 8 chunks) and the 2-bundle lag table. ESTIMATE about 30 min. Checks: the stream packs (selftest), write and read run to the end, the canary is stored, the analysis produces lags -1 and 0 and 1. Output $OUT/e57_qabs_smoke/ ; marker file SMOKE_OK is written only if all stages finished (the main job 1299 refuses to start without it). It does not look at the effect size for pass/fail.
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
cd "$(dirname "$0")/../../.." 2>/dev/null || true
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
D=scripts/exp07_write_ladder; Q=$D/e57_qabs_run.py
mkdir -p $OUT
cat > $OUT/qabs_pinned_sha256.txt <<'PIN'
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
cc13e5089a426e5c98f5821bcb6f89038bcefc9863a84f49974562d8b98092c6  scripts/exp07_write_ladder/e57_pqa2_stream.py
6ec96f3b7d734e7a4df295c0dc28776ca2b41f23063109e985882c5866624e99  scripts/exp07_write_ladder/e57_pqa3_stream.py
9c00d674562abb9f4a9e6ba0a595b0f1f5c2f053d851d40831e9f4a8a9b32158  data/public-qa/rewrite128/paragraphs128.json
934cbec55b4fc2b280acd3eb609b2acf7a44c7f77060eb9a627bd0400f798ce7  data/public-qa/rewrite128/rewrites_16_qa128.json
PIN
sha256sum -c $OUT/qabs_pinned_sha256.txt || { echo "REUSE REFUSED: a pinned file differs"; exit 12; }
$PY $Q --selftest | tail -1
O=$OUT/e57_qabs_smoke; mkdir -p $O
$PY $Q --stage build --out $O --seeds 101 --nb 2
$PY $Q --stage write --out $O --seeds 101 --nb 2 --nbundles 2
$PY $Q --stage read  --out $O --seeds 101 --nb 2 --nbundles 2
$PY $Q --stage analyze --out $O --seeds 101 --nb 2 --nbundles 2 --json $O/qabs_smoke_result.json
$PY - $O <<PYE
import json,sys
o=sys.argv[1];r=json.load(open(o+"/qabs_smoke_result.json"));assert {"-1","0","1","in0"}<=set(r["lags"]),r["lags"].keys()
w=json.load(open(o+"/qabs_s101_writes.json"));assert all(k in w for k in ("NATS","NATMASK")) and all(len(w[a]["states"])==2 for a in w)
open(o+"/SMOKE_OK","w").write("ok")
print("SMOKE_OK")
PYE

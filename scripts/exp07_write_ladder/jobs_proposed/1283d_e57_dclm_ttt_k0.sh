#!/bin/bash
# JOB 1283d = E57 DCLM PRESERVATION GATE G2, REVISED (Council 11/11; HELD until Hans' sentence): 16 streams x 8 DCLM validation documents (stream i = documents 8i..8i+7 of data/dclm_val_subset/val_c0_200x8193.u32le.bin, first 8192 tokens each, 64 chunks per stream), arms W0 (frozen), NATS, NATMASK, mask from chunk 0 (K0 = 0; NATMASKC is not run in the gate job), prequential NLL (every chunk scored before its own update), every stream starts from W0 with a fresh suffix state. Judgement by e57_preservation_analyze.py (one-sided t df=15 AND exact sign-flip inversion, conservative label). Output $OUT/e57_dclm_ttt_k0_16s/. The earlier 4-stream K0=8 job 1274 is superseded (legacy_k08). ESTIMATE ~2.6-2.9 h (W0 ~25 min, NATS ~40 min, NATMASK ~75 min + per-stream setup).
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
cd "$(dirname "$0")/../../.." 2>/dev/null || true
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
D=scripts/exp07_write_ladder
$PY $D/e57_dclm_ttt_run.py --selftest | tail -1
$PY $D/e57_preservation_analyze.py --selftest | tail -1
$PY $D/e57_m1_domains.py --selftest | tail -1
O=$OUT/e57_dclm_ttt_k0_16s
$PY $D/e57_dclm_ttt_run.py --stage run --out $O --k0 0 --n-streams 16 --docs-per-stream 8 --streams 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 --arms W0,NATS,NATMASK
$PY $D/e57_preservation_analyze.py --g2-dir $O --json $OUT/e57_preservation_g2_k0.json
echo "outputs: $OUT/e57_preservation_g2_k0.json"

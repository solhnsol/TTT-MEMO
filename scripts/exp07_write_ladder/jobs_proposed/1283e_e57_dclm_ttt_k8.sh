#!/bin/bash
# JOB 1283e = E57 DCLM PRESERVATION GATE G2 at K0 = 8 (exploratory secondary evidence; pre-specification reports/E57-1283e-prespec.md; HELD until the principal investigator's own sentence in the server-queue thread): same 16 streams x 8 DCLM validation documents as job 1283d (stream i = documents 8i..8i+7 of data/dclm_val_subset/val_c0_200x8193.u32le.bin, first 8192 tokens, 64 chunks per stream), arm NATMASK with the mask from chunk 8 (K0 = 8, runner option --k0 8, K0 tag 8 in the record); W0 and NATS (K0-independent) are REUSED from job 1283d (output experiments/exp07_write_ladder/e57_dclm_ttt_k0_16s) through symbolic links; judgement by e57_preservation_analyze.py with the rule of 1283d (one-sided t df = 15 and exact sign-flip inversion, delta = +0.01 nat/token). Reuse conditions (abort 12 / 13): the full SHA-256 of the runner and analysis files equals the values pinned below (their current values, unchanged since job 1285b); a rerun of NATMASK at K0 = 0 on stream 0 with this job's code path is exactly equal (nll_pre, ce, update_norm, gnorm, coef, ids_sha) to the 1283d record of stream 0. ESTIMATE about 35 min: 16 streams x about 111 s (1283d measurement) + gate about 2 min + selftests about 3 min. Output $OUT/e57_dclm_ttt_k8_16s/ and $OUT/e57_preservation_g2_k8.json.
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
R=$OUT/e57_dclm_ttt_k0_16s; O=$OUT/e57_dclm_ttt_k8_16s; mkdir -p $O
# reuse condition (1): full SHA-256 of the runner and analysis files (pinned before the run)
cat > $O/pinned_sha256.txt <<'PIN'
c3195d34eeedebd84c04513caa289999b5d20d1a2968a2a42ca074ffe77786f5  scripts/exp07_write_ladder/e57_dclm_ttt_run.py
e60142bb5f90b8157672e66a9f0716606efb0e54e30b129e64ee83df1c53849a  scripts/exp07_write_ladder/e57_preservation_analyze.py
bee38401f9eb4b8c6fede16a0a96f3409d6c6b47807a15973bdc2dab5eee9c25  scripts/exp07_write_ladder/e57_m1_domains.py
24f1836b5fe2b1a5b436ba21c0b7a31335f0e578ef73e2f5d8e78d93e6eba0da  scripts/exp07_write_ladder/e57_natmask_run.py
a0f07082a03392f6dc67cd9b2f16ea51436ec4632132bc2ab439f2b92f00e371  scripts/exp07_write_ladder/e57_oracle_masks.py
f05caa0a48114758a38a3834365dc70224ef3b5aee66a0b16f033da638027c48  scripts/exp07_write_ladder/e55_core.py
714b08ea3b90fb951f4d1cb15635df9a5964004938c20c99805363bffba74ad4  scripts/exp07_write_ladder/ladder.py
a123948130e487fc8a43eec07bdc9b0daf2ea2200bdfeb06c6d4b7aa28f4cb8d  scripts/exp07_write_ladder/s3.py
PIN
sha256sum -c $O/pinned_sha256.txt || { echo "REUSE REFUSED: a runner or analysis file differs from the pinned hash"; exit 12; }
# reuse condition (2): NATMASK at K0 = 0 on stream 0 with this code path equals the 1283d record exactly
$PY $D/e57_dclm_ttt_run.py --stage run --out $O/gate --k0 0 --n-streams 16 --docs-per-stream 8 --streams 0 --arms NATMASK
$PY - $O $R <<'PYE'
import json,sys
o,r=sys.argv[1],sys.argv[2];a=json.load(open(f'{o}/gate/E57_dclmttt_NATMASK_st0.json'));b=json.load(open(f'{r}/E57_dclmttt_NATMASK_st0.json'))
ok=all(a[k]==b[k] for k in ('nll_pre','ce','update_norm','gnorm','coef','ids_sha','docs','n_chunks','mask_arm','lr_mult','K0'))
print('gate NATMASK K0 = 0 stream 0 equal to 1283d:',ok);sys.exit(0 if ok else 13)
PYE
for A in W0 NATS; do for S in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do ln -sf "$(realpath $R/E57_dclmttt_${A}_st${S}.json)" $O/E57_dclmttt_${A}_st${S}.json; done; done
$PY $D/e57_dclm_ttt_run.py --stage run --out $O --k0 8 --n-streams 16 --docs-per-stream 8 --streams 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 --arms NATMASK
$PY $D/e57_preservation_analyze.py --g2-dir $O --json $OUT/e57_preservation_g2_k8.json
echo "outputs: $OUT/e57_preservation_g2_k8.json"

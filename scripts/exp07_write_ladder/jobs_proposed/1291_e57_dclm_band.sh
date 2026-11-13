#!/bin/bash
# JOB 1291 = E57 BAND-DISTANCE EXPERIMENT, DCLM PRESERVATION (exploratory; pre-specification reports/E57-distB-prespec.md; HELD until the principal investigator's own sentence): NATMASK with the band mask of W tokens (W = 64, 512) on the 16 streams x 8 DCLM documents of job 1283d, K0 = 0, through e57_dclm_band_run.py (wrapper of the unchanged e57_dclm_ttt_run.py). W0 and NATS records are REUSED from job 1283d (symlinks); W = 256 is the 1283d NATMASK. GATE (abort 12 / 13): pinned SHA-256 of the three runner files; the wrapper with band 256 on stream 0 must reproduce the 1283d NATMASK stream 0 record exactly (nll_pre, ce, update_norm, gnorm, coef, ids_sha). Judgement per width by e57_preservation_analyze.py (rule of 1283d). ESTIMATE about 35 min per width, gate included about 1 h. Output $OUT/e57_dclm_band/w<W>/
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
cd "$(dirname "$0")/../../.." 2>/dev/null || true
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
D=scripts/exp07_write_ladder; R=$OUT/e57_dclm_ttt_k0_16s; O=$OUT/e57_dclm_band; mkdir -p $O
$PY $D/e57_dclm_band_run.py --selftest | tail -1
$PY $D/e57_preservation_analyze.py --selftest | tail -1
cat > $O/pinned_sha256.txt <<'PIN'
c3195d34eeedebd84c04513caa289999b5d20d1a2968a2a42ca074ffe77786f5  scripts/exp07_write_ladder/e57_dclm_ttt_run.py
e60142bb5f90b8157672e66a9f0716606efb0e54e30b129e64ee83df1c53849a  scripts/exp07_write_ladder/e57_preservation_analyze.py
bee38401f9eb4b8c6fede16a0a96f3409d6c6b47807a15973bdc2dab5eee9c25  scripts/exp07_write_ladder/e57_m1_domains.py
PIN
sha256sum -c $O/pinned_sha256.txt || { echo "REUSE REFUSED: a runner file differs from the pinned hash"; exit 12; }
ALL=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
# GATE: band 256 on stream 0 equals the 1283d NATMASK record
$PY $D/e57_dclm_band_run.py --band 256 --stage run --out $O/gate --k0 0 --n-streams 16 --docs-per-stream 8 --streams 0 --arms NATMASK
$PY - $O/gate $R <<'PYE'
import json,sys
a=json.load(open(sys.argv[1]+'/E57_dclmttt_NATMASK_st0.json'));b=json.load(open(sys.argv[2]+'/E57_dclmttt_NATMASK_st0.json'))
ok=all(a[k]==b[k] for k in ('nll_pre','ce','update_norm','gnorm','coef','ids_sha','mask_arm','K0','docs'))
print('gate band256 stream0 vs 1283d NATMASK:',ok);sys.exit(0 if ok else 13)
PYE
for W in 64 512; do
  mkdir -p $O/w$W
  for f in $R/E57_dclmttt_W0_st*.json $R/E57_dclmttt_NATS_st*.json; do ln -sf "$(realpath $f)" $O/w$W/; done
  $PY $D/e57_dclm_band_run.py --band $W --stage run --out $O/w$W --k0 0 --n-streams 16 --docs-per-stream 8 --streams $ALL --arms NATMASK
  $PY $D/e57_preservation_analyze.py --g2-dir $O/w$W --json $OUT/e57_preservation_band$W.json
done
echo "outputs: $OUT/e57_preservation_band{64,512}.json"

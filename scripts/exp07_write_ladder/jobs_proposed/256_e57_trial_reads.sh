#!/bin/bash
# JOB 256 = E57 TRIAL (GPU): NAT single arm over ONE cycle of the natural news stream (871 independent lead articles, global date order, 280 chunks of 1024 tokens, one inner update per chunk) with canary at W0 and every 24 chunks, and with memory reads of the FROZEN BANK (bank_v1.json) at W0 and at the end of the cycle: condition OUT (neutral 8192-token SQuAD tail) and IN (natural window = the last 8192 stream tokens; per question window_info present / absent by gold surface-form match). Reads are read-only (fp32 copy of the weights). Log rows as in E57 section 39 (cumulative exposure n_k, gold logprob, rank, n candidates, hard-only rank / n, accuracy all / hard, per-candidate logprobs). Needs the frozen bank (job 250) and runs on the GPU after job 244.
set -eo pipefail
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder
D4=scripts/exp04_what_ttt_retains
COMMIT=e632178a94ca5143d31c247244ec0bde4927e06b
FILES=" $D/e55_run.py $D/e55_core.py $D/e55_gpu.py $D/e55_swap.py $D/e55_read.py $D/e55_analyze.py $D/e55_metrics.py $D/e55_ic_report.py $D/e55_iso.py $D/e55_diag.py $D/e55_votegrid.py $D/e55_uv.py $D/s3.py $D/rope_fly.py $D/e55_sentdump.py $D/e55_sentdump_labels.py $D/e55_sentsel.py $D/e55_sentgrid.py $D/e55_selmask.py $D/e55_gates.py $D/e55_bgbooks.py $D/e55_bgbooks_sha.json $D/e55_rev.py $D4/gen_facts.py $D4/gen_world_ctl.py $D4/frame_bank.py $D4/frames300.txt"
LADDER_OK="714b08ea3b90"
S3_OLD="f261120d0f03"
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
if [ -z "$NO_INSTALL" ]; then
  GL=$(sha256sum $D/ladder.py | cut -c1-12); [ "$GL" = "$LADDER_OK" ] || { echo "lab ladder.py ($GL) is not the patched W1 version ($LADDER_OK): exit 6"; exit 6; }
  git cat-file -e $COMMIT^{commit} || { echo "commit $COMMIT not in the local clone (exit 4)"; exit 4; }
  S3H=$(sha256sum $D/s3.py | cut -c1-12)
  S3OK=$( (echo "$S3_OLD"; echo a16e6dd0d4c9; for C in $(git log --format=%H --all -- $D/s3.py); do git show $C:$D/s3.py | sha256sum | cut -c1-12; done) )
  echo "$S3OK" | grep -qx "$S3H" || { echo "lab s3.py ($S3H) is not any committed version of s3.py: not overwritten (exit 6)"; exit 6; }
  git checkout $COMMIT -- $FILES
fi
if [ -z "$SKIP_VERIFY" ]; then
  while read HH F; do
    [ -n "$HH" ] || continue
    G=$(sha256sum $D/$F | cut -c1-12)
    [ "$G" = "$HH" ] || { echo "VERSION MISMATCH $F: have $G, expected $HH (exit 5)"; exit 5; }
  done <<'MANIFEST'
71f0fc416f74 e55_run.py
f05caa0a4811 e55_core.py
9827e34dcde6 e55_gpu.py
b82e7ab635c2 e55_swap.py
a6d506949071 e55_read.py
4a811650bcd8 e55_analyze.py
30769e241d5e e55_metrics.py
e8c32d9ffa65 e55_ic_report.py
c4c8a0d4e386 e55_iso.py
725d18d16b65 e55_diag.py
597294ca312b e55_votegrid.py
6d9e9b62cecd e55_uv.py
a123948130e4 s3.py
76802d72e2c4 rope_fly.py
16bf1b4d7808 e55_sentdump.py
df41b7d13739 e55_sentdump_labels.py
3728d3954e71 e55_sentsel.py
24b0b4970d22 e55_sentgrid.py
21324f7fcdb7 e55_selmask.py
eb8bc59bf9a0 e55_gates.py
3d8657814bbb e55_bgbooks.py
75ff3cee6437 e55_bgbooks_sha.json
2215511cee2a e55_rev.py
b867b6f680f3 ../exp04_what_ttt_retains/gen_facts.py
1cf0caa82553 ../exp04_what_ttt_retains/gen_world_ctl.py
eeaaa9135907 ../exp04_what_ttt_retains/frame_bank.py
866eb5989467 ../exp04_what_ttt_retains/frames300.txt
MANIFEST
  echo "installed code verified against the manifest"
fi
# EXPECTED RUN TIME (Hans confirms before it runs): ESTIMATE ONLY, unmeasured, GPU: model load 7-20 min once (plus the fp32 reader copy) + 280 updates at 3.5-4.6 s = 16-22 min + 13 canary readings (guess 1-2 min each) + reads: one Packed scorer for the OUT tail and one for the IN window (about 75 packed chunks of the 1021 candidate rows, 2 prefix passes of 9.2K tokens per chunk, guess 2-4 min each) + about 1 min of scoring per condition and weight state (W0, end) = about 1.0-1.7 h in total. The GPU part (e57_trial_run.py) was NOT run on a GPU, only --check-only on CPU with a pseudo bank; a failure at the first GPU call is possible. Commit c9e5e6a6b60adad98bf917d00cd19678d8c4c1a6. NOT APPROVED yet.
if [ -z "$NO_INSTALL" ]; then git cat-file -e c9e5e6a6b60adad98bf917d00cd19678d8c4c1a6^{commit} || { echo "commit c9e5e6a6b60adad98bf917d00cd19678d8c4c1a6 missing (exit 4)"; exit 4; }; git checkout c9e5e6a6b60adad98bf917d00cd19678d8c4c1a6 -- $D/e57_reads.py $D/e57_trial_run.py; git checkout 57c50b5ce5dee7e3efea0ba036c79ab4eb6e6f02 -- $D/e57_canary_run.py; git checkout 5c835409e9a32b77829fdb20e4022a6547f06705 -- $D/e56/e56_text.py $D/e57/e57_common.py; git checkout c9e5e6a6b60adad98bf917d00cd19678d8c4c1a6 -- $D/e57/e57_bank.py; fi
for x in "b19e986904ba e57_reads.py" "9ee4658ee168 e57_trial_run.py" "fe17f085f622 e57_canary_run.py" "88076d1f3dd1 e57/e57_bank.py" "a0590cd53539 e57/e57_common.py" "f09dcb7e69eb e56/e56_text.py"; do set -- $x; G=$(sha256sum $D/$2 | cut -c1-12); [ "$G" = "$1" ] || { echo "VERSION MISMATCH $2 (exit 5)"; exit 5; }; done
S=$D
$PY $D4/gen_facts.py --selftest | tail -1
$PY $S/e57_reads.py --selftest
BUNDLE=${BUNDLE:-$OUT/e57_news}; [ -f "$BUNDLE/e57_bundle/bundle_store.pkl" ] || { echo "bundle store missing (exit 7)"; exit 7; }
if [ -n "$CPU_ONLY" ]; then CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu $PY $S/e57_trial_run.py --bundle "$BUNDLE" --out "$OUT/_trial_check" --check-only; echo "CPU_ONLY: done before any GPU step"; exit 0; fi
[ -f "$BUNDLE/e57_bank/bank_v1.json" ] || { echo "frozen bank bank_v1.json missing: job 250 first (exit 7)"; exit 7; }
$PY $S/e57_trial_run.py --bundle "$BUNDLE" --out "$OUT" --check-only
$PY $S/e57_trial_run.py --bundle "$BUNDLE" --out "$OUT"

#!/bin/bash
# JOB 254 = E57 NAT SINGLE-ARM CANARY TRAJECTORY over ONE cycle of the natural news stream (GPU, forward + one inner update per chunk, no reads, no questions): 871 independent lead articles of the 10 events in global date order, 280 chunks of 1024 tokens (real tokenizer), canary (pg11 generation NLL, pg1342 tail NLL) at W0 and every 24 chunks. Compare E55: +0.023 (SGD) at 128 updates, limit +0.10.
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
# EXPECTED RUN TIME (Hans confirms before it runs): ESTIMATE ONLY, unmeasured, GPU: model load 7-20 min once + 280 updates at 3.5-4.6 s = 16-22 min + 13 canary readings (guess 1-2 min each) = about 0.8-1.4 h. The GPU part of e57_canary_run.py was NOT run on a GPU (only --check-only on CPU); a failure at the first GPU call is possible.
# Commit of the runner: 57c50b5ce5dee7e3efea0ba036c79ab4eb6e6f02. NOT APPROVED yet.
if [ -z "$NO_INSTALL" ]; then git cat-file -e 57c50b5ce5dee7e3efea0ba036c79ab4eb6e6f02^{commit} || { echo "commit 57c50b5ce5dee7e3efea0ba036c79ab4eb6e6f02 missing (exit 4)"; exit 4; }; git checkout 57c50b5ce5dee7e3efea0ba036c79ab4eb6e6f02 -- $D/e57_canary_run.py; git checkout 5c835409e9a32b77829fdb20e4022a6547f06705 -- $D/e56/e56_text.py $D/e57/e57_common.py; git checkout 2c05884a35c84a2c9906b8beebd9d2a8d58d5979 -- $D/e57/e57_bank.py; fi
for x in "fe17f085f622 e57_canary_run.py" "7843a666e1a8 e57/e57_bank.py" "a0590cd53539 e57/e57_common.py" "f09dcb7e69eb e56/e56_text.py"; do set -- $x; G=$(sha256sum $D/$2 | cut -c1-12); [ "$G" = "$1" ] || { echo "VERSION MISMATCH $2 (exit 5)"; exit 5; }; done
S=$D
$PY $D4/gen_facts.py --selftest | tail -1
$PY $S/e55_core.py --selftest 2>&1 | tail -1 || true
BUNDLE=${BUNDLE:-$OUT/e57_news}; [ -f "$BUNDLE/e57_bundle/bundle_store.pkl" ] || { echo "bundle store missing (exit 7)"; exit 7; }
$PY $S/e57_canary_run.py --bundle "$BUNDLE" --out "$OUT" --check-only
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
$PY $S/e57_canary_run.py --bundle "$BUNDLE" --out "$OUT"

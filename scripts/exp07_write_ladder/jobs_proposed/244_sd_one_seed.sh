#!/bin/bash
# JOB 244 SEED = E55 32.12 sentence dump (per-head AL, forward only, GPU) of the seeds 62, 63, 64 (one seed per call, see wrapper 244); then candidate-1 confirmation and the FROZEN learned head weighting of seed 61 (e55_sentgrid_ext.py --apply-coef; coefficients trained with labels on seed 61).
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
  git cat-file -e 9b9364319dde87d5366ac7c725e645c326f1ba8b^{commit} && git checkout 9b9364319dde87d5366ac7c725e645c326f1ba8b -- $D/e55_sentgrid_ext.py
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
S=$D
SEED="${SEED:?SEED=62|63|64}"   # 61 = candidate selection seed; 62 and 63 = confirmation seeds (same job, SEED=62 / SEED=63); needs E55_q1b_SENTPQ_s$SEED.saved.json (exit 7 if missing)
case "$SEED" in 62|63|64) ;; *) echo "SEED must be 62, 63 or 64 (61 exists)" >&2; exit 3;; esac
for F in q1b_SENTPQ; do [ -f $OUT/E55_${F}_s$SEED.saved.json ] || [ -n "$CPU_ONLY" ] || { echo "file E55_${F}_s$SEED.saved.json missing (exit 7)"; exit 7; }; done
$PY $D4/gen_facts.py --selftest | tail -1
$PY $S/e55_read.py --selftest | tail -1
$PY $S/e55_swap.py | tail -1
$PY $S/e55_sentgrid.py --selftest
[ "$(sha256sum $S/e55_sentgrid_ext.py | cut -c1-12)" = 7de0dd3b97f0 ] || { echo "VERSION MISMATCH e55_sentgrid_ext.py (exit 5)"; exit 5; }
$PY $S/e55_sentgrid_ext.py --selftest | tail -1
$PY $S/e55_gates.py --cell v3_16_1 --seed $SEED --default-refs --require-refs --out $OUT/e55_gates_sd_s$SEED.md > $OUT/e55_gates_sd_s$SEED.log 2>&1 || { tail -30 $OUT/e55_gates_sd_s$SEED.log; echo "background gates failed (exit 7)"; exit 7; }
tail -4 $OUT/e55_gates_sd_s$SEED.md
$PY $S/e55_run.py dry --cell v3_16_1 --seed $SEED --bg pool --arms NAT | tail -3
$PY $S/e55_run.py sentdump --cell v3_16_1 --seed $SEED --bg pool --diag-chunks 16,32,64,127 --norm-ref-tag q1b --out $OUT --check-only
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
REP=$OUT/e55_sentgrid_report_s$SEED.md
if [ -f $REP ] && [ -z "$RERUN" ]; then echo "report $REP exists: not redone (exit 8; RERUN=1 to redo)"; exit 8; fi
$PY $S/e55_run.py sentdump --cell v3_16_1 --seed $SEED --bg pool --diag-chunks 16,32,64,127 --norm-ref-tag q1b --out $OUT
CAND=$OUT/e55_sentgrid_report_s61_candidates.json; [ -f $CAND ] || { echo "candidate file $CAND missing (exit 7)"; exit 7; }
$PY $S/e55_sentgrid.py --dump $OUT/e55_sentdump_s$SEED.npz --labels $OUT/e55_sentdump_labels_s$SEED.npz --candidates $CAND --tag "seed $SEED" --out $REP
COEF=$OUT/e55_sentgrid_ext_s61_coef.npz; [ -f $COEF ] || { echo "coefficient file $COEF missing (exit 7)"; exit 7; }
$PY $S/e55_sentgrid_ext.py --apply-coef $COEF --dump $OUT/e55_sentdump_s$SEED.npz --labels $OUT/e55_sentdump_labels_s$SEED.npz --out $OUT/e55_sentgrid_ext_apply_s$SEED.md
tail -40 $OUT/e55_sentgrid_ext_apply_s$SEED.md

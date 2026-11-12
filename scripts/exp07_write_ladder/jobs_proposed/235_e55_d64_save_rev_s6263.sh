#!/bin/bash
# JOB 235 = job 210 for seeds 62,63 (forward reads, norm files) + job 224 reverse reading in ONE job: the weights at a = 128 are saved in the same run as the inline reads (decomp --save-state-app 128 --with-inline-reads; commit e632178 = c12ba3c + that flag + unit U49). Replaces separate jobs 210 (seeds 62,63) and 224 (seeds 62,63). Approved by Hans 2026-10-25T15:16Z ("응 추천대로 가자"). EXPECTED about 6.5-7.5 h (210 measured 3.2 h per seed; the state save adds little; reverse reading about 10-15 min per seed). Pre-registered criterion (advisor, 15:04Z): in all 3 seeds (61 from job 224, 62, 63) the reverse gain (masked - NAT) > 0 and the seed-pooled interval excludes 0.
set -eo pipefail
PY=$(realpath -s "$PY")   # the runner exports a relative PY (.venv/bin/python): make it absolute BEFORE the cd below, WITHOUT resolving symlinks (uv venv; readlink -f breaks it)
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder
D4=scripts/exp04_what_ttt_retains
COMMIT=e632178a94ca5143d31c247244ec0bde4927e06b
FILES="$D/e55_run.py $D/e55_core.py $D/e55_gpu.py $D/e55_swap.py $D/e55_read.py $D/e55_analyze.py $D/e55_metrics.py $D/e55_ic_report.py $D/e55_iso.py $D/e55_diag.py $D/e55_votegrid.py $D/e55_uv.py $D/s3.py $D4/gen_facts.py $D4/gen_world_ctl.py $D4/frame_bank.py $D/e55_gates.py $D4/frames300.txt $D/rope_fly.py $D/e55_bgbooks.py $D/e55_bgbooks_sha.json $D/e55_rev.py"
LADDER_OK="714b08ea3b90"
S3_OLD="f261120d0f03"   # lab s3.py after job 189 (2-D mask version); the pre-189 version a16e6dd0d4c9 is accepted too
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
if [ -z "$NO_INSTALL" ]; then
  GL=$(sha256sum $D/ladder.py | cut -c1-12); [ "$GL" = "$LADDER_OK" ] || { echo "lab ladder.py ($GL) is not the patched W1 version ($LADDER_OK): exit 6"; exit 6; }
  git cat-file -e $COMMIT^{commit} || { echo "commit $COMMIT not in the local clone: git fetch origin claude/project-thread-rwhdac first (exit 4)"; exit 4; }
  S3NEW=$(git show $COMMIT:$D/s3.py | sha256sum | cut -c1-12); S3H=$(sha256sum $D/s3.py | cut -c1-12)
  S3OK=$( (echo "$S3_OLD"; echo a16e6dd0d4c9; for C in $(git log --format=%H --all -- $D/s3.py); do git show $C:$D/s3.py | sha256sum | cut -c1-12; done) )   # every s3.py version ever committed in the local clone (all branches, so a later job that left a newer committed s3.py on the lab does not block this one) is accepted (a local edit never is)
  echo "$S3OK" | grep -qx "$S3H" || { echo "lab s3.py ($S3H) is not any committed version of s3.py up to $COMMIT (new one: $S3NEW): not overwritten (exit 6)"; exit 6; }
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
b867b6f680f3 ../exp04_what_ttt_retains/gen_facts.py
1cf0caa82553 ../exp04_what_ttt_retains/gen_world_ctl.py
eeaaa9135907 ../exp04_what_ttt_retains/frame_bank.py
eb8bc59bf9a0 e55_gates.py
866eb5989467 ../exp04_what_ttt_retains/frames300.txt
76802d72e2c4 rope_fly.py
3d8657814bbb e55_bgbooks.py
75ff3cee6437 e55_bgbooks_sha.json
2215511cee2a e55_rev.py
MANIFEST
  echo "installed code verified against the manifest"
fi
S=$D
SEEDS="${SEEDS:-62,63}"
ARMS="${ARMS:-NAT,SENTPQ,SENTPQADAMH}"
DCLM=data/dclm_val_subset/val_c0_200x8193.u32le.bin
$PY $S/e55_bgbooks.py verify --dest data/narrative_gutenberg --pins $S/e55_bgbooks_sha.json || { echo "the extra background books are missing or differ from the pins (exit 7)"; exit 7; }
# job 210 for SEEDS (forward reads, as in job 210) and, in the SAME run, the inner weights saved at a = 128 (--save-state-app 128 --with-inline-reads); then the reverse reading (job 224 stage B) per seed
DEC="--cell v3_64_4 --tag d64 --bg pool --out $OUT --seeds $SEEDS --arms $ARMS --norm-files $OUT/E55_d64_NAT_s{seed}.saved.json --decomp-apps 16,32,64,128 --in-window --in-apps 128 --canary-all --save-state-app 128 --with-inline-reads"
$PY $D4/gen_facts.py --selftest | tail -1
$PY $S/e55_read.py --selftest | tail -1
$PY $S/e55_rev.py --selftest | tail -1
$PY $S/e55_swap.py | tail -1
$PY $S/e55_metrics.py --selftest | tail -1
$PY $S/e55_ic_report.py --selftest | tail -1
$PY $S/e55_bgbooks.py --selftest
for SD in $(echo $SEEDS | tr ',' ' '); do
  $PY $S/e55_gates.py --cell v3_64_4 --seed $SD --default-refs --require-refs --out $OUT/e55_gates_d64_s$SD.md > $OUT/e55_gates_d64_s$SD.log 2>&1 || { tail -30 $OUT/e55_gates_d64_s$SD.log; echo "background gates failed for seed $SD (exit 7)"; exit 7; }
  tail -4 $OUT/e55_gates_d64_s$SD.md
  $PY $S/e55_run.py revcheck --cell v3_64_4 --seed $SD --bg pool --tag d64 --arms NAT --out $OUT
done
$PY $S/e55_run.py dry --cell v3_64_4 --seed 61 --bg pool --arms NAT | tail -3
$PY $S/e55_run.py decomp $DEC --check-only
[ -f "$DCLM" ] && echo "DCLM ids present: $(stat -c %s $DCLM) bytes" || echo "DCLM ids file absent here (needed on the GPU run)"
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
[ -f "$DCLM" ] || { echo "DCLM ids file $DCLM missing (exit 7)"; exit 7; }
REP=$OUT/e55_d64_report_s6263.md
if [ -f $REP ] && [ -z "$RERUN" ]; then echo "report $REP exists: not redone (exit 8; RERUN=1 to redo)"; exit 8; fi
$PY $S/e55_run.py decomp $DEC --dclm-ids $DCLM --dclm-n 100
$PY $S/e55_ic_report.py --d64 --dir $OUT --seeds $SEEDS --tag d64 --out $REP
echo "STAGE B: reverse reading"
for SD in $(echo $SEEDS | tr ',' ' '); do
  $PY $S/e55_run.py rev --cell v3_64_4 --seed $SD --seeds $SD --bg pool --tag d64 --arms $ARMS --apps 128 --out $OUT ${RERUN:+--rerun}
  $PY $S/e55_rev.py --analyze --files $(for AR in $(echo $ARMS | tr ',' ' '); do printf "%s," $OUT/E55_d64_rev_${AR}_s$SD.json; done | sed 's/,$//') --out $OUT/e55_rev_d64_report_s$SD.md
  tail -40 $OUT/e55_rev_d64_report_s$SD.md
done

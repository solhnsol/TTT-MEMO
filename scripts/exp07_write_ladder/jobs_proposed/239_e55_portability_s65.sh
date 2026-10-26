#!/bin/bash
# JOB 239 = E55 PORTABILITY OF THE LABEL-FREE SENTENCE MASK, NEW WORLD (seed 65 of cell v3_16_1; GPU, one job file, sequential stages). Candidate 1 of the seed-61 sentence grid (V6 tok, heads 50%, g3, thr 0.1, trigq>=1) is FIXED (file e55_sentgrid_report_s61_candidates.json, rank 1), no tuning, no new selection, no threshold change. The world of seed 65 is a fresh draw (names, places, facts, frames from the 300-bank, background order) of the same generator; nothing about it is read before the masks are computed.
# Stages: (1) NAT of job 181 type (tag v3a, seed 65) = norm target and base; (2) job 195 type q1b arms SENTPQ (norm reference), SENTPQADAMH (oracle SENTPQ-Adam-0.5N), NATADAM; (3) job 216 type SELMASKS seed 65, rank 1 only: per-chunk recall, precision, F1, hidden share (213-type attention dump is NOT run separately: 216 recomputes the attention of every chunk 1..127 and reports the same statistics on the full stream); (4) job 217 type write SELC1, then job 218 type RSELC1 (random sentences, same hidden-token count per chunk as SELC1); reads out-of-window at a=16,32,64,128 (the a = 32 / 64 / 128 are the reported ones), in-window at 128, DCLM first-chunk NLL at 128, canary at all appearances. Report $OUT/e55_sel_report_s64.md (new name).
# Pre-set rule (same as jobs 233 / 234): R = g(arm)/g(SENTPQ-ADAM-0.5N) at a = 128; WORKS if R >= 0.7 and g(SELC1) - g(RSELC1) >= +0.10; FAILS if R <= 0.3. One seed = first observation.
# EXPECTED RUN TIME (Hans confirms before it runs): ESTIMATE from the measured times of the earlier jobs: stage 1 about 0.4 h (job 181: 51 min for 3 seeds x 2 arms incl. load), stage 2 about 0.9 h (3 arms, about 11 min each plus load), stage 3 about 0.5 h (job 216 s61: 28 min), stage 4 about 1.2 h (jobs 217 / 218 s61: 37 min each for 4 arms; 1 arm plus a model load each). Total about 3 h (2.5-3.5 h); contended server up to 2x. Server GPU is free.
# Needs: nothing downloaded; candidate file in $OUT; DCLM ids file data/dclm_val_subset/val_c0_200x8193.u32le.bin. PINNED INSTALL, NO FETCH (exit 4 / 5 / 6). CPU_ONLY=1 = install check, selftests, background gates, dry run, check-only, exit 0 before any GPU step. NO OVERWRITE of finished (arm, seed) files; exit 8 if the report exists unless RERUN=1.
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
S=$D
SD=${SEED:-65}; [ "$SD" = 65 ] || { echo "this job is for seed 65 only (exit 3)"; exit 3; }
CAND=$OUT/e55_sentgrid_report_s61_candidates.json
[ -f $CAND ] || { echo "candidate file $CAND missing (exit 7)"; exit 7; }
DCLM=data/dclm_val_subset/val_c0_200x8193.u32le.bin
NORM="$OUT/E55_v3a_NAT_s{seed}.saved.json"
Q1B="--cell v3_16_1 --tag q1b --bg pool --out $OUT --seeds $SD --arms SENTPQ,SENTPQADAMH,NATADAM --norm-files $NORM --decomp-apps 16,32,64,128 --in-window --in-apps 128 --canary-all"
SELD="--cell v3_16_1 --tag sel --norm-ref-tag q1b --bg pool --out $OUT --seeds $SD --norm-files $NORM --decomp-apps 16,32,64,128 --in-window --in-apps 128 --canary-all --sel-cands $CAND"
$PY $D4/gen_facts.py --selftest | tail -1
$PY $S/e55_read.py --selftest | tail -1
$PY $S/e55_swap.py | tail -1
$PY $S/e55_metrics.py --selftest | tail -1
$PY $S/e55_ic_report.py --selftest | tail -1
$PY $S/e55_analyze.py --selftest | tail -1
$PY $S/e55_gates.py --selftest | tail -1
$PY $S/e55_sentgrid.py --selftest | tail -1
$PY $S/e55_gates.py --cell v3_16_1 --seed $SD --default-refs --require-refs --out $OUT/e55_gates_port_s$SD.md > $OUT/e55_gates_port_s$SD.log 2>&1 || { tail -30 $OUT/e55_gates_port_s$SD.log; echo "background gates failed (exit 7)"; exit 7; }
tail -4 $OUT/e55_gates_port_s$SD.md
$PY $S/e55_run.py dry --cell v3_16_1 --seed $SD --bg pool --arms NAT | tail -3
$PY $S/e55_run.py selmasks --cell v3_16_1 --seed $SD --bg pool --sel-cands $CAND --sel-ranks 1 --out $OUT --check-only || { echo "selmasks check failed (exit 7)"; exit 7; }
$PY $S/e55_run.py decomp $Q1B --check-only --norm-files "$NORM" 2>&1 | tail -3 || true
[ -f "$DCLM" ] && echo "DCLM ids present: $(stat -c %s $DCLM) bytes" || echo "DCLM ids file absent here (needed on the GPU run)"
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
[ -f "$DCLM" ] || { echo "DCLM ids file $DCLM missing (exit 7)"; exit 7; }
REP=$OUT/e55_sel_report_s$SD.md
if [ -f $REP ] && [ -z "$RERUN" ]; then echo "report $REP exists: not redone (exit 8; RERUN=1 to redo)"; exit 8; fi
echo "STAGE 1: NAT (tag v3a) seed $SD"
[ -f $OUT/E55_v3a_NAT_s$SD.saved.json ] || $PY $S/e55_run.py run --cell v3_16_1 --bg pool --tag v3a --out $OUT --decoy names --save-apps 128 --canary-apps 128 --seeds $SD --arms NAT --plan "$SD=16,32,64,128|128"
echo "STAGE 2: q1b arms SENTPQ, SENTPQADAMH, NATADAM seed $SD"
$PY $S/e55_run.py decomp $Q1B
echo "STAGE 3: selmasks seed $SD rank 1 (candidate fixed)"
$PY $S/e55_run.py selmasks --cell v3_16_1 --seed $SD --bg pool --sel-cands $CAND --sel-ranks 1 --out $OUT
tail -25 $OUT/e55_selmasks_report_s$SD.md
echo "STAGE 4a: SELC1 write"
$PY $S/e55_run.py decomp $SELD --arms SELC1 --dclm-ids $DCLM --dclm-n 100
echo "STAGE 4b: RSELC1 write"
$PY $S/e55_run.py decomp $SELD --arms RSELC1 --dclm-ids $DCLM --dclm-n 100
$PY $S/e55_ic_report.py --uv --dir $OUT --seeds $SD --tag sel --ref-tag q1b --sel-ranks 1 --out $REP
tail -60 $REP

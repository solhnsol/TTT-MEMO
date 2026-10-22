#!/bin/bash
# E55 design 29.3 Q1-B (Adam-direction write on the ONLINE stream with the window, masked and unmasked). World v3_16_1 (the 181 world: 16 facts, 128 chunks = 128 exposures, 300-frame bank, pooled background), seeds 61 62 63, reads out-of-window at a = 16, 32, 64, 128 (W0 subtracted, unexposed question frames), in-window read at a = 128, canary NLL at a = 16/32/64/128 (the report uses 32, 64, 128), DCLM not read.
# Arms (tag q1b): SENTPQ = SENTPQ-STD (ordinary inner SGD step, natural update size; the job-190 convention, re-run so that the pairs match); SENTPQADAM = SENTPQ-ADAM; NATADAM = NAT-ADAM (no masking, Adam direction); SENTPQADAMH = SENTPQ-ADAM-0.5N (half the norm; optional, drop with ARMS=SENTPQ,SENTPQADAM,NATADAM). ADAM = the rule of job 192 ISO-ADAM (bias-corrected m / (sqrt(v) + eps), B1 .9, B2 .999, eps 1e-8, fp32, clipped gradient, state kept over the chunks, direction only: the step SIZE of chunk k = the update norm of chunk k of ITS natural-size SGD arm of the same seed (design 29.4): SENTPQ-ADAM <- SENTPQ (this tag, run first), NAT-ADAM <- NAT of job 181, SENTPQ-ADAM-0.5N <- half of the SENTPQ norm; the report lists the norm ratio of each Adam arm to its SGD arm).
# Reports: $OUT/e55_q1b_report.md (NEW name). The verdict rules are written INTO the report body: SPEED (g(64) >= 0.15 in >= 2 of 3 seeds OR g(64) >= 80% of SENTPQ-STD g(128)), GAIN KEPT (g(128) >= STD - 0.05), DAMAGE relative (canary rise over SENTPQ-STD <= +0.05) and absolute (<= 0.10; reference points W0 AND NAT of job 181 both shown), CLIP share separate, and the recorded item NAT-ADAM / SENTPQ-ADAM >= 30%.
# EXPECTED RUN TIME (Hans confirms before it runs): ESTIMATE ONLY, unmeasured. About 2-2.5 h for the 3 default arms x 3 seeds = 9 runs plus the optional 0.5N arm 3 runs (about 11-12 min per run: 128 chunks + 30 GiB K,V recording about 5-8 min, 4 out-of-window reads + 1 in-window read about 3-5 min, 4 canary reads about 3 min) = about 1.8-2.4 h (2.4-3.2 h with the 0.5N arm), plus model load 7-20 min once (one process). Not run on a GPU: the Adam arms on the 3B model, the reads, the canary.
# Outputs (NEW names): $OUT/E55_q1b_<arm>_s<seed>.* , $OUT/e55_q1b_report.md; exit 8 if the report exists unless RERUN=1. Nothing is overwritten.
# CPU-only dry run (validates the arms through decomp --check-only and the real-tokenizer answer-span search through maskdump): git fetch origin claude/project-thread-rwhdac && git worktree add --detach /tmp/dry_q1b FETCH_HEAD && ln -s "$(pwd)/data" /tmp/dry_q1b/data && cd /tmp/dry_q1b && PY=/abs/path/python CPU_ONLY=1 bash scripts/exp07_write_ladder/jobs_proposed/195_e55_q1b_adam_online.sh   (then git worktree remove --force /tmp/dry_q1b)
# PY must be an ABSOLUTE path to the lab python. PINNED INSTALL, NO FETCH (exit 4 if the commit is absent, exit 5 on a sha256 mismatch). Lab ladder.py must be the patched W1 version (exit 6); lab s3.py must be f261120d0f03, a16e6dd0d4c9 or the new version (exit 6).
# Needs: the NAT file of job 181 per seed (norm target of NAT-ADAM; NAT canary reference at a = 128); SENTPQ must be among the ARMS (it is the norm target of the SENTPQ-ADAM arms and runs first). Does not conflict with job 194 (different tag q1b, own output names, same code files; both use the same pinned commit family; run order free).
# Queue position: any time after 190. NOT APPROVED yet.
set -eo pipefail
PY=$(realpath -s "$PY")   # the runner exports a relative PY (.venv/bin/python): make it absolute BEFORE the cd below, WITHOUT resolving symlinks (uv venv; readlink -f breaks it)
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder
D4=scripts/exp04_what_ttt_retains
COMMIT=7f24c6cb670eff626cc2ada0a47030f0bc4c8207
FILES="$D/e55_run.py $D/e55_core.py $D/e55_gpu.py $D/e55_swap.py $D/e55_read.py $D/e55_analyze.py $D/e55_metrics.py $D/e55_ic_report.py $D/e55_iso.py $D/e55_diag.py $D/s3.py $D4/gen_facts.py $D4/gen_world_ctl.py $D4/frame_bank.py $D/e55_gates.py $D4/frames300.txt"
LADDER_OK="714b08ea3b90"
S3_OLD="f261120d0f03"   # lab s3.py after job 189 (2-D mask version); the pre-189 version a16e6dd0d4c9 is accepted too
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
if [ -z "$NO_INSTALL" ]; then
  GL=$(sha256sum $D/ladder.py | cut -c1-12); [ "$GL" = "$LADDER_OK" ] || { echo "lab ladder.py ($GL) is not the patched W1 version ($LADDER_OK): exit 6"; exit 6; }
  git cat-file -e $COMMIT^{commit} || { echo "commit $COMMIT not in the local clone: git fetch origin claude/project-thread-rwhdac first (exit 4)"; exit 4; }
  S3NEW=$(git show $COMMIT:$D/s3.py | sha256sum | cut -c1-12); S3H=$(sha256sum $D/s3.py | cut -c1-12)
  [ "$S3H" = "$S3_OLD" ] || [ "$S3H" = "a16e6dd0d4c9" ] || [ "$S3H" = "$S3NEW" ] || { echo "lab s3.py ($S3H) is neither the committed old version ($S3_OLD) nor the new one ($S3NEW): not overwritten (exit 6)"; exit 6; }
  git checkout $COMMIT -- $FILES
fi
if [ -z "$SKIP_VERIFY" ]; then
  while read HH F; do
    [ -n "$HH" ] || continue
    G=$(sha256sum $D/$F | cut -c1-12)
    [ "$G" = "$HH" ] || { echo "VERSION MISMATCH $F: have $G, expected $HH (exit 5)"; exit 5; }
  done <<'MANIFEST'
35c0bf2ff28f e55_run.py
73cafbbf4177 e55_core.py
9827e34dcde6 e55_gpu.py
04067fefb630 e55_swap.py
a6d506949071 e55_read.py
4a811650bcd8 e55_analyze.py
30769e241d5e e55_metrics.py
e47bf9c370ca e55_ic_report.py
c4c8a0d4e386 e55_iso.py
a6ceb1237d78 e55_diag.py
8111393db79e s3.py
f14491a9ddf7 ../exp04_what_ttt_retains/gen_facts.py
1cf0caa82553 ../exp04_what_ttt_retains/gen_world_ctl.py
eeaaa9135907 ../exp04_what_ttt_retains/frame_bank.py
ee82bf19b0b3 e55_gates.py
866eb5989467 ../exp04_what_ttt_retains/frames300.txt
MANIFEST
  echo "installed code verified against the manifest"
fi
S=$D
SEEDS="${SEEDS:-61,62,63}"
ARMS="${ARMS:-SENTPQ,SENTPQADAM,NATADAM,SENTPQADAMH}"
NORM="$OUT/E55_v3a_NAT_s{seed}.saved.json"
DEC="--cell v3_16_1 --tag q1b --bg pool --out $OUT --seeds $SEEDS --arms $ARMS --norm-files $NORM --decomp-apps 16,32,64,128 --in-window --in-apps 128 --canary-all"
$PY $D4/gen_facts.py --selftest | tail -1
$PY $S/e55_read.py --selftest | tail -1
$PY $S/e55_swap.py | tail -1
$PY $S/e55_metrics.py --selftest | tail -1
$PY $S/e55_ic_report.py --selftest | tail -1
for SD in $(echo $SEEDS | tr ',' ' '); do
  $PY $S/e55_gates.py --cell v3_16_1 --seed $SD --default-refs --require-refs --out $OUT/e55_gates_q1b_s$SD.md > $OUT/e55_gates_q1b_s$SD.log 2>&1 || { tail -30 $OUT/e55_gates_q1b_s$SD.log; echo "background gates failed for seed $SD (exit 7)"; exit 7; }
  tail -4 $OUT/e55_gates_q1b_s$SD.md
  [ -f $OUT/E55_v3a_NAT_s$SD.saved.json ] || [ -n "$CPU_ONLY" ] || { echo "NAT file of job 181 missing for seed $SD (exit 7)"; exit 7; }
done
$PY $S/e55_run.py dry --cell v3_16_1 --seed 61 --bg pool --arms NAT | tail -3
$PY $S/e55_run.py decomp $DEC --check-only
$PY $S/e55_run.py maskdump --cell v3_16_1 --seed 61 --bg pool > $OUT/e55_maskdump_q1b.json
grep -A6 'answer_span_miss' $OUT/e55_maskdump_q1b.json || true
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
REP=$OUT/e55_q1b_report.md
if [ -f $REP ] && [ -z "$RERUN" ]; then echo "report $REP exists: not redone (exit 8; RERUN=1 to redo)"; exit 8; fi
$PY $S/e55_run.py decomp $DEC
$PY $S/e55_ic_report.py --q1b --dir $OUT --seeds $SEEDS --tag q1b --out $REP
tail -60 $REP

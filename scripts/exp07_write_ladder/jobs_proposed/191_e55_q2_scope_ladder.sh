#!/bin/bash
# E55 design 27.2 (Q2 scope ladder: gain versus damage, how little masking keeps the binding). 181 world (v3_16_1: 16 facts, 128 chunks = 128 exposures, 300-frame bank, pooled background), seeds 61 62 63, window kept, reads at a = 64 and 128. Arms (per-query convention: only queries inside a fact sentence lose keys, background queries see everything):
#   OSPAN   earlier fact sentences: only the answer tokens (name and place, found by tokenising them inside the sentence; the job asserts that every sentence has both) are hidden
#   SENTP25 / SENTP50 / SENTP75   a token-wise random 25 / 50 / 75 % of the tokens of every earlier fact sentence hidden (fixed seed 55027; the hidden sets are nested p25 < p50 < p75 < SENTPQ and the same for every query)
#   SENTPQ  100 % (the job-190 convention control), re-run here so that its damage reads exist; NAT 0 % re-run here for the same reason (reference for the damage reads).
#   Extra arms can be added with ARMS=... (KMASK, SENTO, SENT; and the two-loss mix of design 27.5: MIX25 / MIX50 / MIX75 = loss background + lambda x fact-token loss under the masked (SENTPQ) window + (1-lambda) x fact-token loss under the unmasked window, one forward/backward per window, gradients summed, one clip and one step (about 2x the write cost per chunk), lambda = 0.25 / 0.5 / 0.75; ALT = masked loss on the 1st, 3rd, ... chunk, unmasked on the others, one forward each). Each extra arm costs 3 runs; the report lists only the arms whose files exist (its default arm list in e55_ic_report.py --q2 is the 6 default arms: pass --arms through ARMS).
# Reads per run: out-of-window g at a = 64 and 128 (16 candidates, W0 subtracted, unexposed question frames), IN-WINDOW reads at a = 64 and 128 (real window visible; reference = NAT in-window of job 181 at a = 128), DCLM first-chunk NLL at a = 128 (the E37c method, e37_firstchunk_nll.py, on the first 100 samples of data/dclm_val_subset/val_c0_200x8193.u32le.bin, W0 value per seed in the same process; the report prints the rise = written - W0; native reference +0.043), the old canary (pg11 / pg1342, +0.10 limit) at a = 128, and the per-chunk logs (CE, clip coefficient, hidden keys / pairs per chunk).
# Report $OUT/e55_q2_report.md + e55_q2_curves.png (NEW names): per arm hidden keys % of the window and hidden pairs %, g out-of-window, R = g / g(SENTPQ) (>= 0.7 sufficient, <= 0.3 not sufficient, else partial), in-window g, DCLM rise, mean chunk CE of the last 32 chunks (ALL tokens of the chunk; a background-only CE is not logged by the write path), clip rate; pre-set rules: 'no damage' = DCLM rise <= 0.043 + 0.03 AND in-window g >= NAT in-window - 0.10; method candidate = the arm with the least masking that reaches R >= 0.7 (printed with and without the no-damage condition). Exploratory (3 seeds, 16 facts).
# EXPECTED RUN TIME (Hans confirms before it runs): ESTIMATE ONLY, unmeasured. About 3-4 h for the 6 default arms x 3 seeds = 18 runs: per run 128 chunks + 30 GiB K,V recording about 6-10 min, 4 reads (2 out-of-window + 2 in-window) about 2-4 min, DCLM read 100 samples about 1-2 min (twice per seed for W0 once), plus model load 7-20 min once (design estimate for the 4 new arms alone: 2-2.5 h). Dropping NAT (ARMS=SENTPQ,OSPAN,SENTP25,SENTP50,SENTP75) saves about 30 min; adding MIX25,MIX50,MIX75 adds about 9 runs x 10-15 min = 1.5-2.2 h, ALT about 30-45 min; the 190 files are not reused (different tag q2) because they have no in-window / DCLM reads.
# CPU-only dry run (also validates the arm list through decomp --check-only and the real-tokenizer answer-span search through maskdump on the server's data): git fetch origin claude/project-thread-rwhdac && git worktree add --detach /tmp/dry_q2 FETCH_HEAD && ln -s "$(pwd)/data" /tmp/dry_q2/data && cd /tmp/dry_q2 && PY=/abs/path/python CPU_ONLY=1 bash scripts/exp07_write_ladder/jobs_proposed/191_e55_q2_scope_ladder.sh   (then git worktree remove --force /tmp/dry_q2)
# PY must be an ABSOLUTE path to the lab python. PINNED INSTALL, NO FETCH: code checked out of the local clone at $COMMIT (exit 4 if absent), sha256 first 12 hex (exit 5). The lab ladder.py must be the patched W1 version (exit 6). s3.py: this job needs the one-line 2-D mask extension of s3.py (repo version, installed from $COMMIT); the lab s3.py must be the committed repo version a16e6dd0d4c9 (or already the new one), otherwise exit 6 (nothing is overwritten that has local edits).
# Queue position: after job 190. Reuses the NAT files of job 181 (norm bookkeeping + in-window reference). NOT APPROVED yet.
set -eo pipefail
PY=$(realpath -s "$PY")   # the runner exports a relative PY (.venv/bin/python): make it absolute BEFORE the cd below, WITHOUT resolving symlinks (uv venv; readlink -f breaks it)
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder
D4=scripts/exp04_what_ttt_retains
COMMIT=4a3ad77333430c8b02c05c7071fb69917ec92560
FILES="$D/e55_run.py $D/e55_core.py $D/e55_gpu.py $D/e55_swap.py $D/e55_read.py $D/e55_analyze.py $D/e55_metrics.py $D/e55_ic_report.py $D/e55_iso.py $D/s3.py $D4/gen_facts.py $D4/gen_world_ctl.py $D4/frame_bank.py $D/e55_gates.py $D4/frames300.txt"
LADDER_OK="714b08ea3b90"
S3_OLD="a16e6dd0d4c9"
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
if [ -z "$NO_INSTALL" ]; then
  GL=$(sha256sum $D/ladder.py | cut -c1-12); [ "$GL" = "$LADDER_OK" ] || { echo "lab ladder.py ($GL) is not the patched W1 version ($LADDER_OK): exit 6"; exit 6; }
  git cat-file -e $COMMIT^{commit} || { echo "commit $COMMIT not in the local clone: git fetch origin claude/project-thread-rwhdac first (exit 4)"; exit 4; }
  S3NEW=$(git show $COMMIT:$D/s3.py | sha256sum | cut -c1-12); S3H=$(sha256sum $D/s3.py | cut -c1-12)
  [ "$S3H" = "$S3_OLD" ] || [ "$S3H" = "$S3NEW" ] || { echo "lab s3.py ($S3H) is neither the committed old version ($S3_OLD) nor the new one ($S3NEW): not overwritten (exit 6)"; exit 6; }
  git checkout $COMMIT -- $FILES
fi
if [ -z "$SKIP_VERIFY" ]; then
  while read HH F; do
    [ -n "$HH" ] || continue
    G=$(sha256sum $D/$F | cut -c1-12)
    [ "$G" = "$HH" ] || { echo "VERSION MISMATCH $F: have $G, expected $HH (exit 5)"; exit 5; }
  done <<'MANIFEST'
7e9d4b44a859 e55_run.py
00fbe5110e78 e55_core.py
9827e34dcde6 e55_gpu.py
71eb12a3913f e55_swap.py
a6d506949071 e55_read.py
4a811650bcd8 e55_analyze.py
30769e241d5e e55_metrics.py
ec8bf28b0f53 e55_ic_report.py
5723bbc93aa9 e55_iso.py
f261120d0f03 s3.py
1746ce4c159c ../exp04_what_ttt_retains/gen_facts.py
1cf0caa82553 ../exp04_what_ttt_retains/gen_world_ctl.py
eeaaa9135907 ../exp04_what_ttt_retains/frame_bank.py
ee82bf19b0b3 e55_gates.py
866eb5989467 ../exp04_what_ttt_retains/frames300.txt
MANIFEST
  echo "installed code verified against the manifest"
fi
SEEDS="${SEEDS:-61,62,63}"
ARMS="${ARMS:-NAT,SENTPQ,OSPAN,SENTP25,SENTP50,SENTP75}"
DCLM=data/dclm_val_subset/val_c0_200x8193.u32le.bin
[ -f "$DCLM" ] || [ -n "$CPU_ONLY" ] || { echo "DCLM ids file $DCLM missing (exit 7)"; exit 7; }
S=$D
$PY $D4/gen_facts.py --selftest | tail -1
$PY $S/e55_read.py --selftest | tail -1
$PY $S/e55_swap.py | tail -1
$PY $S/e55_metrics.py --selftest | tail -1
$PY $S/e55_ic_report.py --selftest | tail -1
for SD in $(echo $SEEDS | tr ',' ' '); do
  $PY $S/e55_gates.py --cell v3_16_1 --seed $SD --default-refs --require-refs --out $OUT/e55_gates_q2_s$SD.md > $OUT/e55_gates_q2_s$SD.log 2>&1 || { tail -30 $OUT/e55_gates_q2_s$SD.log; echo "background gates failed for seed $SD (exit 7)"; exit 7; }
  tail -4 $OUT/e55_gates_q2_s$SD.md
  [ -f $OUT/E55_v3a_NAT_s$SD.saved.json ] || [ -n "$CPU_ONLY" ] || { echo "NAT file of job 181 missing for seed $SD (exit 7)"; exit 7; }
done
$PY $S/e55_run.py dry --cell v3_16_1 --seed 61 --bg pool --arms NAT | tail -3
$PY $S/e55_run.py decomp --cell v3_16_1 --tag q2 --bg pool --out $OUT --seeds $SEEDS --arms $ARMS --norm-files "$OUT/E55_v3a_NAT_s{seed}.saved.json" --check-only
[ -f "$DCLM" ] && echo "DCLM ids present: $(stat -c %s $DCLM) bytes" || echo "DCLM ids file absent here (needed on the GPU run)"
$PY $S/e55_run.py maskdump --cell v3_16_1 --seed 61 --bg pool | grep -A12 'answer_span_miss' | head -16
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
REP=$OUT/e55_q2_report.md
if [ -f $REP ] && [ -z "$RERUN" ]; then echo "report $REP exists: not redone (exit 8; RERUN=1 to redo)"; exit 8; fi
$PY $S/e55_run.py decomp --cell v3_16_1 --tag q2 --bg pool --out $OUT --seeds $SEEDS --arms $ARMS --norm-files "$OUT/E55_v3a_NAT_s{seed}.saved.json" --in-window --dclm-ids $DCLM --dclm-n 100
$PY $S/e55_ic_report.py --q2 --dir $OUT --seeds $SEEDS --arms "$ARMS" --out $REP --plot $OUT/e55_q2_curves.png
tail -50 $REP

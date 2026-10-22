#!/bin/bash
# EXPECTED RUN TIME (Hans confirms before it runs): ESTIMATE ONLY, NOT MEASURED; the GPU path has never run on the 3B model. About 2.1-4.0 h on one A100: GPU unit checks 20-30 min, ONE run process for the five arms (model + fp32 reader load 20-35 min, training 5 arm-runs of 16 chunks about 4-6 min each incl. stream recording = 20-30 min, reads: per arm 16 out-of-window + 5 in-window reads + the before-training reads, one read (20 items x 16 candidates, fp32) estimated 30-90 s, so about 12-35 min per arm = 1.0-2.9 h, canary 2 reads per arm about 10-20 min). Unit checks first; if one fails (exit 9) nothing else runs.
# E55 first check, SEED 1 = world seed 55, FULL CURVE (job 180b; design 11d / 11e; Hans 2026-10-22 15:36 KST): the same world as 180a (k = 16 facts, n = 1 sentence per fact per appearance, period p = 1, 16 appearances = 16 chunks, streamed ONCE per arm), five arms in this order:
#   SWD  = fake window A: the attention window of every chunk (lags 1-8) holds the SAME previous chunks with every fact's name and place swapped for a fake of the same token length; the WRITTEN chunk is always the real one. (runs first: its per-chunk update norms are the target of NATM)
#   NAT  = the real stream.
#   SWB  = fake window B: the previous chunks with the fact sentences removed and the gap filled with background text of exactly the same token length (no name / place left).
#   MSK  = the window keys are masked (the current k6p-style masking), written chunk real.
#   NATM = size-matched NAT: the NAT update of every chunk is rescaled to the update norm of SWD of the same chunk (same direction as NAT, size as SWD).
# Reads (frozen weights; 16 candidates = gold + 15 places of other facts, chance mean rank 8.5; question frame = fact id mod 3, never used in the stream; 4 control items with a never-seen name): out-of-window (neutral 8192-token context) after EVERY chunk (16 points) in every arm, in-window (the real chunks before the question; shorter than 8 chunks while appearance < 8) after appearance 1, 2, 4, 8, 16, before-training reads once. A read does not touch the fast weights, optimizer, random state or carried window (CPU unit test: bit-identical). States saved at appearance 4 and 16; canary at W0 and 16. Per arm the pre-clip gradient norm, clip coefficient, update norm, rescale factor and CE of every chunk are in the saved log; the report has the tables (training statistics, W0 rank per question frame with ceiling / floor rates, curve, contrasts incl. SWD-A - NATM).
# Code pin: this job runs on a NEWER commit than 180a (adds arm NATM and the U16 / U17 checks); U17 shows that the NAT / SWD final weights and the read scores of the 180a commit are reproduced bit for bit under this code.
# Unit checks (diagnostic mode): ALL checks run and a result table is printed even if one fails; the write path and the native step are different bf16 graphs, so U1 compares the update with the measured noise floor (cap max(0.03, 3 x floor)), CE <= 1e-3, cos >= 0.999, gnorm within 1% (first server attempt: rel update 0.93%, cos 0.99996; same code on CPU: fp32 1e-6, bf16 1.3%). Only a failed check blocks the run (exit 9).
# Order: CPU part (selftests, dry run: world / stream hashes / read plan / memory estimate) -> (CPU_ONLY stops here) -> GPU unit checks (exit 9 on failure: nothing else runs) -> run -> report.
# Outputs: $OUT/E55_st0_<ARM>_s55.saved.json (+ states/), E55_st0_<ARM>_s55.read.json (+ .read.partial.json while running), e55_report_st0_seed55.md, e55_unit_st0.json, e55_dry_p1.json. NO OVERWRITE: finished files are reused (restart-safe); exit 8 if the report exists unless RERUN=1.
# PY must be an ABSOLUTE path to the lab python. PINNED INSTALL, NO FETCH: code checked out of the local clone at $COMMIT (exit 4 if absent), sha256 first 12 hex (exit 5). The lab ladder.py must be the patched W1 version (exit 6). s3.py is the lab's own (its sha is printed).
# CPU-only dry run: git fetch origin claude/project-thread-rwhdac && git worktree add --detach /tmp/dry_e55s FETCH_HEAD && ln -s "$(pwd)/data" /tmp/dry_e55s/data && cd /tmp/dry_e55s && PY=/abs/path/python CPU_ONLY=1 bash scripts/exp07_write_ladder/jobs_proposed/180b_e55_seed1_five_arms.sh   (then git worktree remove --force /tmp/dry_e55s)
# Queue position: after the running W1 / E52 / E54 jobs (file name 180).
set -eo pipefail
PY=$(realpath -s "$PY")   # the runner exports a relative PY (.venv/bin/python): make it absolute BEFORE the cd below, WITHOUT resolving symlinks (uv venv; readlink -f breaks it)
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
case "$PY" in /*) ;; *) echo "PY must be an absolute path (got '$PY')"; exit 3;; esac
D=scripts/exp07_write_ladder
D4=scripts/exp04_what_ttt_retains
COMMIT=e96016ed23f1767f9c72ee287b855b485d299405
FILES="$D/e55_run.py $D/e55_core.py $D/e55_gpu.py $D/e55_swap.py $D/e55_read.py $D/e55_analyze.py $D/e55_metrics.py $D4/gen_facts.py $D4/gen_world_ctl.py $D4/frame_bank.py"
LADDER_OK="714b08ea3b90"
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
if [ -z "$NO_INSTALL" ]; then
  GL=$(sha256sum $D/ladder.py | cut -c1-12); [ "$GL" = "$LADDER_OK" ] || { echo "lab ladder.py ($GL) is not the patched W1 version ($LADDER_OK): exit 6"; exit 6; }
  git cat-file -e $COMMIT^{commit} || { echo "commit $COMMIT not in the local clone: git fetch origin claude/project-thread-rwhdac first (exit 4)"; exit 4; }
  git checkout $COMMIT -- $FILES
fi
if [ -z "$SKIP_VERIFY" ]; then
  while read HH F; do
    [ -n "$HH" ] || continue
    G=$(sha256sum $D/$F | cut -c1-12)
    [ "$G" = "$HH" ] || { echo "VERSION MISMATCH $F: have $G, expected $HH (exit 5)"; exit 5; }
  done <<'MANIFEST'
27b6807f6529 e55_run.py
210b40d2df40 e55_core.py
9827e34dcde6 e55_gpu.py
b0cd3acd36bb e55_swap.py
a6d506949071 e55_read.py
575149220523 e55_analyze.py
1a28133d518d e55_metrics.py
c34815a3c655 ../exp04_what_ttt_retains/gen_facts.py
ada39e55ccc3 ../exp04_what_ttt_retains/gen_world_ctl.py
eeaaa9135907 ../exp04_what_ttt_retains/frame_bank.py
MANIFEST
  echo "installed code verified against the manifest; lab s3.py sha12 $(sha256sum $D/s3.py | cut -c1-12) (repo copy a16e6dd0d4c9)"
fi
S=$D
$PY $D4/gen_facts.py --selftest | tail -1
$PY $S/e55_read.py --selftest | tail -1
$PY $S/e55_swap.py | tail -1
$PY $S/e55_metrics.py --selftest | tail -1
$PY $S/e55_analyze.py --selftest | tail -1
$PY $S/e55_run.py dry --cell p1 --seed 55 --arms SWD,NAT,SWB,MSK,NATM | tee $OUT/e55_dry_p1.json | tail -3
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
REP=$OUT/e55_report_st0_seed55.md
if [ -f $REP ] && [ -z "$RERUN" ]; then echo "report $REP exists: not redone (exit 8; RERUN=1 to redo)"; exit 8; fi
$PY $S/e55_run.py unit --cell p1 --seed 55 --tag st0 --out $OUT --unit-chunks 12
$PY $S/e55_run.py run --cell p1 --tag st0 --out $OUT --decoy names --save-apps 4,16 --canary-apps 16 --seeds 55 --arms SWD,NAT,SWB,MSK,NATM --plan "55=all|1,2,4,8,16"
f() { echo "$OUT/E55_st0_$1_s55.read.json"; }
SAVED=$(ls $OUT/E55_st0_*_s55.saved.json | tr '\n' ',' | sed 's/,$//')
$PY $S/e55_analyze.py --nat "$(f NAT)" --swd "$(f SWD)" --swb "$(f SWB)" --extra "MSK=$(f MSK);NATM=$(f NATM)" --saved "$SAVED" --a 16 --out $REP
tail -80 $REP

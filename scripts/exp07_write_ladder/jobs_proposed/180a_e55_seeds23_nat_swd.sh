#!/bin/bash
# EXPECTED RUN TIME (Hans confirms before it runs): ESTIMATE ONLY, NOT MEASURED; the GPU path has never run on the 3B model. About 1.4-2.4 h on one A100: GPU unit checks 20-30 min (model load 13-23 min + 12 chunks), ONE run process for both seeds and both arms (model + fp32 reader load 20-35 min, training 4 arm-runs of 16 chunks about 4-6 min each incl. stream recording, reads: per arm and seed 4 out-of-window + 1 in-window read + the before-training reads, one read (20 items x 16 candidates, fp32) estimated 30-90 s, so about 5-10 min per arm and seed, canary 2 reads per arm and seed about 2-4 min). Extra cost per additional seed (both arms): about 15-25 min. Unit checks first; if one fails (exit 9) nothing else runs.
# E55 first check, SEEDS 2 AND 3 (job 180a; design 11d; Hans 2026-10-22 15:36 KST): a new controlled world, k = 16 facts (name, place, occupation), n = 1 sentence per fact per appearance (a different one of 50 frames each time), period p = 1 (all 16 facts in every chunk), 16 appearances = 16 chunks, streamed ONCE per arm and seed; world seeds 56 and 57 (names / places / frames / positions are redrawn per seed; seed 55 = the full-curve job 180b).
#   NAT = the real stream. SWD = fake window A: in the attention window of every chunk (lags 1-8) the previous chunks are replaced by the SAME chunks with every fact's name and place swapped for a fake of the same token length (frames, positions, background identical); the WRITTEN chunk is always the real one.
# Reads (frozen weights; 16 candidates = gold + 15 places of other facts, chance mean rank 8.5; question frame = fact id mod 3, never used in the stream; 4 control items with a never-seen name): out-of-window (neutral 8192-token context) after appearance 2, 4, 8, 16 and in-window (the 8 real chunks before the question) after appearance 16, before-training reads once per seed. A read does not touch the fast weights, optimizer, random state or carried window (CPU unit test: bit-identical). States saved at appearance 4 and 16; canary (held-out text NLL, warning above +0.10 nat) at W0 and 16. Per arm the pre-clip gradient norm, clip coefficient, update norm and CE of every chunk are in the saved log and a table is in the report.
# Unit checks (diagnostic mode): ALL checks run and a result table is printed even if one fails; the write path and the native step are different bf16 graphs, so U1 compares the update with the measured noise floor (cap max(0.03, 3 x floor)), CE <= 1e-3, cos >= 0.999, gnorm within 1% (first server attempt: rel update 0.93%, cos 0.99996; same code on CPU: fp32 1e-6, bf16 1.3%). Only a failed check blocks the run (exit 9).
# Order: CPU part (selftests, dry run: world / stream hashes / read plan / memory estimate) -> (CPU_ONLY stops here) -> GPU unit checks (exit 9 on failure: nothing else runs) -> run -> report.
# Outputs: $OUT/E55_st0_<ARM>_s<SEED>.saved.json (+ states/), E55_st0_<ARM>_s<SEED>.read.json (+ .read.partial.json while running), e55_report_st0_seeds23.md, e55_unit_st0.json, e55_dry_p1.json. NO OVERWRITE: finished files are reused (restart-safe); exit 8 if the report exists unless RERUN=1.
# PY must be an ABSOLUTE path to the lab python. PINNED INSTALL, NO FETCH: code checked out of the local clone at $COMMIT (exit 4 if absent), sha256 first 12 hex (exit 5). The lab ladder.py must be the patched W1 version (exit 6). s3.py is the lab's own (its sha is printed).
# CPU-only dry run: git fetch origin claude/project-thread-rwhdac && git worktree add --detach /tmp/dry_e55s FETCH_HEAD && ln -s "$(pwd)/data" /tmp/dry_e55s/data && cd /tmp/dry_e55s && PY=/abs/path/python CPU_ONLY=1 bash scripts/exp07_write_ladder/jobs_proposed/180a_e55_seeds23_nat_swd.sh   (then git worktree remove --force /tmp/dry_e55s)
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
$PY $S/e55_run.py dry --cell p1 --seed 56 --arms NAT,SWD | tee $OUT/e55_dry_p1.json | tail -3
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
REP=$OUT/e55_report_st0_seeds23.md
if [ -f $REP ] && [ -z "$RERUN" ]; then echo "report $REP exists: not redone (exit 8; RERUN=1 to redo)"; exit 8; fi
$PY $S/e55_run.py unit --cell p1 --seed 56 --tag st0 --out $OUT --unit-chunks 12
$PY $S/e55_run.py run --cell p1 --tag st0 --out $OUT --decoy names --save-apps 4,16 --canary-apps 16 --seeds 56,57 --arms NAT,SWD --plan "56=2,4,8,16|16;57=2,4,8,16|16"
files() { local out=""; for s in "$@"; do [ -f "$OUT/E55_st0_${ARM}_s$s.read.json" ] && out="$out,$OUT/E55_st0_${ARM}_s$s.read.json"; done; echo "${out#,}"; }
ARM=NAT; NATF=$(files 56 57); ARM=SWD; SWDF=$(files 56 57); SWBF=""
SAVED=$(ls $OUT/E55_st0_*_s56.saved.json $OUT/E55_st0_*_s57.saved.json | tr '\n' ',' | sed 's/,$//')
$PY $S/e55_analyze.py --nat "$NATF" --swd "$SWDF" ${SWBF:+--swb "$SWBF"} --saved "$SAVED" --a 16 --out $REP
tail -60 $REP

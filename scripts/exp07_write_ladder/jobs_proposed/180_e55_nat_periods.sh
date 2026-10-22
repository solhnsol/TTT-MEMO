#!/bin/bash
# EXPECTED RUN TIME (Hans confirms before it runs): ESTIMATE ONLY, NOT MEASURED, GPU path never run on the 3B model. About 4.0-5.0 h for the default four cells (CELLS="base p1 p4 p9"), of which: model load/setup 13-23 min per call x 3 calls (unit, run, read per cell are separate python calls: up to 9-13 calls = 2-4 h of overhead if every call reloads; the cell loop below keeps run and read as separate calls), GPU unit checks 20-30 min, per cell training of worlds A and B (one pass, 16 x p chunks each: p=1 about 5 min, p=2 about 11, p=4 about 21, p=9 about 48; chunk time unknown, 10 s assumed) + 2-3 min canary, reading at the last point only (neutral tail + 4 in-window tails, worlds A and B, plus W0) about 25-35 min per cell. Run ONE cell first with CELLS="base" (about 1.2-1.6 h incl. unit checks).
# E55 stage 0, NAT only (no window swap): a new controlled synthetic world (8 pairs' facts per chunk, the same pair repeating every p chunks, p = 2 base / 1 / 4 / 9) is streamed ONCE without any masking in worlds A and B; after pair appearance 16 the 16-candidate place ranking (3 unseen question frames, chance 6.25%) is read with the neutral out-of-window context and with the real stream's last 8 chunks ending at e, e-1, e-3, e-7 (in-window), also at W0. Tests Hans's prediction A (in-window reads high for p = 1, 2, 4 and at out-of-window level for p = 9) and gives the NAT side of the later swap contrast. States are saved at appearance 2, 4, 8, 16 (reading only at 16 here; earlier points can be read later without retraining); canary only at 16 and W0.
# Order: CPU dry runs (world, stream hashes, read plan, memory estimate) -> (CPU_ONLY stops here) -> GPU unit checks (exit 9 on failure: nothing else runs) -> per cell: run, read, report.
# Outputs: $OUT/E55_st0_<cell>_NAT_s55_{A,B}.saved.json, states/E55_st0_<cell>_NAT_s55/{A,B}/, E55_st0_<cell>_NAT_s55.read.json, e55_report_<cell>.md, e55_unit_st0.json, e55_dry_<cell>.json. NO OVERWRITE: a finished cell file is reused (restart-safe); exit 8 if a report exists unless RERUN=1.
# PY must be an ABSOLUTE path to the lab python. PINNED INSTALL, NO FETCH: code checked out of the local clone at $COMMIT (exit 4 if absent), sha256 first 12 hex (exit 5). The lab ladder.py must be the patched W1 version (exit 6). s3.py is the lab's own (its sha is printed).
# CPU-only dry run: git fetch origin claude/project-thread-rwhdac && git worktree add --detach /tmp/dry_e55n FETCH_HEAD && ln -s "$(pwd)/data" /tmp/dry_e55n/data && cd /tmp/dry_e55n && PY=/abs/path/python CPU_ONLY=1 bash scripts/exp07_write_ladder/jobs_proposed/180_e55_nat_periods.sh   (then git worktree remove --force /tmp/dry_e55n)
# Queue position: after the running W1 / E52 / E54 jobs (file name 180).
set -eo pipefail
cd "$(dirname "$0")/../../.." 2>/dev/null || true
case "$PY" in /*) ;; *) echo "PY must be an absolute path (got '$PY')"; exit 3;; esac
D=scripts/exp07_write_ladder
D4=scripts/exp04_what_ttt_retains
COMMIT=42bd3eef1e204788daf2ff10ea22f8e51dda1bab
FILES="$D/e55_run.py $D/e55_core.py $D/e55_gpu.py $D/e55_swap.py $D/e55_read.py $D/e55_analyze.py $D/e49_stats.py $D4/gen_world_ctl.py $D4/frame_pools_hold.py"
LADDER_OK="714b08ea3b90"
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
CELLS="${CELLS:-base p1 p4 p9}"
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
06fb0735f222 e55_run.py
1e9482e70b85 e55_core.py
a9d496eac641 e55_gpu.py
f51de80be2d3 e55_swap.py
909e3fbd4623 e55_read.py
eea35cdb04aa e55_analyze.py
794ae88a9baf e49_stats.py
ada39e55ccc3 ../exp04_what_ttt_retains/gen_world_ctl.py
0aa17d06bfd7 ../exp04_what_ttt_retains/frame_pools_hold.py

MANIFEST
  echo "installed code verified against the manifest; lab s3.py sha12 $(sha256sum $D/s3.py | cut -c1-12) (repo copy a16e6dd0d4c9)"
fi
S=$D
for C in $CELLS; do
  $PY $S/e55_run.py dry --cell $C --seed 55 | tee $OUT/e55_dry_$C.json | tail -3
done
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
$PY $S/e55_run.py unit --cell base --seed 55 --tag st0 --out $OUT --unit-chunks 12
for C in $CELLS; do
  TAG=st0_$C
  if [ -f $OUT/e55_report_$C.md ] && [ -z "$RERUN" ]; then echo "report e55_report_$C.md exists: cell skipped (exit 8 semantics; RERUN=1 to redo)"; continue; fi
  $PY $S/e55_run.py run  --cell $C --seed 55 --arms NAT --tag $TAG --out $OUT --save-apps 2,4,8,16 --canary-apps 16
  $PY $S/e55_run.py read --cell $C --seed 55 --arms NAT --tag $TAG --out $OUT --apps 16 --in-apps 16
  $PY $S/e55_analyze.py --nat $OUT/E55_${TAG}_NAT_s55.read.json --a 16 --out $OUT/e55_report_$C.md
  tail -25 $OUT/e55_report_$C.md
done

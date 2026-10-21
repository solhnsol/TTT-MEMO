#!/bin/bash
# EXPECTED RUN TIME (Hans confirms before it runs): about 40 min, range 30-60 min (ESTIMATE, not measured; the GPU path has never run). One call: model load + Ladder setup 13-23 min (measured per-call overhead of ladder.py / s3.py calls) + building the 900-row candidate readouts (2 readouts) 2-3 min + per state about 20-40 s (30 items x 30 candidates, R1 and alt) plus loading a 1.4 GB state (~10 s): 8 cells x 2 worlds + the untrained model = 17 reads x ~0.7 min = about 12 min. No training, no streaming, no state is written or removed. CPU steps (unit tests, tables) about 1 min.
# E49 (design / core request: does the D gain mean picking the RIGHT place?): re-read SAVED s101 states (default pass 4 = the state D_alt is computed from; E49_EPOCHS=1,2,3,4 reads every pass, cell names <name>_p<pass>, time x4 in the read part, about +35 min) of native fskeep, K3-all, k3h, k3hc, SENT and, when their states still exist, K6' (k6p), oracle p8, oracle lrx2 p4 (oracle p4 s101 states are deleted) plus the untrained model, scoring ALL 30 places of the world as the continuation of each of the 30 L3 prompts (R1 neutral tail and R1-alt SQuAD tail). Output per cell: OUT/e49_cand_s101_<name>.json (raw 30x30 score matrices of world A and B, restart-safe: an existing file is reused) and a table (bound target log-prob, normalized target probability among the 30, rank, top-1/top-3, mass on the swap partner's place, 2-candidate accuracy; 15-pair bootstrap CIs; change vs the reference k3all). Pre-registered rule: a change counts only if the CI excludes 0 and |delta| >= 0.2 nats (log-prob metrics) / 0.10 (accuracy-like). PROPOSED, NOT QUEUED; waits for Hans's approval in the experiment-run thread.
# Independent of the E48 jobs 130-135 (they are not touched); where to put it in the queue: after 130 finished (any time later is fine; it only reads s101 states of old cells). It does not need 120.
# PINNED INSTALL, NO FETCH: checks the code out of the local repository at commit $COMMIT (exit 4 if absent: git fetch origin claude/project-thread-rwhdac first), verifies sha256 (first 12 hex) of the 4 files (exit 5). ladder.py / gen_world.py / readouts are the lab files (not replaced). NO_INSTALL=1 skips the checkout, SKIP_VERIFY=1 the check. The cell list can be overridden with E49_CELLS="name=PREFIX ..." (default below); cells whose saved states are missing are skipped with a message (STRICT=1: exit 7).
# CPU-only dry run (separate directory): git fetch origin claude/project-thread-rwhdac && git worktree add --detach /tmp/dry_e49 FETCH_HEAD && ln -s "$(pwd)/data" /tmp/dry_e49/data && cd /tmp/dry_e49 && PY=python3 S=scripts/exp07_write_ladder CPU_ONLY=1 bash scripts/exp07_write_ladder/jobs_proposed/140_e49_candidates.sh   (afterwards: git worktree remove --force /tmp/dry_e49). The CPU step runs e49_stats selftest and e49_test (stub tokenizer, gen_world of the checkout) and lists the saved states found per cell.
set -e
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder
COMMIT=68d9a8c531df659f5bb1ef6286a778319f484b57
FILES="$D/e49_stats.py $D/e49_candidates.py $D/e49_test.py $D/e48_world_test.py"
E49_CELLS="${E49_CELLS:-k3all=S3_k3all k3h=S3_k3h k3hc=S3_k3hc sent=S3_sent native=L0fs_fskeep k6p=S3_k6p oracle_p8=S3_oracle_p8 oracle_lrx2_p4=S3_oracle_lrx2_p4}"
if [ -z "$NO_INSTALL" ]; then
  git cat-file -e $COMMIT^{commit} || { echo "commit $COMMIT not in the local clone: git fetch origin claude/project-thread-rwhdac first"; exit 4; }
  git checkout $COMMIT -- $FILES
fi
if [ -z "$SKIP_VERIFY" ]; then
  while read HH F; do
    [ -n "$HH" ] || continue
    G=$(sha256sum $D/$F | cut -c1-12)
    [ "$G" = "$HH" ] || { echo "VERSION MISMATCH $F: have $G, expected $HH"; exit 5; }
  done <<'MANIFEST'
794ae88a9baf e49_stats.py
7181363d66bf e49_candidates.py
7578ebd07e3d e49_test.py
70a70838d527 e48_world_test.py

MANIFEST
  echo "installed code verified against the manifest"
fi
$PY $D/e49_stats.py | tail -1
$PY $D/e49_test.py | tail -1
CARGS=""; for C in $E49_CELLS; do
  N=${C%%=*}; P=${C#*=}
  echo "$N ($P): states A $(ls $OUT/states/${P}_s101/A 2>/dev/null | wc -l), B $(ls $OUT/states/${P}_s101/B 2>/dev/null | wc -l), saved log: $([ -f $OUT/${P}_s101_A.saved.json ] && echo yes || echo NO)"
  CARGS="$CARGS --cell $C"
done
[ -n "$CPU_ONLY" ] && exit 0
$PY $D/e49_candidates.py --seed 101 --epochs ${E49_EPOCHS:-4} --w0 --ref k3all $CARGS ${STRICT:+--strict}

#!/bin/bash
# E57 stage 1a-1 (design 11.3 item 5; GPU): plain TTT (NAT, no masks, no Adam, NO reads of any kind) over R = 4, 8, 12, 16 rounds of the frozen fixed-order news round (job 207), one continuous run, canary NLL (gen + tail) and DCLM first-chunk NLL measured at the ends of rounds 4 / 8 / 12 / 16 from the saved weights; R_max = the largest listed R whose canary rise (max of gen / tail against W0) stays <= 0.10 before the first R above the limit. NOT APPROVED yet: Hans must confirm the time (his sentence for 1a-1 is in the design 11.5).
# K,V sharing: the stream is a cyclic repetition of one round (about 103 chunks), so the natural K,V of the window and the prefix outputs are recorded for the first P rounds only, P = ceil(prefix layers x 8192 / round tokens) + 1 = 3 for the 3B model (EXACT: the deepest layer's receptive field is covered), and reused for all later rounds with the query positions of the last recorded round (relative RoPE). The RoPE is computed at the requested positions (rope_fly.py, no 2 x seq_len table; unit U44: equal to the table inside it, finite and relative-position invariant beyond it, so P = 3 = about 316K positions fits although the upstream table holds 262144). CPU-tested on the tiny model: reuse with the exact P equals the plain run (CE 8e-8, update norm 5e-6, U42 / U45), P below the need (the old P = 2) does NOT: update norm 13 % off and 87 % of the weight change differs (U45, tiny model, not 3B). Runner + resume bundle: U43. The attention of the 3B model, the real canary, memory and speed at 1650 chunks are NOT run; the sin / cos values of the table-free RoPE on the GPU may differ from the table version in the last float32 digits.
# 1a-2 continues the SAME run: ROUNDS=20,24,28,32 TAG=1a2 RESUME=<$OUT/e57_1a/E57_1a1_R16.summary.json> (weights state + suffix chain state + canary W0 are in its resume_bundle); only chunks after the first part are run. Note: the first part's stream mask drops the loss of its very last token (0.4 % difference in the next update norm in the tiny test).
# EXPECTED RUN TIME (Hans confirms before it runs): ESTIMATE ONLY, unmeasured. About 2.1-3.4 h for 1a-1: model load 7-20 min; recording the natural K,V of 3 rounds (about 310 chunks, about 75 GiB host) 15-30 min; 16 x 103 = about 1650 updates at 3-5 s per chunk = 1.4-2.3 h (design estimate; E55 measured about 3.5-4.6 s per chunk incl. K,V recording share); 4 canary reads + 4 DCLM reads + 4 state saves 15-30 min. Contended server up to 2x. 1a-2 (R = 20-32): another 1.4-2.8 h + load.
# Needs: the frozen directory of job 207 ($OUT/e57_news/e57_freeze by default; FROZEN=... to override; exit 7 if missing), optional DCLM ids data/dclm_val_subset/val_c0_200x8193.u32le.bin (DCLM reads skipped with a note if absent). Outputs (NEW names) under $OUT/e57_$TAG*: E57_<tag>_R<last>.saved.json / .summary.json / .sts.npz, states/E57_<tag>/, report e57_<tag>_report.md; exit 8 if the report exists unless RERUN=1.
# CPU-only dry run: git fetch origin claude/project-thread-rwhdac && git worktree add --detach /tmp/dry_1a FETCH_HEAD && ln -s "$(pwd)/data" /tmp/dry_1a/data && cd /tmp/dry_1a && PY=/abs/path/python CPU_ONLY=1 bash scripts/exp07_write_ladder/jobs_proposed/208_e57_stage1a_nat_rounds.sh   (then git worktree remove --force /tmp/dry_1a)
# PY must be an ABSOLUTE path to the lab python. PINNED INSTALL, NO FETCH (exit 4 if the commit is absent, exit 5 on a sha256 mismatch). The lab ladder.py must be the patched W1 version (exit 6); the lab s3.py may be any committed version up to the pinned commit (exit 6 otherwise).
set -eo pipefail
PY=$(realpath -s "$PY")   # the runner exports a relative PY (.venv/bin/python): make it absolute BEFORE the cd below, WITHOUT resolving symlinks (uv venv; readlink -f breaks it)
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder
D4=scripts/exp04_what_ttt_retains
COMMIT=904556b2aa276945a720dbec83860d2701b05c07
FILES="$D/e55_run.py $D/e55_core.py $D/e55_gpu.py $D/e55_swap.py $D/e55_read.py $D/e55_analyze.py $D/e55_metrics.py $D/e55_ic_report.py $D/e55_iso.py $D/e55_diag.py $D/e55_votegrid.py $D/e55_uv.py $D/s3.py $D4/gen_facts.py $D4/gen_world_ctl.py $D4/frame_bank.py $D/e55_gates.py $D4/frames300.txt $D/rope_fly.py $D/e57/e57_run1a.py"
LADDER_OK="714b08ea3b90"
S3_OLD="f261120d0f03"   # lab s3.py after job 189 (2-D mask version); the pre-189 version a16e6dd0d4c9 is accepted too
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
if [ -z "$NO_INSTALL" ]; then
  GL=$(sha256sum $D/ladder.py | cut -c1-12); [ "$GL" = "$LADDER_OK" ] || { echo "lab ladder.py ($GL) is not the patched W1 version ($LADDER_OK): exit 6"; exit 6; }
  git cat-file -e $COMMIT^{commit} || { echo "commit $COMMIT not in the local clone: git fetch origin claude/project-thread-rwhdac first (exit 4)"; exit 4; }
  S3NEW=$(git show $COMMIT:$D/s3.py | sha256sum | cut -c1-12); S3H=$(sha256sum $D/s3.py | cut -c1-12)
  S3OK=$( (echo "$S3_OLD"; echo a16e6dd0d4c9; for C in $(git log --format=%H $COMMIT -- $D/s3.py); do git show $C:$D/s3.py | sha256sum | cut -c1-12; done) )   # every s3.py version that was ever committed up to $COMMIT is accepted (a local edit never is)
  echo "$S3OK" | grep -qx "$S3H" || { echo "lab s3.py ($S3H) is not any committed version of s3.py up to $COMMIT (new one: $S3NEW): not overwritten (exit 6)"; exit 6; }
  git checkout $COMMIT -- $FILES
fi
if [ -z "$SKIP_VERIFY" ]; then
  while read HH F; do
    [ -n "$HH" ] || continue
    G=$(sha256sum $D/$F | cut -c1-12)
    [ "$G" = "$HH" ] || { echo "VERSION MISMATCH $F: have $G, expected $HH (exit 5)"; exit 5; }
  done <<'MANIFEST'
7eb02381e4a6 e55_run.py
a5eea0436cd9 e55_core.py
9827e34dcde6 e55_gpu.py
3ee7e3fa1960 e55_swap.py
a6d506949071 e55_read.py
4a811650bcd8 e55_analyze.py
30769e241d5e e55_metrics.py
6374d29cc6db e55_ic_report.py
c4c8a0d4e386 e55_iso.py
725d18d16b65 e55_diag.py
597294ca312b e55_votegrid.py
8aa8585e0153 e55_uv.py
ad73e05f9cb9 s3.py
c839d014a09c ../exp04_what_ttt_retains/gen_facts.py
1cf0caa82553 ../exp04_what_ttt_retains/gen_world_ctl.py
eeaaa9135907 ../exp04_what_ttt_retains/frame_bank.py
eb8bc59bf9a0 e55_gates.py
866eb5989467 ../exp04_what_ttt_retains/frames300.txt
76802d72e2c4 rope_fly.py
bdf270186410 e57/e57_run1a.py
MANIFEST
  echo "installed code verified against the manifest"
fi
S=$D
FROZEN="${FROZEN:-${EXP07_OUT:-experiments/exp07_write_ladder}/e57_news/e57_freeze}"; ROUNDS="${ROUNDS:-4,8,12,16}"; TAG="${TAG:-1a1}"; O1A="$OUT/e57_1a"; mkdir -p "$O1A"
DCLM=data/dclm_val_subset/val_c0_200x8193.u32le.bin
[ -f "$FROZEN/freeze_manifest.json" ] || { echo "frozen directory $FROZEN missing: job 207 must have finished (exit 7)"; exit 7; }
$PY $S/e55_swap.py | tail -1
$PY $S/e57/e57_run1a.py --frozen "$FROZEN" --rounds "$ROUNDS" --check-only
[ -f "$DCLM" ] && echo "DCLM ids present: $(stat -c %s $DCLM) bytes" || echo "DCLM ids file absent here (DCLM reads skipped)"
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
REP="$O1A/e57_${TAG}_report.md"
if [ -f "$REP" ] && [ -z "$RERUN" ]; then echo "report $REP exists: not redone (exit 8; RERUN=1 to redo)"; exit 8; fi
EXTRA=""; [ -f "$DCLM" ] && EXTRA="--dclm-ids $DCLM --dclm-n 100"; [ -n "$RESUME" ] && EXTRA="$EXTRA --resume $RESUME"; [ -n "$RERUN" ] && EXTRA="$EXTRA --rerun"
$PY $S/e57/e57_run1a.py --frozen "$FROZEN" --rounds "$ROUNDS" --tag "$TAG" --out "$O1A" $EXTRA
tail -40 "$REP"

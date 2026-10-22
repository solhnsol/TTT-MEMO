#!/bin/bash
# E55 design 22.1 stage 1, TEST: fast-weight fine-tuning upper-bound probe on the TEST world (seed 61, cell ic_u64: 16 facts, 64 chunks, unique frames, pooled background) at the learning rate chosen from the dev sweep (env LR, e.g. LR=3e-5; exit 3 if unset), 16 epochs, reads after epochs 1, 2, 4, 8, 16.
# Arms: FT-NAT (the 64-chunk stream of the unique-frames world, real window), FT-MASK (same, earlier fact sentences hidden from attention in the training forward only: exactly the SENT mask of job 183, mask dump hash printed), FT-ISO (the isolated rows of stage 0, all 16 x 64 sentences per epoch, reshuffled each epoch). ONLY the fast (TTT inner-update) weights are trained, the rest is frozen; AdamW (weight decay 0, betas 0.9/0.999, eps 1e-8, global-norm clip 1.0), fp32 master weights + fp32 Adam state, ONE step per 1024-token chunk (FT-ISO: per packed chunk), loss = whole-chunk next-token loss (the TTT inner-step loss and write path; U27 checks the gradient pipeline against write_step). Epoch = one pass over the 64-chunk stream; weights, master and Adam moments carry on; the carried SWA window is reset at the start of every epoch, so the first 8 chunks of every epoch have a short window (recorded in the files). Reads after the listed epochs, frozen weights, no extra training: out-of-window, in-window (last 8 chunks) and POSITIVE CONTROL (the same 16-candidate question in the frame of the fact's latest training sentence = memorization ceiling, not used for the verdict). Also logged per epoch: mean CE and the CE of the place tokens (fact tokens; stream arms: chunks 7, 15, ..., 63 before their update; FT-ISO: all rows), and the peak device memory.
# EXPECTED RUN TIME (Hans confirms before it runs): ESTIMATE ONLY, NOT MEASURED. About 2.4 h (range 1.7-3.3 h): CPU part 5-10 min; model + fp32 reader load 20-35 min once; training: FT-NAT and FT-MASK 16 epochs x 64 steps at about 1.3-2.5 s = 22-43 min each, FT-ISO 16 epochs x about 26 packed chunks = 8-17 min; reads: 3 arms x 5 read points x (out + in-window + positive control, about 1.5-4.5 min) = 22-68 min.
# Needs the dev sweep (186) done and the learning rate chosen. Outputs: $OUT/E55_ftt_<ARM>_lr<lr>_s61.saved.json (+ _ep<e>.read.json / .pc.read.json), e55_gates_ftt_s61.md, e55_maskdump_ftt_s61.json, e55_ftt_report.md. NO OVERWRITE (restart-safe); exit 8 if the report exists unless RERUN=1.
# CPU-only dry run: git fetch origin claude/project-thread-rwhdac && git worktree add --detach /tmp/dry_ftt FETCH_HEAD && ln -s "$(pwd)/data" /tmp/dry_ftt/data && cd /tmp/dry_ftt && PY=/abs/path/python LR=3e-5 CPU_ONLY=1 bash scripts/exp07_write_ladder/jobs_proposed/187_e55_ft_test_seed61.sh   (then git worktree remove --force /tmp/dry_ftt)
# PY must be an ABSOLUTE path to the lab python. PINNED INSTALL, NO FETCH: code checked out of the local clone at $COMMIT (exit 4 if absent), sha256 first 12 hex (exit 5). The lab ladder.py must be the patched W1 version (exit 6). s3.py: this job needs the one-line 2-D mask extension of s3.py (repo version, installed from $COMMIT); the lab s3.py must be the committed repo version a16e6dd0d4c9 (or already the new one), otherwise exit 6 (nothing is overwritten that has local edits).
# Queue position: any time (independent of 181-185).
set -eo pipefail
PY=$(realpath -s "$PY")   # the runner exports a relative PY (.venv/bin/python): make it absolute BEFORE the cd below, WITHOUT resolving symlinks (uv venv; readlink -f breaks it)
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder
D4=scripts/exp04_what_ttt_retains
COMMIT=0a418144808cc65aae7351abb2ea215b695fe31c
FILES="$D/e55_run.py $D/e55_core.py $D/e55_gpu.py $D/e55_swap.py $D/e55_read.py $D/e55_analyze.py $D/e55_metrics.py $D/e55_ic_report.py $D/e55_iso.py $D/e55_ft.py $D/s3.py $D4/gen_facts.py $D4/gen_world_ctl.py $D4/frame_bank.py $D/e55_gates.py $D4/frames300.txt"
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
d8d5d4cee125 e55_run.py
ca9f3280f5d6 e55_core.py
9827e34dcde6 e55_gpu.py
5179c07fb43e e55_swap.py
a6d506949071 e55_read.py
4a811650bcd8 e55_analyze.py
30769e241d5e e55_metrics.py
6c09d81fd207 e55_ic_report.py
91a471c61d21 e55_iso.py
f261120d0f03 s3.py
1746ce4c159c ../exp04_what_ttt_retains/gen_facts.py
1cf0caa82553 ../exp04_what_ttt_retains/gen_world_ctl.py
eeaaa9135907 ../exp04_what_ttt_retains/frame_bank.py
ee82bf19b0b3 e55_gates.py
866eb5989467 ../exp04_what_ttt_retains/frames300.txt
MANIFEST
  echo "installed code verified against the manifest"
fi
[ -n "$LR" ] || { echo "set LR (learning rate chosen from the dev sweep), e.g. LR=3e-5 (exit 3)"; exit 3; }
LRS="$LR"; EPOCHS=16; READ="1,2,4,8,16"
S=$D
$PY $D4/gen_facts.py --selftest | tail -1
$PY $S/e55_read.py --selftest | tail -1
$PY $S/e55_swap.py | tail -1
$PY $S/e55_metrics.py --selftest | tail -1
$PY $S/e55_ic_report.py --selftest | tail -1
$PY $S/e55_gates.py --cell ic_u64 --seed 61 --default-refs --require-refs --out $OUT/e55_gates_ftt_s61.md > $OUT/e55_gates_ftt_s61.log 2>&1 || { tail -30 $OUT/e55_gates_ftt_s61.log; echo "background gates failed (exit 7)"; exit 7; }
tail -4 $OUT/e55_gates_ftt_s61.md
$PY $S/e55_run.py maskdump --cell ic_u64 --seed 61 --bg pool --arms NAT > $OUT/e55_maskdump_ftt_s61.json || { echo "mask dump failed (exit 7)"; exit 7; }
grep -E "sent_mask_sha12_all_chunks|ALL_OK" $OUT/e55_maskdump_ftt_s61.json
$PY $S/e55_run.py dry --cell ic_u64 --seed 61 --bg pool --arms NAT | tail -3
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
REP=$OUT/e55_ftt_report.md
if [ -f $REP ] && [ -z "$RERUN" ]; then echo "report $REP exists: not redone (exit 8; RERUN=1 to redo)"; exit 8; fi
$PY $S/e55_run.py ft --cell ic_u64 --tag ftt --bg pool --out $OUT --seeds 61 --lrs "$LRS" --epochs $EPOCHS --read-epochs "$READ" --ft-arms FT-NAT,FT-MASK,FT-ISO
$PY $S/e55_ic_report.py --ft --dir $OUT --tag ftt --seed 61 --lrs "$LRS" --epochs $EPOCHS --out $REP
tail -80 $REP

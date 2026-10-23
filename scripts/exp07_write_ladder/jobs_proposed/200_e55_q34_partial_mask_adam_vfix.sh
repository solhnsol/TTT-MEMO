#!/bin/bash
# E55 34 bundle (one job, one model load; design 34.2). World v3_16_1 (the 181 world), seeds 61 62 63, window kept, write = the Adam direction of the clipped write gradient (rule of job 192 ISO-ADAM) with a per-chunk step size = FACTOR x the update norm of SENTPQ-STD of the same seed (design 29.4; the SENTPQ files of tag q1b, jobs 195/196, are the norm targets and are reused, NOT re-run). Reads: out-of-window at a = 16, 32, 64, 128; in-window at a = 128; DCLM first-chunk NLL at a = 128 (100 samples, E37c method); canary NLL at a = 16, 32, 64, 128 (report: 32, 64, 128).
# (2') partial token masks on SENTPQ-ADAM-0.5N: SENTP25ADAMH / SENTP50ADAMH / SENTP75ADAMH = a random 25 / 50 / 75 % of the tokens of every earlier fact sentence of the window hidden from the fact-sentence queries (per-query convention as SENTPQ; fixed nested draws, seed 55027), Adam direction, factor 0.5. Reference: SENTPQ-ADAM-0.5N and SENTPQ-STD of tag q1b. Question: does a partly hidden sentence give back the gain (threshold type or dose-response)?
# (3') Adam damage reduction on SENTPQ: SENTPQADAM75 = A1 ADAM-0.75N (factor 0.75); SENTPQVFIX10 = A2 ADAM-vfix-1.0N and SENTPQVFIX05 = A3 ADAM-vfix-0.5N (the second moment v is updated only during the first 8 chunks and then frozen: a diagonal preconditioner + momentum; factors 1.0 and 0.5). Pre-set success rule (written into the report): g(128) >= 0.326 AND g(64) >= 0.20 AND canary rise at a=128 <= SENTPQ-STD + 0.05; ties go to the lower canary rise.
# Report $OUT/e55_q34_report.md (NEW name). Arms of the default list = 6 arms x 3 seeds = 18 runs; REFS=1 adds SENTPQ and SENTPQ-ADAM-0.5N under the new tag (6 more runs, needed only if the reference arms must also carry DCLM NLL reads: the q1b files have no DCLM read).
# EXPECTED RUN TIME (Hans confirms before it runs): ESTIMATE ONLY, unmeasured. About 3.3-5.3 h: CPU part (selftests, 3 background gate checks, dry run, maskdump) 5-10 min; model load 7-20 min once (one process); 18 runs of about 10-16 min each (128 chunks + 30 GiB K,V recording 6-10 min, 4 out-of-window reads + 1 in-window read + 1 DCLM read + 4 canary reads 3-6 min) = 3.0-4.8 h. REFS=1 adds about 1.0-1.6 h. Design estimate 3.7-4 h. Not run on a GPU: the Adam / vfix arms with the random-token masks on the 3B model, the DCLM reads in this combination.
# Outputs (NEW names): $OUT/E55_q34_<arm>_s<seed>.* , $OUT/e55_q34_report.md; the arms whose saved.json exists are skipped (RERUN=1 to redo them); exit 8 if the report exists unless RERUN=1.
# CPU-only dry run (validates the arms through decomp --check-only, the real-tokenizer answer-span search through maskdump): git fetch origin claude/project-thread-rwhdac && git worktree add --detach /tmp/dry_q34 FETCH_HEAD && ln -s "$(pwd)/data" /tmp/dry_q34/data && cd /tmp/dry_q34 && PY=/abs/path/python CPU_ONLY=1 bash scripts/exp07_write_ladder/jobs_proposed/200_e55_q34_partial_mask_adam_vfix.sh   (then git worktree remove --force /tmp/dry_q34)
# PY must be an ABSOLUTE path to the lab python. PINNED INSTALL, NO FETCH (exit 4 if the commit is absent, exit 5 on a sha256 mismatch). The lab ladder.py must be the patched W1 version (exit 6); the lab s3.py may be any committed version of s3.py up to the pinned commit (exit 6 otherwise). The Adam moments stay on the device during the reads (E55_ADAM_OFFLOAD=1 would move them to the host); the [mem] line is printed after every arm.
# Needs: NAT file of job 181 (tag v3a) per seed; SENTPQ, SENTPQADAM, SENTPQADAMH files of tag q1b (jobs 195/196) per seed (norm targets and report references). Independent of the other proposed jobs. NOT APPROVED yet.
set -eo pipefail
PY=$(realpath -s "$PY")   # the runner exports a relative PY (.venv/bin/python): make it absolute BEFORE the cd below, WITHOUT resolving symlinks (uv venv; readlink -f breaks it)
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder
D4=scripts/exp04_what_ttt_retains
COMMIT=cda001ccb322af9d7cfd0764c58cdbd2603a6be5
FILES="$D/e55_run.py $D/e55_core.py $D/e55_gpu.py $D/e55_swap.py $D/e55_read.py $D/e55_analyze.py $D/e55_metrics.py $D/e55_ic_report.py $D/e55_iso.py $D/e55_diag.py $D/e55_votegrid.py $D/s3.py $D4/gen_facts.py $D4/gen_world_ctl.py $D4/frame_bank.py $D/e55_gates.py $D4/frames300.txt"
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
4b047d412a7b e55_run.py
c1db5049740e e55_core.py
9827e34dcde6 e55_gpu.py
e60430681325 e55_swap.py
a6d506949071 e55_read.py
4a811650bcd8 e55_analyze.py
30769e241d5e e55_metrics.py
7a402b38ea4b e55_ic_report.py
c4c8a0d4e386 e55_iso.py
725d18d16b65 e55_diag.py
597294ca312b e55_votegrid.py
558d7974f051 s3.py
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
ARMS="${ARMS:-SENTP25ADAMH,SENTP50ADAMH,SENTP75ADAMH,SENTPQADAM75,SENTPQVFIX10,SENTPQVFIX05}"
[ -n "$REFS" ] && ARMS="SENTPQ,SENTPQADAMH,$ARMS"
NORM="$OUT/E55_v3a_NAT_s{seed}.saved.json"
DCLM=data/dclm_val_subset/val_c0_200x8193.u32le.bin
DEC="--cell v3_16_1 --tag q34 --norm-ref-tag q1b --bg pool --out $OUT --seeds $SEEDS --arms $ARMS --norm-files $NORM --decomp-apps 16,32,64,128 --in-window --in-apps 128 --canary-all"
$PY $D4/gen_facts.py --selftest | tail -1
$PY $S/e55_read.py --selftest | tail -1
$PY $S/e55_swap.py | tail -1
$PY $S/e55_metrics.py --selftest | tail -1
$PY $S/e55_ic_report.py --selftest | tail -1
for SD in $(echo $SEEDS | tr ',' ' '); do
  $PY $S/e55_gates.py --cell v3_16_1 --seed $SD --default-refs --require-refs --out $OUT/e55_gates_q34_s$SD.md > $OUT/e55_gates_q34_s$SD.log 2>&1 || { tail -30 $OUT/e55_gates_q34_s$SD.log; echo "background gates failed for seed $SD (exit 7)"; exit 7; }
  tail -4 $OUT/e55_gates_q34_s$SD.md
  for F in v3a_NAT q1b_SENTPQ q1b_SENTPQADAM q1b_SENTPQADAMH; do [ -f $OUT/E55_${F}_s$SD.saved.json ] || [ -n "$CPU_ONLY" ] || { echo "file E55_${F}_s$SD.saved.json missing (exit 7)"; exit 7; }; done
done
$PY $S/e55_run.py dry --cell v3_16_1 --seed 61 --bg pool --arms NAT | tail -3
$PY $S/e55_run.py decomp $DEC --check-only
[ -f "$DCLM" ] && echo "DCLM ids present: $(stat -c %s $DCLM) bytes" || echo "DCLM ids file absent here (needed on the GPU run)"
$PY $S/e55_run.py maskdump --cell v3_16_1 --seed 61 --bg pool > $OUT/e55_maskdump_q34.json
grep -A6 'answer_span_miss' $OUT/e55_maskdump_q34.json || true
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
[ -f "$DCLM" ] || { echo "DCLM ids file $DCLM missing (exit 7)"; exit 7; }
REP=$OUT/e55_q34_report.md
if [ -f $REP ] && [ -z "$RERUN" ]; then echo "report $REP exists: not redone (exit 8; RERUN=1 to redo)"; exit 8; fi
$PY $S/e55_run.py decomp $DEC --dclm-ids $DCLM --dclm-n 100
$PY $S/e55_ic_report.py --q34 --dir $OUT --seeds $SEEDS --tag q34 --ref-tag q1b --out $REP
tail -60 $REP

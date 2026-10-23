#!/bin/bash
# E55 34 part (3') ONLY, split out of job 200 (the (2') random partial token masks of job 200 are done, e55_q34_report.md). One job, one model load. World v3_16_1 (the 181 world), seeds 61 62 63, window kept, write = the Adam direction of the clipped write gradient (rule of job 192 ISO-ADAM), per-chunk step = FACTOR x the update norm of SENTPQ-STD of the same seed (tag q1b files = norm targets and references, reused, NOT re-run). Reads: out-of-window at a = 16, 32, 64, 128; in-window at a = 128; DCLM first-chunk NLL at a = 128 (100 samples); canary NLL at a = 16, 32, 64, 128.
# Arms (Adam damage reduction on SENTPQ): SENTPQADAM75 = A1 ADAM-0.75N (factor 0.75); SENTPQVFIX10 = A2 ADAM-vfix-1.0N and SENTPQVFIX05 = A3 ADAM-vfix-0.5N (second moment v updated only during the first 8 chunks, then frozen; factors 1.0 and 0.5). Pre-set success rule (written into the report): g(128) >= 0.326 AND g(64) >= 0.20 AND canary rise at a=128 <= SENTPQ-STD + 0.05; ties go to the lower canary rise. References: SENTPQ-STD, SENTPQ-ADAM, SENTPQ-ADAM-0.5N of tag q1b (no DCLM read there; REFS=1 adds SENTPQ and SENTPQ-ADAM-0.5N under the new tag).
# Same code path as job 200 for these arms (cda001c plus the later commits: jax.clear_caches() after every arm = the cross-arm memory growth fix, [mem] lines with live-array counts, UV files). Tag q34c, report $OUT/e55_q34c_report.md (NEW names; the P2 rows of that report are empty by design, they are in e55_q34_report.md). 3 arms x 3 seeds = 9 runs.
# EXPECTED RUN TIME (Hans confirms before it runs): ESTIMATE ONLY, unmeasured. About 1.7-2.8 h: CPU part (selftests, 3 background gate checks, dry run, maskdump) 5-10 min; model load 7-20 min once; 9 runs of about 10-16 min each (128 chunks + K,V recording 6-10 min, reads 3-6 min) = 1.5-2.4 h. Use the observed per-run time of job 200a (the lab has it) to tighten this; contended server up to 2x. REFS=1 adds about 0.7-1.1 h.
# Outputs (NEW names): $OUT/E55_q34c_<arm>_s<seed>.* , $OUT/e55_q34c_report.md; arms whose saved.json exists are skipped (RERUN=1 to redo them); exit 8 if the report exists unless RERUN=1.
# CPU-only dry run: git fetch origin claude/project-thread-rwhdac && git worktree add --detach /tmp/dry_q34c FETCH_HEAD && ln -s "$(pwd)/data" /tmp/dry_q34c/data && cd /tmp/dry_q34c && PY=/abs/path/python CPU_ONLY=1 bash scripts/exp07_write_ladder/jobs_proposed/204_e55_q34c_adam_damage_reduction.sh   (then git worktree remove --force /tmp/dry_q34c)
# PY must be an ABSOLUTE path to the lab python. PINNED INSTALL, NO FETCH (exit 4 / 5 / 6 as in job 200). Needs: NAT file of job 181 (tag v3a) and SENTPQ, SENTPQADAM, SENTPQADAMH files of tag q1b per seed. NOT APPROVED yet.
set -eo pipefail
PY=$(realpath -s "$PY")   # the runner exports a relative PY (.venv/bin/python): make it absolute BEFORE the cd below, WITHOUT resolving symlinks (uv venv; readlink -f breaks it)
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder
D4=scripts/exp04_what_ttt_retains
COMMIT=8efab155def3694da77230ced2a52077e61151e6
FILES="$D/e55_run.py $D/e55_core.py $D/e55_gpu.py $D/e55_swap.py $D/e55_read.py $D/e55_analyze.py $D/e55_metrics.py $D/e55_ic_report.py $D/e55_iso.py $D/e55_diag.py $D/e55_votegrid.py $D/e55_uv.py $D/s3.py $D4/gen_facts.py $D4/gen_world_ctl.py $D4/frame_bank.py $D/e55_gates.py $D4/frames300.txt"
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
0709c64388bc e55_run.py
859ad9f01249 e55_core.py
9827e34dcde6 e55_gpu.py
3ee7e3fa1960 e55_swap.py
a6d506949071 e55_read.py
4a811650bcd8 e55_analyze.py
30769e241d5e e55_metrics.py
d3537bb81a11 e55_ic_report.py
c4c8a0d4e386 e55_iso.py
725d18d16b65 e55_diag.py
597294ca312b e55_votegrid.py
8aa8585e0153 e55_uv.py
ad73e05f9cb9 s3.py
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
ARMS="${ARMS:-SENTPQADAM75,SENTPQVFIX10,SENTPQVFIX05}"
[ -n "$REFS" ] && ARMS="SENTPQ,SENTPQADAMH,$ARMS"
NORM="$OUT/E55_v3a_NAT_s{seed}.saved.json"
DCLM=data/dclm_val_subset/val_c0_200x8193.u32le.bin
DEC="--cell v3_16_1 --tag q34c --norm-ref-tag q1b --bg pool --out $OUT --seeds $SEEDS --arms $ARMS --norm-files $NORM --decomp-apps 16,32,64,128 --in-window --in-apps 128 --canary-all"
$PY $D4/gen_facts.py --selftest | tail -1
$PY $S/e55_read.py --selftest | tail -1
$PY $S/e55_swap.py | tail -1
$PY $S/e55_metrics.py --selftest | tail -1
$PY $S/e55_ic_report.py --selftest | tail -1
for SD in $(echo $SEEDS | tr ',' ' '); do
  $PY $S/e55_gates.py --cell v3_16_1 --seed $SD --default-refs --require-refs --out $OUT/e55_gates_q34c_s$SD.md > $OUT/e55_gates_q34c_s$SD.log 2>&1 || { tail -30 $OUT/e55_gates_q34c_s$SD.log; echo "background gates failed for seed $SD (exit 7)"; exit 7; }
  tail -4 $OUT/e55_gates_q34c_s$SD.md
  for F in v3a_NAT q1b_SENTPQ q1b_SENTPQADAM q1b_SENTPQADAMH; do [ -f $OUT/E55_${F}_s$SD.saved.json ] || [ -n "$CPU_ONLY" ] || { echo "file E55_${F}_s$SD.saved.json missing (exit 7)"; exit 7; }; done
done
$PY $S/e55_run.py dry --cell v3_16_1 --seed 61 --bg pool --arms NAT | tail -3
$PY $S/e55_run.py decomp $DEC --check-only
[ -f "$DCLM" ] && echo "DCLM ids present: $(stat -c %s $DCLM) bytes" || echo "DCLM ids file absent here (needed on the GPU run)"
$PY $S/e55_run.py maskdump --cell v3_16_1 --seed 61 --bg pool > $OUT/e55_maskdump_q34c.json
grep -A6 'answer_span_miss' $OUT/e55_maskdump_q34c.json || true
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
[ -f "$DCLM" ] || { echo "DCLM ids file $DCLM missing (exit 7)"; exit 7; }
REP=$OUT/e55_q34c_report.md
if [ -f $REP ] && [ -z "$RERUN" ]; then echo "report $REP exists: not redone (exit 8; RERUN=1 to redo)"; exit 8; fi
$PY $S/e55_run.py decomp $DEC --dclm-ids $DCLM --dclm-n 100
$PY $S/e55_ic_report.py --q34 --dir $OUT --seeds $SEEDS --tag q34c --ref-tag q1b --out $REP
tail -60 $REP

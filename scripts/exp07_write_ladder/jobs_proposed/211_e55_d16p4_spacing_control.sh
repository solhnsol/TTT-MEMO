#!/bin/bash
# E55 world v3_16_4 (E57 section 12 recommendation D, SPACING CONTROL of job 210; GPU): the baseline fact count 16 (as v3_16_1) at period p = 4 (every fact in every 4th chunk, 4 fact sentences per chunk = a quarter of the density), 128 exposures per fact = 512 chunks = 512 updates, frame bank of 300, background from the extended public-domain book pool (job 209, 24 books). Seed 61 by default (SEEDS=61,62,63 for all). Window kept. Arms: NAT and SENTPQ (earlier fact sentences masked from attention, SGD step). Reads: out-of-window at a = 16, 32, 64, 128 exposures, in-window at a = 128, canary NLL at a = 16, 32, 64, 128, DCLM first-chunk NLL at a = 128 (100 samples). Tag d16p4, report $OUT/e55_d16p4_report.md. Read with job 210 and v3_16_1: fact count effect = (v3_16_4 - v3_16_4); gap + density + update-count effect (they change together at fixed 128 exposures) = (v3_16_4 - v3_16_1). Meant to run ONLY if the effect of job 210 (seed 61) is smaller than at k = 16.
# RoPE: positions go up to 524288 > the upstream table (2 x seq_len = 262144), so this job REQUIRES the table-free RoPE (rope_fly.py, installed by e55_core; unit U44: inside the table equal to the table version, beyond it finite and relative-position invariant). Not measured on the GPU: values may differ from the table version in the last float32 digits (sin / cos of the same arguments inside a different fusion); the k = 16 results were produced with the table.
# EXPECTED RUN TIME (Hans confirms before it runs; longer than 1 h, the coordinator is told before it is queued): ESTIMATE ONLY, unmeasured. Per arm and seed: recording the natural K,V of the 512-chunk stream 24-40 min (about 120 GiB host RAM) + 512 updates about 11 min + reads and canary 4-10 min (fewer groups to read than job 210) = about 40-60 min. Default 2 arms x 1 seed = about 1.4-2.1 h plus model load 7-20 min and the CPU part 5-15 min; 2 arms x 3 seeds about 4.3-6.3 h.
# Needs: job 209 done (20 more books in data/narrative_gutenberg, hashes pinned in e55_bgbooks_sha.json once pinned; before that, a first CPU_ONLY run checks the pool capacity without hash pins). The pool gate (e) of e55_gates.py (need / available >= 1.5) decides whether 24 books are enough: exit 7 if it fails.
# Outputs (NEW names): $OUT/E55_d16p4_<arm>_s<seed>.*, $OUT/e55_d16p4_report.md, $OUT/e55_gates_d16p4_s<seed>.md; exit 8 if the report exists unless RERUN=1.
# CPU-only dry run (lab server, real tokenizer and the real pool path): git fetch origin claude/project-thread-rwhdac && git worktree add --detach /tmp/dry_d16p4 FETCH_HEAD && ln -s "$(pwd)/data" /tmp/dry_d16p4/data && cd /tmp/dry_d16p4 && PY=/abs/path/python CPU_ONLY=1 bash scripts/exp07_write_ladder/jobs_proposed/211_e55_d16p4_spacing_control.sh   (then git worktree remove --force /tmp/dry_d16p4)
# PY must be an ABSOLUTE path to the lab python. PINNED INSTALL, NO FETCH (exit 4 / 5 / 6 as in job 200). NOT APPROVED yet.
set -eo pipefail
PY=$(realpath -s "$PY")   # the runner exports a relative PY (.venv/bin/python): make it absolute BEFORE the cd below, WITHOUT resolving symlinks (uv venv; readlink -f breaks it)
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder
D4=scripts/exp04_what_ttt_retains
COMMIT=f93db3c60c64c1804cd76958307d4b1240adc1dc
FILES="$D/e55_run.py $D/e55_core.py $D/e55_gpu.py $D/e55_swap.py $D/e55_read.py $D/e55_analyze.py $D/e55_metrics.py $D/e55_ic_report.py $D/e55_iso.py $D/e55_diag.py $D/e55_votegrid.py $D/e55_uv.py $D/s3.py $D4/gen_facts.py $D4/gen_world_ctl.py $D4/frame_bank.py $D/e55_gates.py $D4/frames300.txt $D/rope_fly.py $D/e55_bgbooks.py $D/e55_bgbooks_sha.json"
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
7a22509c96c5 e55_ic_report.py
c4c8a0d4e386 e55_iso.py
725d18d16b65 e55_diag.py
597294ca312b e55_votegrid.py
8aa8585e0153 e55_uv.py
ad73e05f9cb9 s3.py
b867b6f680f3 ../exp04_what_ttt_retains/gen_facts.py
1cf0caa82553 ../exp04_what_ttt_retains/gen_world_ctl.py
eeaaa9135907 ../exp04_what_ttt_retains/frame_bank.py
eb8bc59bf9a0 e55_gates.py
866eb5989467 ../exp04_what_ttt_retains/frames300.txt
76802d72e2c4 rope_fly.py
3d8657814bbb e55_bgbooks.py
75ff3cee6437 e55_bgbooks_sha.json
MANIFEST
  echo "installed code verified against the manifest"
fi
S=$D
SEEDS="${SEEDS:-61}"
ARMS="${ARMS:-NAT,SENTPQ}"
DCLM=data/dclm_val_subset/val_c0_200x8193.u32le.bin
$PY $S/e55_bgbooks.py verify --dest data/narrative_gutenberg --pins $S/e55_bgbooks_sha.json || { echo "the extra background books are missing or differ from the pins: run job 209 first (exit 7)"; exit 7; }
DEC="--cell v3_16_4 --tag d16p4 --bg pool --out $OUT --seeds $SEEDS --arms $ARMS --norm-files $OUT/E55_d16p4_NAT_s{seed}.saved.json --decomp-apps 16,32,64,128 --in-window --in-apps 128 --canary-all"
$PY $D4/gen_facts.py --selftest | tail -1
$PY $S/e55_read.py --selftest | tail -1
$PY $S/e55_swap.py | tail -1
$PY $S/e55_metrics.py --selftest | tail -1
$PY $S/e55_ic_report.py --selftest | tail -1
$PY $S/e55_bgbooks.py --selftest
for SD in $(echo $SEEDS | tr ',' ' '); do
  $PY $S/e55_gates.py --cell v3_16_4 --seed $SD --default-refs --require-refs --out $OUT/e55_gates_d16p4_s$SD.md > $OUT/e55_gates_d16p4_s$SD.log 2>&1 || { tail -30 $OUT/e55_gates_d16p4_s$SD.log; echo "background gates failed for seed $SD (pool too small or overlap; exit 7)"; exit 7; }
  tail -4 $OUT/e55_gates_d16p4_s$SD.md
done
$PY $S/e55_run.py dry --cell v3_16_4 --seed 61 --bg pool --arms NAT | tail -3
$PY $S/e55_run.py decomp $DEC --check-only
[ -f "$DCLM" ] && echo "DCLM ids present: $(stat -c %s $DCLM) bytes" || echo "DCLM ids file absent here (needed on the GPU run)"
$PY $S/e55_run.py maskdump --cell v3_16_4 --seed 61 --bg pool > $OUT/e55_maskdump_d16p4.json
grep -A6 'answer_span_miss' $OUT/e55_maskdump_d16p4.json || true
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
[ -f "$DCLM" ] || { echo "DCLM ids file $DCLM missing (exit 7)"; exit 7; }
REP=$OUT/e55_d16p4_report.md
if [ -f $REP ] && [ -z "$RERUN" ]; then echo "report $REP exists: not redone (exit 8; RERUN=1 to redo)"; exit 8; fi
$PY $S/e55_run.py decomp $DEC --dclm-ids $DCLM --dclm-n 100
$PY $S/e55_ic_report.py --d64 --dir $OUT --seeds $SEEDS --tag d16p4 --cell v3_16_4 --arms $ARMS --out $REP
tail -60 $REP

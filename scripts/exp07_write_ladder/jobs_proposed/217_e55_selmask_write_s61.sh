#!/bin/bash
# JOB 217_e55_selmask_write_s61 = E55 32.11 WRITE EXPERIMENT WITH LABEL-FREE CANDIDATE MASKS (SELC1, SELC2, SELC3, SELC5 (ARMS env to change)). World v3_16_1 (the 181 world), SEEDS (default 61), window kept, write = Adam direction with factor 0.5 and per-chunk step = 0.5 x the update norm of SENTPQ-STD of the same seed (tag q1b files are the norm targets, reused, NOT re-run), exactly as UVA / RSA of job 201 / 212. Reads: out-of-window at a = 16, 32, 64, 128; in-window at a = 128; DCLM first-chunk NLL at a = 128; canary reads at all appearances.
# Arms the four candidate masks: SELC<r> = the stored label-free mask of the sentence-grid candidate of rank r (job 216 writes it; the 4 candidates and their configurations are in the candidate file of the seed-61 213 report and are copied into every arm's saved.json under extra.sel), RSELC<r> = random window sentences with the same hidden-token count per chunk as SELC<r> (they read n_masked_keys of the SELC<r> arm of the same seed). Tag sel, reports under NEW names.
# Pre-set rule (unchanged from design 32.8, in the report): R = g(arm)/g(SENTPQ-ADAM-0.5N) at a = 128 (reference = tag q1b); R >= 0.7 and g(SELC) - g(RSELC) >= +0.10 -> label-free sentence masking works; R <= 0.3 -> does not. One seed = first observation; seeds 62 / 63 (SEEDS=62 etc., after job 216 with SEED=62 / 63) replicate with the candidates FIXED on seed 61. Canary and DCLM damage are reported per arm (canary rise a = 128 reference: Adam-0.5N SENTPQ).
# Report $OUT/e55_sel_report.md (NEW name).
# EXPECTED RUN TIME (Hans confirms before it runs): ESTIMATE ONLY, unmeasured. About 1.2-2.2 h for 4 arms: CPU part 5-10 min; model load 7-20 min; per arm about 15-25 min (128 chunks with the Adam write, no attention pass at write time, plus K,V recording 6-10 min and reads 3-6 min, as the RSA estimate of job 212); DCLM reads included. Contended server up to 2x (then up to about 4 h).
# Needs: job 216 of the same seed ($OUT/e55_selmasks_s<seed>.npz, exit 7 if missing), $OUT/e55_sentgrid_report_s61_candidates.json, NAT file of job 181 (tag v3a) and SENTPQ, SENTPQADAMH, NATADAM files of tag q1b per seed. Not run on a GPU: everything (the CPU test U48 runs the arms SELC1 and RSELC1 through the Adam write on the tiny model).
# CPU-only dry run: git fetch origin claude/project-thread-rwhdac && git worktree add --detach /tmp/dry_sw FETCH_HEAD && ln -s "$(pwd)/data" /tmp/dry_sw/data && cd /tmp/dry_sw && PY=/abs/path/python CPU_ONLY=1 bash scripts/exp07_write_ladder/jobs_proposed/217_e55_selmask_write_s61.sh   (then git worktree remove --force /tmp/dry_sw)
# PY must be an ABSOLUTE path to the lab python. PINNED INSTALL, NO FETCH (exit 4 / 5 / 6 as in job 213). Memory as jobs 201 / 212 (the arms have no attention pass at write time). NOT APPROVED yet.
set -eo pipefail
PY=$(realpath -s "$PY")   # the runner exports a relative PY (.venv/bin/python): make it absolute BEFORE the cd below, WITHOUT resolving symlinks (uv venv; readlink -f breaks it)
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder
D4=scripts/exp04_what_ttt_retains
COMMIT=34841f101cf49d46f94fff4fa7fb73bbed125ef1
FILES="$D/e55_run.py $D/e55_core.py $D/e55_gpu.py $D/e55_swap.py $D/e55_read.py $D/e55_analyze.py $D/e55_metrics.py $D/e55_ic_report.py $D/e55_iso.py $D/e55_diag.py $D/e55_votegrid.py $D/e55_uv.py $D/s3.py $D/rope_fly.py $D/e55_sentdump.py $D/e55_sentdump_labels.py $D/e55_sentsel.py $D/e55_sentgrid.py $D/e55_selmask.py $D/e55_gates.py $D4/gen_facts.py $D4/gen_world_ctl.py $D4/frame_bank.py $D4/frames300.txt"
LADDER_OK="714b08ea3b90"
S3_OLD="f261120d0f03"   # lab s3.py after job 189 (2-D mask version); the pre-189 version a16e6dd0d4c9 is accepted too
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
if [ -z "$NO_INSTALL" ]; then
  GL=$(sha256sum $D/ladder.py | cut -c1-12); [ "$GL" = "$LADDER_OK" ] || { echo "lab ladder.py ($GL) is not the patched W1 version ($LADDER_OK): exit 6"; exit 6; }
  git cat-file -e $COMMIT^{commit} || { echo "commit $COMMIT not in the local clone: git fetch origin claude/project-thread-rwhdac first (exit 4)"; exit 4; }
  S3NEW=$(git show $COMMIT:$D/s3.py | sha256sum | cut -c1-12); S3H=$(sha256sum $D/s3.py | cut -c1-12)
  S3OK=$( (echo "$S3_OLD"; echo a16e6dd0d4c9; for C in $(git log --format=%H --all -- $D/s3.py); do git show $C:$D/s3.py | sha256sum | cut -c1-12; done) )   # every s3.py version ever committed in the local clone (all branches, so a later job that left a newer committed s3.py on the lab does not block this one) is accepted (a local edit never is)
  echo "$S3OK" | grep -qx "$S3H" || { echo "lab s3.py ($S3H) is not any committed version of s3.py up to $COMMIT (new one: $S3NEW): not overwritten (exit 6)"; exit 6; }
  git checkout $COMMIT -- $FILES
fi
if [ -z "$SKIP_VERIFY" ]; then
  while read HH F; do
    [ -n "$HH" ] || continue
    G=$(sha256sum $D/$F | cut -c1-12)
    [ "$G" = "$HH" ] || { echo "VERSION MISMATCH $F: have $G, expected $HH (exit 5)"; exit 5; }
  done <<'MANIFEST'
bf121294e3ce e55_run.py
f05caa0a4811 e55_core.py
9827e34dcde6 e55_gpu.py
b82e7ab635c2 e55_swap.py
a6d506949071 e55_read.py
4a811650bcd8 e55_analyze.py
30769e241d5e e55_metrics.py
e8c32d9ffa65 e55_ic_report.py
c4c8a0d4e386 e55_iso.py
725d18d16b65 e55_diag.py
597294ca312b e55_votegrid.py
6d9e9b62cecd e55_uv.py
a123948130e4 s3.py
76802d72e2c4 rope_fly.py
16bf1b4d7808 e55_sentdump.py
df41b7d13739 e55_sentdump_labels.py
3728d3954e71 e55_sentsel.py
24b0b4970d22 e55_sentgrid.py
21324f7fcdb7 e55_selmask.py
eb8bc59bf9a0 e55_gates.py
b867b6f680f3 ../exp04_what_ttt_retains/gen_facts.py
1cf0caa82553 ../exp04_what_ttt_retains/gen_world_ctl.py
eeaaa9135907 ../exp04_what_ttt_retains/frame_bank.py
866eb5989467 ../exp04_what_ttt_retains/frames300.txt
MANIFEST
  echo "installed code verified against the manifest"
fi
S=$D
SEEDS="${SEEDS:-61}"
ARMS="${ARMS:-SELC1,SELC2,SELC3,SELC5}"
NORM="$OUT/E55_v3a_NAT_s{seed}.saved.json"
CAND=$OUT/e55_sentgrid_report_s61_candidates.json
DCLM=data/dclm_val_subset/val_c0_200x8193.u32le.bin
RANKS=$(echo $ARMS | tr ',' '\n' | sed 's/^RSELC//; s/^SELC//' | sort -un | paste -sd,)
DEC="--cell v3_16_1 --tag sel --norm-ref-tag q1b --bg pool --out $OUT --seeds $SEEDS --arms $ARMS --norm-files $NORM --decomp-apps 16,32,64,128 --in-window --in-apps 128 --canary-all --sel-cands $CAND"
[ -f $CAND ] || { echo "candidate file $CAND missing: job 213 with SEED=61 must have run (exit 7)"; [ -n "$CPU_ONLY" ] || exit 7; }
$PY $D4/gen_facts.py --selftest | tail -1
$PY $S/e55_read.py --selftest | tail -1
$PY $S/e55_swap.py | tail -1
$PY $S/e55_metrics.py --selftest | tail -1
$PY $S/e55_ic_report.py --selftest | tail -1
for SD in $(echo $SEEDS | tr ',' ' '); do
  $PY $S/e55_gates.py --cell v3_16_1 --seed $SD --default-refs --require-refs --out $OUT/e55_gates_sel_s$SD.md > $OUT/e55_gates_sel_s$SD.log 2>&1 || { tail -30 $OUT/e55_gates_sel_s$SD.log; echo "background gates failed for seed $SD (exit 7)"; exit 7; }
  tail -4 $OUT/e55_gates_sel_s$SD.md
  for F in v3a_NAT q1b_SENTPQ q1b_SENTPQADAMH q1b_NATADAM; do [ -f $OUT/E55_${F}_s$SD.saved.json ] || [ -n "$CPU_ONLY" ] || { echo "file E55_${F}_s$SD.saved.json missing (exit 7)"; exit 7; }; done
  [ -f $OUT/e55_selmasks_s$SD.npz ] || { echo "mask file e55_selmasks_s$SD.npz missing: job 216 with SEED=$SD first"; [ -n "$CPU_ONLY" ] || exit 7; }
  for A in $(echo $ARMS | tr ',' ' '); do case "$A" in RSELC*) U=SELC${A#RSELC}; [ -f $OUT/E55_sel_${U}_s$SD.saved.json ] || { echo "E55_sel_${U}_s$SD.saved.json (matched arm of $A) missing here"; [ -n "$CPU_ONLY" ] || exit 7; };; esac; done
done
$PY $S/e55_run.py dry --cell v3_16_1 --seed 61 --bg pool --arms NAT | tail -3
$PY $S/e55_run.py decomp $DEC --check-only
[ -f "$DCLM" ] && echo "DCLM ids present: $(stat -c %s $DCLM) bytes" || echo "DCLM ids file absent here (needed on the GPU run)"
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
[ -f "$DCLM" ] || { echo "DCLM ids file $DCLM missing (exit 7)"; exit 7; }
REP=$OUT/e55_sel_report.md
if [ -f $REP ] && [ -z "$RERUN" ]; then echo "report $REP exists: not redone (exit 8; RERUN=1 to redo)"; exit 8; fi
$PY $S/e55_run.py decomp $DEC --dclm-ids $DCLM --dclm-n 100
$PY $S/e55_ic_report.py --uv --dir $OUT --seeds $SEEDS --tag sel --ref-tag q1b --sel-ranks $RANKS --out $REP
tail -60 $REP

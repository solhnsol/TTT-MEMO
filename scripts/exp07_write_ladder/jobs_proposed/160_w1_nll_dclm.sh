#!/bin/bash
# EXPECTED RUN TIME (Hans confirms before it runs): about 25-30 min, with slack about 40 min (ESTIMATE, not measured on W1). Basis: job 96 (E44 SENT, model load ~5 min + W0 + 2 states + W0 repeat) was estimated 14 min, E43 3 cells (6 states) 13.8 min; here 10 states + W0 + W0 repeat = 12 reads of 200 samples (about 1.1-2.3 min per read) + ~5 min load = about 18-33 min.
# W1 DCLM first-chunk damage read (E37c method, e37_firstchunk_nll.py; NO training, NO weights written): per-sample NLL of the first 1024-token chunk of the 200 DCLM validation samples (data/dclm_val_subset/val_c0_200x8193.u32le.bin) under W0 (always included) and the saved PASS-4 states of the five W1 cells, worlds A and B: native (L0fs_w1main), oracle (S3_w1_oracle), k3all (S3_w1_k3all), sent (S3_w1_sent), k6p (S3_w1_k6p). W0 is read twice (repeat: max |diff| must be 0).
# Output (tag e37w1, no overlap with old files e37_firstchunk_{e37,e43..e44,...}.*): $OUT/e37_firstchunk_e37w1.{json,csv} (labels w1_native_p4_A ... w1_k6p_p4_B, W0, W0_repeat) and $OUT/e37_w1_nll_table.md (the CPU table: mean increase over W0 per cell for world A, B and their per-sample average, 95% bootstrap CI over samples, samples 145-147 [pg11 overlap] excluded as in E37 section 5, all-sample column given). The job stops (exit 8) if the json already exists.
# Queue: after 160_w1_e49_candidates_5cells.sh (name order: 160_w1_e49... < 160_w1_nll... < 161_...). It does not need that job's outputs. Could be run in the same call as the candidate readout only with new code (two different scripts each load the model); kept as a separate file: one more model load (~5 min).
# PINNED INSTALL, NO FETCH: checks e37_firstchunk_nll.py and e37_nll_table.py out of the local repository at commit $COMMIT (exit 4 if absent: git fetch origin claude/project-thread-rwhdac first) and verifies sha256 (first 12 hex; exit 5). NO_INSTALL=1 skips the checkout, SKIP_VERIFY=1 the check. The lab ladder.py / gen_world.py are used as they are.
# CPU-only dry run: git fetch origin claude/project-thread-rwhdac && git worktree add --detach /tmp/dry_w1n FETCH_HEAD && ln -s "$(pwd)/data" /tmp/dry_w1n/data && cd /tmp/dry_w1n && PY=python3 S=scripts/exp07_write_ladder CPU_ONLY=1 bash scripts/exp07_write_ladder/jobs_proposed/160_w1_nll_dclm.sh   (then git worktree remove --force /tmp/dry_w1n). It lists the ten state files (exit 3 if one is missing) and the output names.
set -e
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder
COMMIT=a08d2661a29abaa4cc430403a4e8dae9886fb0ca
FILES="$D/e37_firstchunk_nll.py $D/e37_nll_table.py"
if [ -z "$NO_INSTALL" ]; then
  git cat-file -e $COMMIT^{commit} || { echo "commit $COMMIT not in the local clone: git fetch origin claude/project-thread-rwhdac first"; exit 4; }
  git checkout $COMMIT -- $FILES
fi
if [ -z "$SKIP_VERIFY" ]; then
  [ "$(sha256sum $D/e37_firstchunk_nll.py | cut -c1-12)" = "e167e94bc935" ] || { echo "VERSION MISMATCH e37_firstchunk_nll.py"; exit 5; }
  [ "$(sha256sum $D/e37_nll_table.py | cut -c1-12)" = "5d2cd4cae5e9" ] || { echo "VERSION MISMATCH e37_nll_table.py"; exit 5; }
  echo "installed code verified"
fi
export E48_FRAMES=pair
IDS=data/dclm_val_subset/val_c0_200x8193.u32le.bin
[ -f "$IDS" ] || { echo "$IDS missing"; exit 3; }
ARGS=""
for C in w1_native=L0fs_w1main w1_oracle=S3_w1_oracle w1_k3all=S3_w1_k3all w1_sent=S3_w1_sent w1_k6p=S3_w1_k6p; do
  N=${C%%=*}; P=${C#*=}
  for W in A B; do
    F=$OUT/states/${P}_s101/$W/state_004.npz; [ -f "$F" ] || { echo "missing $F (states not kept?)"; exit 3; }
    ARGS="$ARGS --state ${N}_p4_${W}=$F"
  done
done
echo "output: $OUT/e37_firstchunk_e37w1.json ($([ -f $OUT/e37_firstchunk_e37w1.json ] && echo EXISTS || echo new)), table $OUT/e37_w1_nll_table.md"
$PY $D/e37_nll_table.py --selftest | tail -1
[ -n "$CPU_ONLY" ] && exit 0
[ -f $OUT/e37_firstchunk_e37w1.json ] && { echo "$OUT/e37_firstchunk_e37w1.json already exists: not overwritten (exit 8)"; exit 8; }
$PY $D/e37_firstchunk_nll.py --ids $IDS --seed 101 --tag e37w1 $ARGS
$PY $D/e37_nll_table.py --json $OUT/e37_firstchunk_e37w1.json --out $OUT/e37_w1_nll_table.md

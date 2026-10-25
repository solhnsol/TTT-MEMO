#!/bin/bash
# E55 REVERSE READING (place -> name) on the weights after a = 128 appearances of world v3_64_4, seed 61 (E55 section 37; GPU). TWO STAGES, because job 210 saved NO weights (its saved.json have reads = {} and no state files): STAGE A re-trains the three arms NAT, SENTPQ and SENTPQADAMH (= SENTPQ with Adam, 0.5 N) of job 210 on the same stream and saves the inner weights ONLY at a = 128 (e55_run.py decomp --save-state-app 128 --tag d64s, no inline reads, no canary, no DCLM; the Adam arm takes its norm target from the 210 file E55_d64_SENTPQ_s62.saved.json); arms whose new saved.json already holds the state file are skipped. The final weight hash of every re-trained arm is compared with the final_hash of job 210 (a difference is printed, not fatal: the code of the arms changed since 210, deterministic ops are on). STAGE B reads (read only, one model load): 'rev' = a place is given, the model completes a person name, 16 candidates (gold name + the 15 names of the other candidate facts of the forward item of the same fact: 7 of the same chunk group, 8 of other groups), >= 3 reverse frames (the first 3 of 6 that pass the CPU checks), and 'fwd' = the forward items of the same facts as paired reference F, at W0 and at the three states. CPU checks first (E55_d64s_rev_check_s62.md): frame overlaps (5-gram with training and question frames, word-order permutation, 13-gram with stream and reference texts), facts that share a place (excluded), candidate-name token lengths and first-token collisions, same / other group split. Analysis (CPU): per-fact ranks and u minus W0, strata same / other group names, bootstrap over facts, rule of section 37.3 (R / F, NAT as control).
# NOT bundled with the section 36.3 new-format reads (their frames do not exist in code yet): the reader (e55_run.py rev) takes any item family, so they can share one model load later.
# Needs: job 209 books (pool path) and the 210 files E55_d64_NAT_s62.saved.json and E55_d64_SENTPQ_s62.saved.json (update norms of the norm targets; exit 7 if missing). Outputs (NEW names, tag d64s): $OUT/E55_d64s_<arm>_s62.saved.json, $OUT/states/E55_d64s_<arm>_s62/state_dc*_511.npz, $OUT/E55_d64s_rev_check_s62.md / .json, $OUT/E55_d64s_rev_<arm>_s62.json, $OUT/e55_rev_d64s_report_s62.md. Exit 8 if the report exists unless RERUN=1.
# EXPECTED RUN TIME (Hans confirms before it runs; this is LONGER than 1 h, the coordinator is told before it is queued): ESTIMATE ONLY, unmeasured. STAGE A per arm: recording the natural K,V of the 512-chunk stream 24-40 min (job 210 header estimate, 4 x the 6-10 min of the 128-chunk runs, about 120 GiB host RAM) + 512 updates at about 1.3 s = 11 min + one state save = about 35-55 min; 3 arms = about 1.8-2.8 h (NAT alone about 0.6-0.9 h of that; set ARMS=SENTPQ,SENTPQADAMH to skip the control, then the NAT rule of section 37.3 cannot be applied). STAGE B about 20-30 min (load 13-23 min, 8 reads about 6 min). CPU checks 1-5 min. Total about 2.2-3.4 h. Server load (~80) up to 2x.
# CPU-only dry run: PY=/abs/path/python CPU_ONLY=1 bash scripts/exp07_write_ladder/jobs_proposed/224_e55_reverse_read_s62.sh   (unit tests, decomp check-only for stage A, CPU frame / place / name checks with the real tokenizer and pool, then exits). PY must be an ABSOLUTE path to the lab python. PINNED INSTALL, NO FETCH (exit 4 / 5 / 6 as in job 210). NOT APPROVED yet.
set -eo pipefail
PY=$(realpath -s "$PY")   # the runner exports a relative PY (.venv/bin/python): make it absolute BEFORE the cd below, WITHOUT resolving symlinks (uv venv; readlink -f breaks it)
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder
D4=scripts/exp04_what_ttt_retains
COMMIT=c12ba3cef078d304e42e177219637cb0059b261f
FILES="$D/e55_run.py $D/e55_core.py $D/e55_gpu.py $D/e55_swap.py $D/e55_read.py $D/e55_analyze.py $D/e55_metrics.py $D/e55_ic_report.py $D/e55_iso.py $D/e55_diag.py $D/e55_votegrid.py $D/e55_uv.py $D/s3.py $D4/gen_facts.py $D4/gen_world_ctl.py $D4/frame_bank.py $D/e55_gates.py $D4/frames300.txt $D/rope_fly.py $D/e55_bgbooks.py $D/e55_bgbooks_sha.json $D/e55_rev.py"
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
46f1065a3762 e55_run.py
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
b867b6f680f3 ../exp04_what_ttt_retains/gen_facts.py
1cf0caa82553 ../exp04_what_ttt_retains/gen_world_ctl.py
eeaaa9135907 ../exp04_what_ttt_retains/frame_bank.py
eb8bc59bf9a0 e55_gates.py
866eb5989467 ../exp04_what_ttt_retains/frames300.txt
76802d72e2c4 rope_fly.py
3d8657814bbb e55_bgbooks.py
75ff3cee6437 e55_bgbooks_sha.json
2215511cee2a e55_rev.py
MANIFEST
  echo "installed code verified against the manifest"
fi
S=$D
ARMS="${ARMS:-NAT,SENTPQ,SENTPQADAMH}"
$PY $S/e55_bgbooks.py verify --dest data/narrative_gutenberg --pins $S/e55_bgbooks_sha.json || { echo "the extra background books are missing or differ from the pins: run job 209 first (exit 7)"; exit 7; }
SAVE="--cell v3_64_4 --tag d64s --bg pool --out $OUT --seeds 62 --arms $ARMS --norm-files $OUT/E55_d64_NAT_s{seed}.saved.json --norm-ref-tag d64 --decomp-apps 16,32,64,128 --save-state-app 128 --no-canary"
has_state() { $PY -c "import json,sys; r=json.load(open(sys.argv[1])); sys.exit(0 if any('_file' in v for v in r.get('reads',{}).values() if isinstance(v,dict)) else 1)" "$1" 2>/dev/null; }
$PY $D4/gen_facts.py --selftest | tail -1
$PY $S/e55_read.py --selftest | tail -1
$PY $S/e55_rev.py --selftest | tail -1
$PY $S/e55_bgbooks.py --selftest
$PY $S/e55_run.py decomp $SAVE --check-only | tail -3
$PY $S/e55_run.py revcheck --cell v3_64_4 --seed 62 --bg pool --tag d64s --arms NAT --out $OUT
grep -E "^- (items|frames|candidate)|^\| [0-9] " $OUT/E55_d64s_rev_check_s62.md
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
for F in NAT SENTPQ; do [ -f $OUT/E55_d64_${F}_s62.saved.json ] || { echo "$OUT/E55_d64_${F}_s62.saved.json missing (job 210 norm targets; exit 7)"; exit 7; }; done
REP=$OUT/e55_rev_d64s_report_s62.md
if [ -f $REP ] && [ -z "$RERUN" ]; then echo "report $REP exists: not redone (exit 8; RERUN=1 to redo)"; exit 8; fi
NEED=""
for AR in $(echo $ARMS | tr ',' ' '); do has_state $OUT/E55_d64s_${AR}_s62.saved.json || NEED="$NEED,$AR"; done
NEED=${NEED#,}
if [ -n "$NEED" ]; then
  for AR in $(echo $NEED | tr ',' ' '); do [ -f $OUT/E55_d64s_${AR}_s62.saved.json ] && { echo "E55_d64s_${AR}_s62.saved.json exists WITHOUT a state file: move it away first (exit 7)"; exit 7; }; done
  echo "STAGE A: training and saving the weights at a = 128 for: $NEED"
  $PY $S/e55_run.py decomp ${SAVE/--arms $ARMS/--arms $NEED}
fi
for AR in $(echo $ARMS | tr ',' ' '); do
  has_state $OUT/E55_d64s_${AR}_s62.saved.json || { echo "no state file in $OUT/E55_d64s_${AR}_s62.saved.json after stage A (exit 7)"; exit 7; }
  O=$($PY -c "import json,sys; print(json.load(open(sys.argv[1])).get('final_hash'))" $OUT/E55_d64_${AR}_s62.saved.json 2>/dev/null); N=$($PY -c "import json,sys; print(json.load(open(sys.argv[1])).get('final_hash'))" $OUT/E55_d64s_${AR}_s62.saved.json)
  echo "final weight hash $AR: job 210 $O, this job $N $([ "$O" = "$N" ] && echo '(identical to job 210)' || echo '(DIFFERENT from job 210: not a bit-identical re-run)')"
done
echo "STAGE B: reverse reading"
$PY $S/e55_run.py rev --cell v3_64_4 --seed 62 --seeds 62 --bg pool --tag d64s --arms $ARMS --apps 128 --out $OUT ${RERUN:+--rerun}
$PY $S/e55_rev.py --analyze --files $(for AR in $(echo $ARMS | tr ',' ' '); do printf "%s," $OUT/E55_d64s_rev_${AR}_s62.json; done | sed 's/,$//') --out $REP
tail -40 $REP

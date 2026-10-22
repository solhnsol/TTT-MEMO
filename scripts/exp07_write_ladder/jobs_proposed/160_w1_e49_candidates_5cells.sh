#!/bin/bash
# EXPECTED RUN TIME (Hans confirms before it runs): about 70 min, with slack up to 90 min (ESTIMATE, not measured on W1). Range 35-90 min: one call = model load + Ladder setup 13-23 min (measured per-call overhead) + building the 900-row candidate readouts 2-3 min + 5 cells x pass 4 x 2 worlds + the untrained model = 11 state reads at 1-4 min each (20-40 s scoring + loading a 1.4 GB state; the 41 min of job 141 do not say whether they covered pass 4 only or passes 1-4, hence the wide range) + CPU steps ~1 min. NO training, NO streaming, no state is written or removed.
# E49 candidate readout on the W1 world (Hans: "W1 30후보 판독부터"; design request): the same readout definition as job 141 (all 30 places scored as the continuation of each L3 item, packed fp32 R1 neutral tail and R1-alt SQuAD tail; abs log-prob, normalized probability, rank, top-1/top-3, mass on the swap partner, 2-candidate accuracy, 15-pair bootstrap CI table, changes vs the reference cell), on the 5 SAVED W1 states at pass 4: native (L0fs_w1main), oracle (S3_w1_oracle), k6p (S3_w1_k6p), k3all (S3_w1_k3all), sent (S3_w1_sent), worlds A and B, one process (one model load). Reference cell for the change table: w1_k3all.
# W1 = E48_FRAMES=pair (every pair owns its own R sentence frames); the job exports it. The 30 candidate places, names and the L3 prefixes are identical in the shared-frame world and in W1 (e48_world_test asserts triples and ctrl lists are identical), so the 900 candidate rows are the same as in job 141; only the SAVED STATES (trained on the W1 stream) differ. The stream itself is not used for the readout.
# NO OVERWRITE: result files are named e49_cand_s101_w1_<cell>.json and e49_cand_s101_w1_W0.json (W0 gets its own name via --w0-name), the summary is e49_candidates_s101_w1.json (--tag _w1); none of them is one of the old-world files (e49_cand_s101_{native,k3all,sent,...}.json, W0.json, e49_candidates_s101.json). The job stops (exit 8) if the summary file already exists unless RERUN=1 (existing per-cell files are reused, restart-safe). --strict: a missing saved state or log aborts instead of skipping.
# PINNED INSTALL, NO FETCH: checks the code out of the local repository at commit $COMMIT (exit 4 if absent), verifies sha256 (first 12 hex) of the e49 files (exit 5). The lab ladder.py / gen_world.py must be the patched W1 versions the W1 states were written with (exit 6 otherwise; not overwritten). NO_INSTALL=1 skips the checkout, SKIP_VERIFY=1 the check.
# CPU-only dry run (separate directory): git fetch origin claude/project-thread-rwhdac && git worktree add --detach /tmp/dry_w1 FETCH_HEAD && ln -s "$(pwd)/data" /tmp/dry_w1/data && cd /tmp/dry_w1 && PY=python3 S=scripts/exp07_write_ladder CPU_ONLY=1 bash scripts/exp07_write_ladder/jobs_proposed/160_w1_e49_candidates_5cells.sh   (then: git worktree remove --force /tmp/dry_w1). It runs the CPU tests, checks the lab files, and lists for each cell the saved-state directories and logs (A, B) and the result files that would be written.
# Queue position (design / core): after the E52 L2a job (160_e52_L2a_native.sh) and before 161_e52_L2b_native.sh; file name 160_w1_... sorts after 160_e52_... and before 161_... in both C and en_US collation.
set -e
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder
D4=scripts/exp04_what_ttt_retains
COMMIT=fc4bb7efa2739a093980a4347f151d4a3626b49c
FILES="$D/e49_stats.py $D/e49_candidates.py $D/e49_test.py $D/e48_world_test.py"
LADDER_OK="714b08ea3b90"
GW_OK="c1ae126eadda"
FB_OK="1927a9ee330b eeaaa9135907"
W1_CELLS="${W1_CELLS:-w1_native=L0fs_w1main w1_oracle=S3_w1_oracle w1_k6p=S3_w1_k6p w1_k3all=S3_w1_k3all w1_sent=S3_w1_sent}"
if [ -z "$NO_INSTALL" ]; then
  for X in "$D/ladder.py:$LADDER_OK" "$D4/gen_world.py:$GW_OK" "$D4/frame_bank.py:$FB_OK"; do
    GL=$(sha256sum ${X%%:*} | cut -c1-12); case " ${X#*:} " in *" $GL "*) ;; *) echo "lab file ${X%%:*} ($GL) is not the patched W1 version (${X#*:}): exit 6, the code thread must rebase"; exit 6;; esac
  done
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
961e4116343d e49_candidates.py
7578ebd07e3d e49_test.py
70a70838d527 e48_world_test.py

MANIFEST
  echo "installed code verified against the manifest"
fi
export E48_FRAMES=pair
$PY $D/e49_stats.py | tail -1
$PY $D/e49_test.py | tail -1
$PY $D/e48_world_test.py | tail -1
CARGS=""; for C in $W1_CELLS; do
  N=${C%%=*}; P=${C#*=}
  echo "$N ($P): states A $(ls $OUT/states/${P}_s101/A 2>/dev/null | wc -l), B $(ls $OUT/states/${P}_s101/B 2>/dev/null | wc -l), saved log A: $([ -f $OUT/${P}_s101_A.saved.json ] && echo yes || echo NO), B: $([ -f $OUT/${P}_s101_B.saved.json ] && echo yes || echo NO); result file: $OUT/e49_cand_s101_${N}.json $([ -f $OUT/e49_cand_s101_${N}.json ] && echo '(EXISTS, will be reused)' || echo '(new)')"
  CARGS="$CARGS --cell $C"
done
echo "W0 result file: $OUT/e49_cand_s101_w1_W0.json $([ -f $OUT/e49_cand_s101_w1_W0.json ] && echo '(EXISTS, will be reused)' || echo '(new)'); summary: $OUT/e49_candidates_s101_w1.json $([ -f $OUT/e49_candidates_s101_w1.json ] && echo '(EXISTS)' || echo '(new)')"
[ -n "$CPU_ONLY" ] && exit 0
[ -f $OUT/e49_candidates_s101_w1.json ] && [ -z "$RERUN" ] && { echo "summary e49_candidates_s101_w1.json already exists: not overwritten (exit 8; RERUN=1 to redo the analysis)"; exit 8; }
$PY $D/e49_candidates.py --seed 101 --epoch 4 --w0 --w0-name w1_W0 --ref w1_k3all --tag _w1 --strict $CARGS

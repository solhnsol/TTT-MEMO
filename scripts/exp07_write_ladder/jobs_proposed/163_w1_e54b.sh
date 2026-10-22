#!/bin/bash
# EXPECTED RUN TIME (Hans confirms before it runs): ESTIMATE ONLY, NOT MEASURED, GPU path never run: default (alt readout only, 5 W1 cells x worlds A,B at pass 4 + W0) about 80-110 min, with slack about 140 min. Basis: job 160 W1 readout = about 1 min per 30-candidate state read; here each state is read with about 180 candidates x 30 items (about 6x the rows, about 6 min per state if the cost is linear in rows = 5 cells x 2 worlds x about 6 min = about 60 min) + 8 greedy steps x 30 items (small) + 13-23 min model load/setup + W0. Variant WITH_R1=1 adds the pg1342-tail readout (about +45 min). One call (one model load). NO training, NO streaming, no state is written or removed.
# E54(b) (Hans approved: "후보 없는 생성 정확도/수백 후보 순위 큐에 추가하는 것 승인할게"): on the 5 SAVED W1 states at pass 4 (native L0fs_w1main, oracle S3_w1_oracle, k6p S3_w1_k6p, k3all S3_w1_k3all, sent S3_w1_sent), worlds A and B, plus W0: (i) ranking of the 30 stream places (R30) and of the 30 stream places + up to 150 distractor places (R_ext; distractors come from the e51 place pool, are not used by world A/B, dev2 or the sealed holdout, are stratified to the token-length distribution of the 30 places with a fixed seed; the job prints how many are available per length and the shortfall), per-token log-prob ranking reported next to the summed one, same-length-only variant; (ii) candidate-free greedy generation (max 8 tokens, cut at the first newline or period): exact-match rate, share inside the 30 stream places, share of a non-gold stream place, and the swap partner place separately. 15-pair bootstrap CI table.
# W1 = E48_FRAMES=pair; the job exports it. Optional old-world cells (states must exist; given as OLD_CELLS="native=L0fs_main k3all=S3_k3all sent=S3_sent" (names=prefixes), none by default).
# NO OVERWRITE: result files e54b_s101_<cell>.json (restart-safe: existing ones are reused), report e54b_report_s101_w1.md; the job stops (exit 8) if the report exists unless RERUN=1. --strict: a missing saved state aborts.
# PINNED INSTALL, NO FETCH: checks the code out of the local repository at commit $COMMIT (exit 4 if absent), verifies sha256 (first 12 hex) (exit 5). The lab ladder.py / gen_world.py / frame_bank.py must be the patched W1 versions (exit 6 otherwise). NO_INSTALL=1 skips the checkout, SKIP_VERIFY=1 the check.
# CPU-only dry run: git fetch origin claude/project-thread-rwhdac && git worktree add --detach /tmp/dry_w1 FETCH_HEAD && ln -s "$(pwd)/data" /tmp/dry_w1/data && cd /tmp/dry_w1 && PY=python3 CPU_ONLY=1 bash scripts/exp07_write_ladder/jobs_proposed/163_w1_e54b.sh  (then: git worktree remove --force /tmp/dry_w1). It runs the CPU selftest, checks the lab files and lists states and result files.
# Queue position: after 162_e52_L1.sh (file name sorts after it).
set -e
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder
D4=scripts/exp04_what_ttt_retains
COMMIT=fca57574ac6b5ea6a9c7f4845e4bed9a95ab25dc
FILES="$D/e54b.py $D/e49_stats.py"
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
6ef9755fd6f7 e54b.py
794ae88a9baf e49_stats.py

MANIFEST
  echo "installed code verified against the manifest"
fi
export E48_FRAMES=pair
cd $D && $PY e54b.py --selftest | tail -1; cd - >/dev/null
CARGS=""; for C in $W1_CELLS $OLD_CELLS; do
  N=${C%%=*}; P=${C#*=}
  echo "$N ($P): states A $(ls $OUT/states/${P}_s101/A 2>/dev/null | wc -l), B $(ls $OUT/states/${P}_s101/B 2>/dev/null | wc -l), saved log A: $([ -f $OUT/${P}_s101_A.saved.json ] && echo yes || echo NO), B: $([ -f $OUT/${P}_s101_B.saved.json ] && echo yes || echo NO); result $OUT/e54b_s101_$N.json $([ -f $OUT/e54b_s101_$N.json ] && echo '(EXISTS, reused)' || echo '(new)')"
  CARGS="$CARGS --cell $C"
done
echo "W0 result: $OUT/e54b_s101_w1_W0.json; report: $OUT/e54b_report_s101_w1.md $([ -f $OUT/e54b_report_s101_w1.md ] && echo '(EXISTS)' || echo '(new)')"
[ -n "$CPU_ONLY" ] && exit 0
[ -f $OUT/e54b_report_s101_w1.md ] && [ -z "$RERUN" ] && { echo "report e54b_report_s101_w1.md already exists: not overwritten (exit 8; RERUN=1 to redo)"; exit 8; }
R1FLAG="--no-r1"; [ -n "$WITH_R1" ] && R1FLAG=""
$PY $D/e54b.py --seed 101 --epoch 4 --extra 150 --w0 --w0-name w1_W0 --tag _w1 --strict $R1FLAG $CARGS

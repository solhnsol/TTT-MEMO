#!/bin/bash
# E55 32.10 sentence-level selection dump (FORWARD ONLY, no training step of its own, one model load; GPU): world v3_16_1, seed 61. The stream is run in two arms to chunk 127, NAT and SENTPQ-ADAM-0.5N (SENTPQADAMH, norm target = the SENTPQ update norms of tag q1b, reused); at chunks 16, 32, 64, 127 the weights BEFORE the write of the chunk (= the state after k writes, the a = k read-point state; chunk 127 is the exact state before the last write, not the a = 128 approximation) are used for the dump with the UNMASKED real 8-chunk window. The saved read-point states are NOT used: the decomp runs of tag q1b did not save weights, so both arms are re-run (the K,V of the stream are recorded once for both).
# Dump per (arm, chunk): A[g1/g2/g3, head, query sentence of the write chunk, window sentence] float16 = attention mass of the query sentence tokens (mean over its tokens) on the key tokens of the window sentence (sum), layer-group mean; query sentence token count, offset and mean token CE of the write forward (arm's own window, weights before the update); window sentence token counts, offsets, chunk numbers; token ids of window and chunk. Sentences from end punctuation + space / newline only (e55_uv.sentence_ends); the fact table is NOT used for them. Labels (fact / background, fact id, chunk) in a SEPARATE file e55_sentdump_labels_s61.npz; the selection code (e55_sentsel.py) never imports or opens it (self-test checks the source).
# EXTENDED (commit above): per layer and head A (AL) and attention x |v| (AV, V-norm of every window key), V norms per kv head (VN), per query token and head: max window mass, summed mass, entropy, max key index, window sentence of the max key, mass on the first 4 keys (sink) and on the previous chunk, token CE of the write chunk. Disk about 0.5 GB for the 8 dumps (AL + AV float16 about 26 MB each per dump, compressed npz), GPU part +0.1-0.2 h over the first version (second aggregation einsum per layer and the token statistics).
# CPU analysis in the same job (e55_sentgrid.py): grid V1 sharpness gate, V2 rank (top q % of the sentences by votes), V3 relative threshold (mean + z std), V4 query selection by sentence CE, V5 rare-and-repeated token score, mixes V1+V4 / V1+V5, layer groups g2 / g3 / all, heads summed or voting separately; configurations 589 (V1-V5 as before; V6 sharpest heads + k-times trigger; V7 head scores sharp / copy / induction, top 5 / 10 / 25 % of heads, sink and previous-chunk exclusion, attention x value, fact-like query tokens only, sentence aggregation sum / max / top-5 mean, exact n-gram string baseline); metrics accuracy, precision, recall, F1 (primary) (chance = fact share of the window), hidden share, background false-mask per state and chunk. MULTIPLE-COMPARISON RULE fixed in the report: select on chunks 16, 32, confirm on 64, 127, verdict from the confirmation numbers of the selection-top-1 configuration; pass rules fixed in advance: STRICT recall >= 0.8, precision >= 0.8, hidden <= 0.40; RELAXED recall >= 0.75, precision >= 0.65, hidden <= 0.45 (report says which one is met). Report $OUT/e55_sentgrid_report.md (NEW name); outputs $OUT/e55_sentdump_s61.npz / .json / e55_sentdump_labels_s61.npz.
# EXPECTED RUN TIME (Hans confirms before it runs): ESTIMATE ONLY, unmeasured. About 0.8-1.3 h, longer than the design estimate of 20-30 min (that assumed stored read-point states, which do not exist): CPU part 5-10 min; model load 7-20 min once; recording the natural K,V of 128 chunks once 6-10 min; per arm 128 updates at 3.5-4.6 s = 7.5-10 min + 4 dumps (one eager attention pass with all 1024 query tokens, about 1.2 GB score tensors per layer, plus the sentence aggregation einsums; guessed 30-60 s each) 2-4 min = about 10-14 min per arm; CPU grid 3-10 min (589 configurations, V7 part measured about 1 min on synthetic arrays of real size). Contended server up to 2x.
# Needs: E55_q1b_SENTPQ_s61.saved.json (norm target). Not run on a GPU: the real attention pass with the sentence aggregation on the 3B model (memory), the write forward CE path, the real-tokenizer sentence layout (checked on the lab by the CPU_ONLY part). NOT APPROVED yet.
# CPU-only dry run: git fetch origin claude/project-thread-rwhdac && git worktree add --detach /tmp/dry_sd FETCH_HEAD && ln -s "$(pwd)/data" /tmp/dry_sd/data && cd /tmp/dry_sd && PY=/abs/path/python CPU_ONLY=1 bash scripts/exp07_write_ladder/jobs_proposed/213_e55_sentdump_s61.sh   (then git worktree remove --force /tmp/dry_sd)
# PY must be an ABSOLUTE path to the lab python. PINNED INSTALL, NO FETCH (exit 4 / 5 / 6 as in job 200).
set -eo pipefail
PY=$(realpath -s "$PY")   # the runner exports a relative PY (.venv/bin/python): make it absolute BEFORE the cd below, WITHOUT resolving symlinks (uv venv; readlink -f breaks it)
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder
D4=scripts/exp04_what_ttt_retains
COMMIT=85b9aab99a4968305024080381b49d986600ba6b
FILES="$D/e55_run.py $D/e55_core.py $D/e55_gpu.py $D/e55_swap.py $D/e55_read.py $D/e55_analyze.py $D/e55_metrics.py $D/e55_ic_report.py $D/e55_iso.py $D/e55_diag.py $D/e55_votegrid.py $D/e55_uv.py $D/s3.py $D/rope_fly.py $D/e55_sentdump.py $D/e55_sentdump_labels.py $D/e55_sentsel.py $D/e55_sentgrid.py $D/e55_gates.py $D4/gen_facts.py $D4/gen_world_ctl.py $D4/frame_bank.py $D4/frames300.txt"
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
1c2e709f0839 e55_run.py
f05caa0a4811 e55_core.py
9827e34dcde6 e55_gpu.py
3ee7e3fa1960 e55_swap.py
a6d506949071 e55_read.py
4a811650bcd8 e55_analyze.py
30769e241d5e e55_metrics.py
7a22509c96c5 e55_ic_report.py
c4c8a0d4e386 e55_iso.py
725d18d16b65 e55_diag.py
597294ca312b e55_votegrid.py
6d9e9b62cecd e55_uv.py
d0f6be8c5479 s3.py
76802d72e2c4 rope_fly.py
16bf1b4d7808 e55_sentdump.py
df41b7d13739 e55_sentdump_labels.py
3728d3954e71 e55_sentsel.py
68f5bdb120b6 e55_sentgrid.py
eb8bc59bf9a0 e55_gates.py
b867b6f680f3 ../exp04_what_ttt_retains/gen_facts.py
1cf0caa82553 ../exp04_what_ttt_retains/gen_world_ctl.py
eeaaa9135907 ../exp04_what_ttt_retains/frame_bank.py
866eb5989467 ../exp04_what_ttt_retains/frames300.txt
MANIFEST
  echo "installed code verified against the manifest"
fi
S=$D
SEED=61
for F in q1b_SENTPQ; do [ -f $OUT/E55_${F}_s$SEED.saved.json ] || [ -n "$CPU_ONLY" ] || { echo "file E55_${F}_s$SEED.saved.json missing (exit 7)"; exit 7; }; done
$PY $D4/gen_facts.py --selftest | tail -1
$PY $S/e55_read.py --selftest | tail -1
$PY $S/e55_swap.py | tail -1
$PY $S/e55_sentgrid.py --selftest
$PY $S/e55_gates.py --cell v3_16_1 --seed $SEED --default-refs --require-refs --out $OUT/e55_gates_sd_s$SEED.md > $OUT/e55_gates_sd_s$SEED.log 2>&1 || { tail -30 $OUT/e55_gates_sd_s$SEED.log; echo "background gates failed (exit 7)"; exit 7; }
tail -4 $OUT/e55_gates_sd_s$SEED.md
$PY $S/e55_run.py dry --cell v3_16_1 --seed $SEED --bg pool --arms NAT | tail -3
$PY $S/e55_run.py sentdump --cell v3_16_1 --seed $SEED --bg pool --diag-chunks 16,32,64,127 --norm-ref-tag q1b --out $OUT --check-only
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
REP=$OUT/e55_sentgrid_report.md
if [ -f $REP ] && [ -z "$RERUN" ]; then echo "report $REP exists: not redone (exit 8; RERUN=1 to redo)"; exit 8; fi
$PY $S/e55_run.py sentdump --cell v3_16_1 --seed $SEED --bg pool --diag-chunks 16,32,64,127 --norm-ref-tag q1b --out $OUT
$PY $S/e55_sentgrid.py --dump $OUT/e55_sentdump_s$SEED.npz --labels $OUT/e55_sentdump_labels_s$SEED.npz --out $REP
tail -60 $REP

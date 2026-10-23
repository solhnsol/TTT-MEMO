#!/bin/bash
# E55 32.8 online label-free sentence-vote masks (UNIONVOTE), one job, one model load. World v3_16_1 (the 181 world), seed 61 first (SEEDS=61,62,63 for the 3-seed extension), window kept, write = Adam direction with factor 0.5 and per-chunk step = 0.5 x the update norm of SENTPQ-STD of the same seed (tag q1b files are the norm targets, reused, NOT re-run). Reads: out-of-window at a = 16, 32, 64, 128; in-window at a = 128; DCLM first-chunk NLL at a = 128; canary reads at all appearances.
# Arms: UVA (sentence votes, last third of the prefix layers, p = 25 %, k = 2), UVB (p = 25 %, k = 3), UVC (p = 20 %, k = 1). For every write chunk, BEFORE its write and with the current weights: a forward-only attention pass of the prefix layers with ALL tokens of the chunk as queries (nothing from the fact table); each (layer, head, token) contributes its top-8 window keys as votes; sentence boundaries come from the token stream (end punctuation + space/newline token, newline, chunk start); each query sentence selects the top p % of the window sentences (candidates >= 4 tokens) by total votes (score = count of top-8 hits, zero excluded, as the lab sentsel vote variant; the lab's offline diagnostic used the 16 question spans as queries, here every sentence of the chunk is a query); a window sentence is hidden from every query of the chunk when >= k query sentences selected it (1-D key mask like SENT). RSA / RSB / RSC = random window sentences with the same hidden-token count per chunk as UVA / UVB / UVC (they read the n_masked_keys of the UV arm of the same seed, so UV arms run first). The fact table is used for the LOG only (fact recall, precision, background false-mask rate per chunk).
# Pre-set rule (design 32.8, in the report): R = g(UV)/g(SENTPQ-ADAM-0.5N) at a = 128 (reference = tag q1b); R >= 0.7 and g(UV) - g(RS) >= +0.10 -> label-free sentence masking works; R <= 0.3 -> does not. One seed = first observation.
# Report $OUT/e55_uv_report.md (NEW name). Arms of the default list = 6 arms x 1 seed = 6 runs.
# EXPECTED RUN TIME (Hans confirms before it runs): ESTIMATE ONLY, unmeasured. About 1.4-2.5 h for seed 61: CPU part 5-10 min; model load 7-20 min once; 6 runs of 12-22 min each (128 chunks + K,V recording 6-10 min + reads 3-6 min; the UV arms add one eager attention pass per chunk, guessed 2-6 min per arm; the RS arms have no extra pass). The extra pass is the unmeasured part: if one pass takes 3 s the UV arms add 6.4 min each, if 10 s 21 min each. 3 seeds: about 3x the run part.
# Outputs (NEW names): $OUT/E55_uv_<arm>_s<seed>.* , $OUT/e55_uv_report.md; the arms whose saved.json exists are skipped (RERUN=1 to redo them); exit 8 if the report exists unless RERUN=1.
# CPU-only dry run (arms through decomp --check-only; no GPU, no model): git fetch origin claude/project-thread-rwhdac && git worktree add --detach /tmp/dry_uv FETCH_HEAD && ln -s "$(pwd)/data" /tmp/dry_uv/data && cd /tmp/dry_uv && PY=/abs/path/python CPU_ONLY=1 bash scripts/exp07_write_ladder/jobs_proposed/201_e55_uv_online_sentence_votes.sh   (then git worktree remove --force /tmp/dry_uv)
# PY must be an ABSOLUTE path to the lab python. PINNED INSTALL, NO FETCH (exit 4 if the commit is absent, exit 5 on a sha256 mismatch). The lab ladder.py must be the patched W1 version (exit 6); the lab s3.py may be any committed version up to the pinned commit (exit 6 otherwise; this job's s3.py adds the att_noout flag, installed from the commit). Memory: the attention pass materialises [heads, 1024, 9216] float32 score tensors per layer (about 1.2 GB each, a few alive at once) on top of the training state; the Adam fp32 temporaries are freed before it. A [mem] line is printed after every arm.
# Needs: NAT file of job 181 (tag v3a) and SENTPQ, SENTPQADAMH, NATADAM files of tag q1b (jobs 195/196) per seed. Independent of the other proposed jobs. NOT APPROVED yet. Not run on a GPU: the real attention pass of the 3B model with all 1024 query tokens, the vote selection on real attention, the RS matching on the lab, DCLM reads with these arms.
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
SEEDS="${SEEDS:-61}"
ARMS="${ARMS:-UVA,UVB,UVC,RSA,RSB,RSC}"
NORM="$OUT/E55_v3a_NAT_s{seed}.saved.json"
DCLM=data/dclm_val_subset/val_c0_200x8193.u32le.bin
DEC="--cell v3_16_1 --tag uv --norm-ref-tag q1b --bg pool --out $OUT --seeds $SEEDS --arms $ARMS --norm-files $NORM --decomp-apps 16,32,64,128 --in-window --in-apps 128 --canary-all"
$PY $D4/gen_facts.py --selftest | tail -1
$PY $S/e55_read.py --selftest | tail -1
$PY $S/e55_swap.py | tail -1
$PY $S/e55_metrics.py --selftest | tail -1
$PY $S/e55_ic_report.py --selftest | tail -1
for SD in $(echo $SEEDS | tr ',' ' '); do
  $PY $S/e55_gates.py --cell v3_16_1 --seed $SD --default-refs --require-refs --out $OUT/e55_gates_uv_s$SD.md > $OUT/e55_gates_uv_s$SD.log 2>&1 || { tail -30 $OUT/e55_gates_uv_s$SD.log; echo "background gates failed for seed $SD (exit 7)"; exit 7; }
  tail -4 $OUT/e55_gates_uv_s$SD.md
  for F in v3a_NAT q1b_SENTPQ q1b_SENTPQADAMH q1b_NATADAM; do [ -f $OUT/E55_${F}_s$SD.saved.json ] || [ -n "$CPU_ONLY" ] || { echo "file E55_${F}_s$SD.saved.json missing (exit 7)"; exit 7; }; done
done
$PY $S/e55_run.py dry --cell v3_16_1 --seed 61 --bg pool --arms NAT | tail -3
$PY $S/e55_run.py decomp $DEC --check-only
[ -f "$DCLM" ] && echo "DCLM ids present: $(stat -c %s $DCLM) bytes" || echo "DCLM ids file absent here (needed on the GPU run)"
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
[ -f "$DCLM" ] || { echo "DCLM ids file $DCLM missing (exit 7)"; exit 7; }
REP=$OUT/e55_uv_report.md
if [ -f $REP ] && [ -z "$RERUN" ]; then echo "report $REP exists: not redone (exit 8; RERUN=1 to redo)"; exit 8; fi
$PY $S/e55_run.py decomp $DEC --dclm-ids $DCLM --dclm-n 100
$PY $S/e55_ic_report.py --uv --dir $OUT --seeds $SEEDS --tag uv --ref-tag q1b --out $REP
tail -60 $REP

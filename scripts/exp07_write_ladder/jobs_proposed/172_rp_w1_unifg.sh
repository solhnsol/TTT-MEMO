#!/bin/bash
# EXPECTED RUN TIME (Hans confirms before it runs): about 75 min on the RunPod A100, with slack about 1.6 h (design thread E53 section 8: 53 min + about 20 min overhead; NOT measured on the pod).
# E53 UNIF-G on W1: O + uniformly random positions, the number per chunk equal to the per-chunk number of masked tokens of BG (s3.py --cell oraclefill --fill-size-from w1_bg --fill-no-o: O-free chunks are filled too, because BG masks the background there; the total O + fill per chunk is the n_masked of BG, asserted after the run by rp_unifg_check.py; job 88 did this in the old world without --fill-no-o). Tag w1_unifg (results S3_w1_unifg_s101_*). Reads $OUT/S3_w1_bg_s101_{A,B}.saved.json written by job 171: it must run on the same volume AFTER job 171 (exit 3 if missing). Known limit (E53 section 8): the random positions may include exposure-sentence tokens and BOS.
# NO OVERWRITE: exit 8 if S3_w1_unifg_s101_A.saved.json exists.
# CPU-only dry run (checks the commit, the manifest, the CPU unit tests, the environment and the output names; no GPU, no model): git fetch origin claude/project-thread-rwhdac && git worktree add --detach /tmp/dry_rp FETCH_HEAD && ln -s "$(pwd)/data" /tmp/dry_rp/data && ln -s "$(pwd)/checkpoints" /tmp/dry_rp/checkpoints && cd /tmp/dry_rp && PY=python3 CPU_ONLY=1 NO_INSTALL=1 bash scripts/exp07_write_ladder/jobs_proposed/172_rp_w1_unifg.sh   (then: git worktree remove --force /tmp/dry_rp; with NO_INSTALL=1 the checkout step is skipped because the worktree is already at the pinned commit; the manifest check still runs)

set -e
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder
D4=scripts/exp04_what_ttt_retains
COMMIT=c92e4b36e07af112e4c4101d816d6d4a70873a5c
FILES="scripts/exp07_write_ladder/s3.py scripts/exp07_write_ladder/s3_sets.py scripts/exp07_write_ladder/e43_sets_test.py scripts/exp07_write_ladder/e44_sets_test.py scripts/exp07_write_ladder/e46_sets_test.py scripts/exp07_write_ladder/e47_sets_test.py scripts/exp07_write_ladder/e42_k2_tokens.py scripts/exp07_write_ladder/k3_counts.py scripts/exp07_write_ladder/suppress.py scripts/exp07_write_ladder/ladder.py scripts/exp07_write_ladder/margins_split.py scripts/exp07_write_ladder/margins_split_test.py scripts/exp07_write_ladder/e48_world_test.py scripts/exp07_write_ladder/e48_world_stats.py scripts/exp07_write_ladder/e47_native_da.py scripts/exp07_write_ladder/rp_native_check.py scripts/exp07_write_ladder/rp_unifg_check.py scripts/exp04_what_ttt_retains/gen_world.py scripts/exp04_what_ttt_retains/frame_bank.py"
: "${PY:=python3}"; : "${S:=$D}"; : "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"
JOB=172_rp_w1_unifg
RPE="${RP_ENV:-/workspace/rp_env.sh}"; [ -f "$RPE" ] && . "$RPE"   # written by jobs_proposed/rp_setup.sh on the pod (PY, OUT, XLA_FLAGS, LD_LIBRARY_PATH); on the lab server this file does not exist and the runner environment is used
# RunPod: every file of the pinned commit is checked out (the pod holds a repo clone, not the lab-patched copies); NO_INSTALL=1 skips the checkout, SKIP_VERIFY=1 the manifest check.
if [ -z "$NO_INSTALL" ]; then
  git cat-file -e $COMMIT^{commit} || { echo "commit $COMMIT not in the local clone: git fetch origin claude/project-thread-rwhdac first (exit 4)"; exit 4; }
  git checkout $COMMIT -- $FILES
fi
if [ -z "$SKIP_VERIFY" ]; then
  while read HH F; do
    [ -n "$HH" ] || continue
    G=$(sha256sum $F | cut -c1-12)
    [ "$G" = "$HH" ] || { echo "VERSION MISMATCH $F: have $G, expected $HH (exit 5)"; exit 5; }
  done <<'MANIFEST'
a16e6dd0d4c9 scripts/exp07_write_ladder/s3.py
b69477a8f778 scripts/exp07_write_ladder/s3_sets.py
a256dbbcb9ec scripts/exp07_write_ladder/e43_sets_test.py
3dc2e49e33fc scripts/exp07_write_ladder/e44_sets_test.py
0b7f4b351d23 scripts/exp07_write_ladder/e46_sets_test.py
99da9b569218 scripts/exp07_write_ladder/e47_sets_test.py
9153e937b8c1 scripts/exp07_write_ladder/e42_k2_tokens.py
fe0ba0b228a6 scripts/exp07_write_ladder/k3_counts.py
9dac9fdda99e scripts/exp07_write_ladder/suppress.py
714b08ea3b90 scripts/exp07_write_ladder/ladder.py
2638a505ef0c scripts/exp07_write_ladder/margins_split.py
1daebd776335 scripts/exp07_write_ladder/margins_split_test.py
70a70838d527 scripts/exp07_write_ladder/e48_world_test.py
364c5a73bb47 scripts/exp07_write_ladder/e48_world_stats.py
3841e2bdecb5 scripts/exp07_write_ladder/e47_native_da.py
14e4cac89515 scripts/exp07_write_ladder/rp_native_check.py
fc551e690ff6 scripts/exp07_write_ladder/rp_unifg_check.py
c1ae126eadda scripts/exp04_what_ttt_retains/gen_world.py
eeaaa9135907 scripts/exp04_what_ttt_retains/frame_bank.py

MANIFEST
  echo "installed code verified against the manifest"
fi
export E48_FRAMES=pair
# environment checks (no GPU work): deterministic ops flag, GPU visible, checkpoint, tokenizer, free disk
case "$XLA_FLAGS" in *deterministic_ops=true*) ;; *) echo "XLA_FLAGS must contain --xla_gpu_deterministic_ops=true (ladder.py asserts it): exit 3"; exit 3;; esac
if [ -z "$CPU_ONLY" ]; then nvidia-smi -L || { echo "no GPU visible: exit 3"; exit 3; }; fi
[ -d checkpoints/3b_ttt_e2e_finetune_books_128k_3x_cc/1949 ] || { echo "checkpoint checkpoints/3b_ttt_e2e_finetune_books_128k_3x_cc/1949 (the 3B TTT-E2E 128K Books checkpoint, step 1949, about 11 GB; baseline_config.CHECKPOINT_DIR) missing: exit 3"; exit 3; }
[ -f data/tokenizer/llama3/tokenizer.json ] || { echo "tokenizer data/tokenizer/llama3/tokenizer.json missing: exit 3"; exit 3; }
mkdir -p "$OUT/states"
AV=$(df -P --block-size=1G "$OUT" | awk 'NR==2{print $4}'); echo "free disk at $OUT: $AV GB (need >= 40)"
[ "$AV" -ge 40 ] || { echo "not enough free disk (exit 3 outside CPU_ONLY)"; [ -n "$CPU_ONLY" ] || exit 3; }
MARK=$(mktemp)
$PY $S/e43_sets_test.py
$PY $S/e44_sets_test.py
$PY $S/e47_sets_test.py
$PY $S/margins_split_test.py
$PY $S/e48_world_test.py
$PY $S/rp_native_check.py --selftest | tail -1
$PY $S/rp_unifg_check.py --selftest | tail -1
for W in A B; do [ -f "$OUT/S3_w1_bg_s101_$W.saved.json" ] || { echo "$OUT/S3_w1_bg_s101_$W.saved.json missing (job 171 must run first on this volume): exit 3"; exit 3; }; done
[ -f "$OUT/S3_w1_unifg_s101_A.saved.json" ] && { echo "S3_w1_unifg_s101_A.saved.json exists: not overwritten (exit 8)"; exit 8; }
[ -n "$CPU_ONLY" ] && { echo "would write S3_w1_unifg_s101_* (+ states)"; exit 0; }
for M in save eval; do
  $PY $S/s3.py --cell oraclefill --fill-size-from w1_bg --fill-no-o --world-seed 101 --passes 4 --worlds AB --keep --tag w1_unifg --mode $M
done
$PY $S/e47_native_da.py --seed 101 --cells S3:w1_unifg --out-dir "$OUT/rp_unifg"
# size assertion: the TOTAL masked tokens (O + fill) per world, pass and chunk equal BG's n_masked
$PY $S/rp_unifg_check.py --bg w1_bg --unifg w1_unifg | tee "$OUT/rp_unifg_check.txt"
# provenance: pod environment and sha256 (first 64 hex) of every result file this job wrote (states: sizes + sha256 of the pass-4 states)
{ echo "job $JOB commit $COMMIT"; date -u; hostname; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; $PY -c "import jax;print('jax',jax.__version__,jax.devices())"; echo "XLA_FLAGS=$XLA_FLAGS"; } > "$OUT/rp_env_$JOB.txt" 2>&1 || true
(cd "$OUT" && find . -maxdepth 1 -type f -newer "$MARK" ! -name 'rp_sha_*' -exec sha256sum {} +) > "$OUT/rp_sha_$JOB.txt"
for T in S3_w1_unifg; do for W in A B; do F="$OUT/states/${T}_s101/$W/state_004.npz"; [ -f "$F" ] && { ls -l "$F"; sha256sum "$F"; }; done; done >> "$OUT/rp_sha_$JOB.txt" 2>&1 || true
echo "provenance: $OUT/rp_env_$JOB.txt, $OUT/rp_sha_$JOB.txt"

#!/bin/bash
# EXPECTED RUN TIME (Hans confirms before it runs): about 25-30 min on the RunPod A100, with slack about 40 min (same basis as job 160_w1_nll_dclm: 8 states + W0 + W0 repeat = 10 reads of 200 samples at about 1.1-2.3 min plus about 5 min load; NOT measured). NO training, NO weights written.
# DCLM first-chunk damage read (E37c method) of the PASS-4 states of the RunPod decomposition cells, worlds A and B: re-run native (L0fs_w1rp), BG (S3_w1_bg), NOO (S3_w1_noo), UNIF-G (S3_w1_unifg), plus W0 and a W0 repeat (max |diff| must be 0). It must run BEFORE any state of these cells is deleted or the pod is terminated; run it right after job 172 on the same volume. Needs data/dclm_val_subset/val_c0_200x8193.u32le.bin on the pod. Output (tag e37w1dec): $OUT/e37_firstchunk_e37w1dec.{json,csv} and the table $OUT/e37_w1dec_nll_table.md. NO OVERWRITE: exit 8 if the json exists. Cells whose pass-4 states are absent are listed and the job stops (exit 3).
# CPU-only dry run (checks the commit, the manifest, the CPU unit tests, the environment and the output names; no GPU, no model): git fetch origin claude/project-thread-rwhdac && git worktree add --detach /tmp/dry_rp FETCH_HEAD && ln -s "$(pwd)/data" /tmp/dry_rp/data && ln -s "$(pwd)/checkpoints" /tmp/dry_rp/checkpoints && cd /tmp/dry_rp && PY=python3 CPU_ONLY=1 NO_INSTALL=1 bash scripts/exp07_write_ladder/jobs_proposed/173_rp_w1_dclm_nll.sh   (then: git worktree remove --force /tmp/dry_rp; with NO_INSTALL=1 the checkout step is skipped because the worktree is already at the pinned commit; the manifest check still runs)

set -e
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder
D4=scripts/exp04_what_ttt_retains
COMMIT=c92e4b36e07af112e4c4101d816d6d4a70873a5c
FILES="scripts/exp07_write_ladder/e37_firstchunk_nll.py scripts/exp07_write_ladder/e37_nll_table.py scripts/exp07_write_ladder/ladder.py scripts/exp07_write_ladder/suppress.py scripts/exp07_write_ladder/margins_split.py scripts/exp04_what_ttt_retains/gen_world.py scripts/exp04_what_ttt_retains/frame_bank.py scripts/exp07_write_ladder/e47_native_da.py"
: "${PY:=python3}"; : "${S:=$D}"; : "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"
JOB=173_rp_w1_dclm_nll
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
e167e94bc935 scripts/exp07_write_ladder/e37_firstchunk_nll.py
5d2cd4cae5e9 scripts/exp07_write_ladder/e37_nll_table.py
714b08ea3b90 scripts/exp07_write_ladder/ladder.py
9dac9fdda99e scripts/exp07_write_ladder/suppress.py
2638a505ef0c scripts/exp07_write_ladder/margins_split.py
c1ae126eadda scripts/exp04_what_ttt_retains/gen_world.py
eeaaa9135907 scripts/exp04_what_ttt_retains/frame_bank.py
3841e2bdecb5 scripts/exp07_write_ladder/e47_native_da.py

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
AV=$(df -P --block-size=1G "$OUT" | awk 'NR==2{print $4}'); echo "free disk at $OUT: $AV GB (need >= 10)"
[ "$AV" -ge 10 ] || { echo "not enough free disk (exit 3 outside CPU_ONLY)"; [ -n "$CPU_ONLY" ] || exit 3; }
MARK=$(mktemp)
IDS=data/dclm_val_subset/val_c0_200x8193.u32le.bin
[ -f "$IDS" ] || { echo "$IDS missing: exit 3"; exit 3; }
ARGS=""
for C in w1rp_native=L0fs_w1rp w1_bg=S3_w1_bg w1_noo=S3_w1_noo w1_unifg=S3_w1_unifg; do
  N=${C%%=*}; P=${C#*=}
  for W in A B; do
    F=$OUT/states/${P}_s101/$W/state_004.npz; [ -f "$F" ] || { echo "missing $F (exit 3)"; exit 3; }
    ARGS="$ARGS --state ${N}_p4_${W}=$F"
  done
done
$PY $S/e37_nll_table.py --selftest | tail -1
[ -f "$OUT/e37_firstchunk_e37w1dec.json" ] && { echo "$OUT/e37_firstchunk_e37w1dec.json exists: not overwritten (exit 8)"; exit 8; }
[ -n "$CPU_ONLY" ] && { echo "would read: $ARGS"; exit 0; }
$PY $S/e37_firstchunk_nll.py --ids $IDS --seed 101 --tag e37w1dec $ARGS
$PY $S/e37_nll_table.py --json $OUT/e37_firstchunk_e37w1dec.json --out $OUT/e37_w1dec_nll_table.md
# provenance: pod environment and sha256 (first 64 hex) of every result file this job wrote (states: sizes + sha256 of the pass-4 states)
{ echo "job $JOB commit $COMMIT"; date -u; hostname; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; $PY -c "import jax;print('jax',jax.__version__,jax.devices())"; echo "XLA_FLAGS=$XLA_FLAGS"; } > "$OUT/rp_env_$JOB.txt" 2>&1 || true
(cd "$OUT" && find . -maxdepth 1 -type f -newer "$MARK" ! -name 'rp_sha_*' -exec sha256sum {} +) > "$OUT/rp_sha_$JOB.txt"
for T in ; do for W in A B; do F="$OUT/states/${T}_s101/$W/state_004.npz"; [ -f "$F" ] && { ls -l "$F"; sha256sum "$F"; }; done; done >> "$OUT/rp_sha_$JOB.txt" 2>&1 || true
echo "provenance: $OUT/rp_env_$JOB.txt, $OUT/rp_sha_$JOB.txt"

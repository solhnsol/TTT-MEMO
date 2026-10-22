#!/bin/bash
# EXPECTED RUN TIME (Hans confirms before it runs): about 40-55 min on the RunPod A100, with slack about 70 min (ESTIMATE, not measured: job 160 W1 readout basis, about 1 min per state read; here 4 cells x 4 passes x worlds A,B = 32 reads + W0 + about 5 min load + 2-3 min building rows; one call, one model load). NO training, no state written or removed.
# E49 30-candidate readout (same definition as job 160_w1_e49_candidates_5cells: all 30 places scored as the continuation of each L3 item, packed fp32 R1 and R1-alt tails) on EVERY pass 1-4 of the RunPod decomposition cells: re-run native (L0fs_w1rp), BG (S3_w1_bg), NOO (S3_w1_noo), UNIF-G (S3_w1_unifg), worlds A and B, plus W0; result names e49_cand_s101_<cell>_p<pass>.json, summary e49_candidates_s101_w1dec.json (tag _w1dec, W0 name w1dec_W0; none of them is a lab file name). Reference cell for the change table: w1rp_native. Must run BEFORE the states are deleted or the pod is terminated, after job 172 on the same volume. NO OVERWRITE: exit 8 if the summary exists unless RERUN=1; --strict aborts on a missing state.
# CPU-only dry run (checks the commit, the manifest, the CPU unit tests, the environment and the output names; no GPU, no model): git fetch origin claude/project-thread-rwhdac && git worktree add --detach /tmp/dry_rp FETCH_HEAD && ln -s "$(pwd)/data" /tmp/dry_rp/data && ln -s "$(pwd)/checkpoints" /tmp/dry_rp/checkpoints && cd /tmp/dry_rp && PY=python3 CPU_ONLY=1 NO_INSTALL=1 bash scripts/exp07_write_ladder/jobs_proposed/174_rp_w1_candidates.sh   (then: git worktree remove --force /tmp/dry_rp; with NO_INSTALL=1 the checkout step is skipped because the worktree is already at the pinned commit; the manifest check still runs)

set -e
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder
D4=scripts/exp04_what_ttt_retains
COMMIT=0ab15fe2cf9038ed26d0f67013db8125823ea002
FILES="scripts/exp07_write_ladder/e49_stats.py scripts/exp07_write_ladder/e49_candidates.py scripts/exp07_write_ladder/e49_test.py scripts/exp07_write_ladder/e48_world_test.py scripts/exp07_write_ladder/ladder.py scripts/exp07_write_ladder/suppress.py scripts/exp07_write_ladder/margins_split.py scripts/exp04_what_ttt_retains/gen_world.py scripts/exp04_what_ttt_retains/frame_bank.py"
: "${PY:=python3}"; : "${S:=$D}"; : "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"
JOB=174_rp_w1_candidates
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
794ae88a9baf scripts/exp07_write_ladder/e49_stats.py
961e4116343d scripts/exp07_write_ladder/e49_candidates.py
7578ebd07e3d scripts/exp07_write_ladder/e49_test.py
70a70838d527 scripts/exp07_write_ladder/e48_world_test.py
714b08ea3b90 scripts/exp07_write_ladder/ladder.py
9dac9fdda99e scripts/exp07_write_ladder/suppress.py
2638a505ef0c scripts/exp07_write_ladder/margins_split.py
c1ae126eadda scripts/exp04_what_ttt_retains/gen_world.py
eeaaa9135907 scripts/exp04_what_ttt_retains/frame_bank.py

MANIFEST
  echo "installed code verified against the manifest"
fi
export E48_FRAMES=pair
# environment checks (no GPU work): deterministic ops flag, GPU visible, checkpoint, tokenizer, free disk
case "$XLA_FLAGS" in *deterministic_ops=true*) ;; *) echo "XLA_FLAGS must contain --xla_gpu_deterministic_ops=true (ladder.py asserts it): exit 3"; exit 3;; esac
if [ -z "$CPU_ONLY" ]; then nvidia-smi -L || { echo "no GPU visible: exit 3"; exit 3; }; fi
[ -d checkpoints/125m_ttt_e2e_pretrain_dclm_8k_1x_cc ] || { echo "checkpoint checkpoints/125m_ttt_e2e_pretrain_dclm_8k_1x_cc missing: exit 3"; exit 3; }
[ -f data/tokenizer/llama3/tokenizer.json ] || { echo "tokenizer data/tokenizer/llama3/tokenizer.json missing: exit 3"; exit 3; }
mkdir -p "$OUT/states"
AV=$(df -P --block-size=1G "$OUT" | awk 'NR==2{print $4}'); echo "free disk at $OUT: $AV GB (need >= 10)"
[ "$AV" -ge 10 ] || { echo "not enough free disk (exit 3 outside CPU_ONLY)"; [ -n "$CPU_ONLY" ] || exit 3; }
MARK=$(mktemp)
$PY $S/e49_stats.py | tail -1
$PY $S/e49_test.py | tail -1
$PY $S/e48_world_test.py | tail -1
CARGS=""; for C in w1rp_native=L0fs_w1rp w1_bg=S3_w1_bg w1_noo=S3_w1_noo w1_unifg=S3_w1_unifg; do
  N=${C%%=*}; P=${C#*=}
  echo "$N ($P): states A $(ls $OUT/states/${P}_s101/A 2>/dev/null | wc -l), B $(ls $OUT/states/${P}_s101/B 2>/dev/null | wc -l), saved log A: $([ -f $OUT/${P}_s101_A.saved.json ] && echo yes || echo NO)"
  CARGS="$CARGS --cell $C"
done
[ -n "$CPU_ONLY" ] && exit 0
[ -f $OUT/e49_candidates_s101_w1dec.json ] && [ -z "$RERUN" ] && { echo "summary e49_candidates_s101_w1dec.json exists: not overwritten (exit 8)"; exit 8; }
$PY $S/e49_candidates.py --seed 101 --epochs 1,2,3,4 --w0 --w0-name w1dec_W0 --ref w1rp_native --tag _w1dec --strict $CARGS
# provenance: pod environment and sha256 (first 64 hex) of every result file this job wrote (states: sizes + sha256 of the pass-4 states)
{ echo "job $JOB commit $COMMIT"; date -u; hostname; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; $PY -c "import jax;print('jax',jax.__version__,jax.devices())"; echo "XLA_FLAGS=$XLA_FLAGS"; } > "$OUT/rp_env_$JOB.txt" 2>&1 || true
(cd "$OUT" && find . -maxdepth 1 -type f -newer "$MARK" ! -name 'rp_sha_*' -exec sha256sum {} +) > "$OUT/rp_sha_$JOB.txt"
for T in ; do for W in A B; do F="$OUT/states/${T}_s101/$W/state_004.npz"; [ -f "$F" ] && { ls -l "$F"; sha256sum "$F"; }; done; done >> "$OUT/rp_sha_$JOB.txt" 2>&1 || true
echo "provenance: $OUT/rp_env_$JOB.txt, $OUT/rp_sha_$JOB.txt"

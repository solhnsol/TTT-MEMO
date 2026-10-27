#!/bin/bash
# JOB 283 = E57 T1/T2 CYCLE-2 EXPERIMENT (post-hoc, exploratory; Hans: T1/T2 up to cycle 2): ORC2 (g0, as job 282) and RAND2 written for two cycles from W0 with the weights saved at the end of cycle 1 and 2; NAT = the saved states of job 261 at chunks 279 / 559; OUT reads of the frozen 150 T1/T2 frames (hash e3897ebd...) at W0 and the six states, raw scores only (rows as job 276). NOT APPROVED yet. (Header of the copied job 282 follows)
set -eo pipefail
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder
D4=scripts/exp04_what_ttt_retains
COMMIT=e632178a94ca5143d31c247244ec0bde4927e06b
FILES=" $D/e55_run.py $D/e55_core.py $D/e55_gpu.py $D/e55_swap.py $D/e55_read.py $D/e55_analyze.py $D/e55_metrics.py $D/e55_ic_report.py $D/e55_iso.py $D/e55_diag.py $D/e55_votegrid.py $D/e55_uv.py $D/s3.py $D/rope_fly.py $D/e55_sentdump.py $D/e55_sentdump_labels.py $D/e55_sentsel.py $D/e55_sentgrid.py $D/e55_selmask.py $D/e55_gates.py $D/e55_bgbooks.py $D/e55_bgbooks_sha.json $D/e55_rev.py $D4/gen_facts.py $D4/gen_world_ctl.py $D4/frame_bank.py $D4/frames300.txt"
LADDER_OK="714b08ea3b90"
S3_OLD="f261120d0f03"
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
if [ -z "$NO_INSTALL" ]; then
  GL=$(sha256sum $D/ladder.py | cut -c1-12); [ "$GL" = "$LADDER_OK" ] || { echo "lab ladder.py ($GL) is not the patched W1 version ($LADDER_OK): exit 6"; exit 6; }
  git cat-file -e $COMMIT^{commit} || { echo "commit $COMMIT not in the local clone (exit 4)"; exit 4; }
  S3H=$(sha256sum $D/s3.py | cut -c1-12)
  S3OK=$( (echo "$S3_OLD"; echo a16e6dd0d4c9; for C in $(git log --format=%H --all -- $D/s3.py); do git show $C:$D/s3.py | sha256sum | cut -c1-12; done) )
  echo "$S3OK" | grep -qx "$S3H" || { echo "lab s3.py ($S3H) is not any committed version of s3.py: not overwritten (exit 6)"; exit 6; }
  git checkout $COMMIT -- $FILES
fi
if [ -z "$SKIP_VERIFY" ]; then
  while read HH F; do
    [ -n "$HH" ] || continue
    G=$(sha256sum $D/$F | cut -c1-12)
    [ "$G" = "$HH" ] || { echo "VERSION MISMATCH $F: have $G, expected $HH (exit 5)"; exit 5; }
  done <<'MANIFEST'
71f0fc416f74 e55_run.py
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
3d8657814bbb e55_bgbooks.py
75ff3cee6437 e55_bgbooks_sha.json
2215511cee2a e55_rev.py
b867b6f680f3 ../exp04_what_ttt_retains/gen_facts.py
1cf0caa82553 ../exp04_what_ttt_retains/gen_world_ctl.py
eeaaa9135907 ../exp04_what_ttt_retains/frame_bank.py
866eb5989467 ../exp04_what_ttt_retains/frames300.txt
MANIFEST
  echo "installed code verified against the manifest"
fi
# FIFTH VERSION (08:3xZ, E57 section 53): the segmented prefix is the reference method; stage -1 is only the stop condition on the values already measured by job 259 (rank changes <= 4 of 84, canary differences <= 0.01). THIRD VERSION (07:33Z): the full-stream prefix output of 6 cycles needs 77 GiB in one call and failed even without the fp32 reader; the arms stage now computes it in segments (280 chunks + 192 chunks of context, cached for the other arms) and stage -1 checks the segmented against the one-call prefix on 560 chunks. SECOND VERSION after an out-of-memory failure of the first attempt (06:02Z): two stages. EXPECTED RUN TIME (Hans confirms before it runs): ESTIMATE ONLY, unmeasured: stage 1 = model load 15 min + one record of the natural K,V of 1680 chunks (about 8 min, about 400 GiB host RAM, 35 s per 128 chunks in job 238) + the arms (NAT 30-40 min, three Adam arms 95-100 min each) + canary at 6 cycle ends; stage 2 = model load + fp32 reader 15 min + scorers 10 min + W0 reads + 24 reads x about 70 s = 30 min; about 6.5-7.5 h in total plus the paired check stage 0 (about 30 min: model load 15 min, 280 updates 5 min, scorers 10 min) = about 7-8 h, plus about 20 min for stage -1 and about 7 min per arm for the segmented prefix (once, cached) = about 7.5-8.5 h (stage -1 now about 45 min instead of 20 min: 8-9 h in total). 24 state files of 1.4 GB (34 GB) are written to $OUT/e57_main/states. The earlier estimate follows: From the measured trial (job 256: 280 updates + 13 canary readings 7 min, reads 61 s per read position after the scorers exist, scorer build about 4 min per context) and job 238 (masked Adam arm about 3.5 s per update): W0 reads about 55 min (10 IN scorers + 1 OUT scorer built once, shared by all arms), NAT 560 updates about 12 min, NATADAMH and ORC and RAND about 25-35 min each, reads 20 positions x 61 s = 20 min per arm, model load 7-20 min once, CPU oracle masks a few min = about 4-5 h in total. Code: e57_main_run.py was NOT run on a GPU (CPU selftest and --check-only only); the arm engines are the proven ones of jobs 233 \/ 238 (SELC1 via SelMask, RSELC1 via UV, Adam via the ADAM table with the added NATADAMH entry) but their use on the news stream (dummy chunk table, 560 chunks) is new: a failure at the first arm is possible. Commit 175d3f78cb8b886cdbc647ad45b0e3201bfcac4b. NOT APPROVED yet.
if [ -z "$NO_INSTALL" ]; then git cat-file -e 175d3f78cb8b886cdbc647ad45b0e3201bfcac4b^{commit} || { echo "commit 175d3f78cb8b886cdbc647ad45b0e3201bfcac4b missing (exit 4)"; exit 4; }; git checkout 175d3f78cb8b886cdbc647ad45b0e3201bfcac4b -- $D/e57_main_run.py $D/e57_main_report.py $D/e57_t1t2c2_run.py $D/e57_t1t2_run.py $D/e57_orca2_masks.py $D/e57_mask_stage.py $D/e57_attdump.py $D/e57/e57_orc2.py; git checkout c9e5e6a6b60adad98bf917d00cd19678d8c4c1a6 -- $D/e57_reads.py; git checkout c9e5e6a6b60adad98bf917d00cd19678d8c4c1a6 -- $D/e57_trial_run.py; git checkout 57c50b5ce5dee7e3efea0ba036c79ab4eb6e6f02 -- $D/e57_canary_run.py; git checkout 5c835409e9a32b77829fdb20e4022a6547f06705 -- $D/e56/e56_text.py $D/e57/e57_common.py; git checkout c9e5e6a6b60adad98bf917d00cd19678d8c4c1a6 -- $D/e57/e57_bank.py; fi
for x in "7c9a26918129 e57_t1t2c2_run.py" "3419b588f180 e57_t1t2_run.py" "8e0f84a16240 e57_orca2_masks.py" "cdc465c76bb0 e57_mask_stage.py" "7bc1b396e2d5 e57_attdump.py" "a1946090a14c e57/e57_orc2.py" "d13a0ac9ee4a e57_main_run.py" "a268feab4d3f e57_main_report.py" "b19e986904ba e57_reads.py" "9ee4658ee168 e57_trial_run.py" "fe17f085f622 e57_canary_run.py" "88076d1f3dd1 e57/e57_bank.py" "a0590cd53539 e57/e57_common.py" "f09dcb7e69eb e56/e56_text.py"; do set -- $x; G=$(sha256sum $D/$2 | cut -c1-12); [ "$G" = "$1" ] || { echo "VERSION MISMATCH $2 (exit 5)"; exit 5; }; done
S=$D
$PY $D4/gen_facts.py --selftest | tail -1
$PY $S/e55_swap.py | tail -1
$PY $S/e57_reads.py --selftest
$PY $S/e57_main_run.py --selftest | tail -1
$PY $S/e57/e57_orc2.py --selftest | tail -1
$PY $S/e57_t1t2c2_run.py --selftest | tail -1
$PY $S/e57_t1t2_run.py --selftest | tail -1
BUNDLE=${BUNDLE:-$OUT/e57_news};  MOUT=${MOUT:-$OUT/e57_t1t2c2}; FROZEN=${FROZEN:-$BUNDLE/e57_t1t2/t1t2_v1.json}; NATARMS=${NATARMS:-$OUT/e57_main/E57_main_NAT.arms.json}
[ -f "$BUNDLE/e57_bundle/bundle_store.pkl" ] && [ -f "$BUNDLE/e57_bank/bank_v1.json" ] || { echo "bundle store or frozen bank missing (exit 7)"; exit 7; }
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu $PY $S/e57_t1t2c2_run.py --cpu-check --frozen "$FROZEN" --nat-arms "$NATARMS" --out "$MOUT"
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu $PY $S/e57_t1t2c2_run.py --bundle "$BUNDLE" --out "$MOUT" --cycles 2 --npos 1 --canary-npos 4 --arms ORC2,RAND2 --check-only
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
[ -f "$MOUT/E57_t1t2c2_scores.json" ] && [ -z "$RERUN" ] && { echo "E57_t1t2c2_scores.json exists (exit 8; RERUN=1 to redo)"; exit 8; }
mkdir -p "$MOUT"
$PY $S/e57_orca2_masks.py --bundle "$BUNDLE" --out "$MOUT/e57_orca2_masks.npz" --cycles 2
ARGS="--bundle $BUNDLE --out $MOUT --cycles 2 --npos 1 --canary-npos 4 --arms ORC2,RAND2 --pod-masks $MOUT/e57_orca2_masks.npz --ntg-from $NATARMS --nat-arms $NATARMS --frozen $FROZEN --guard 0"
$PY $S/e57_t1t2c2_run.py $ARGS --stage arms
$PY $S/e57_t1t2c2_run.py $ARGS --stage reads
echo "outputs: $MOUT/E57_t1t2c2_scores.json $MOUT/E57_t1t2c2_score_keys.json (copy flat to the shared root as e57_t1t2c2_scores.json)"

#!/bin/bash
# JOB 288 = E57 NEWS TOKEN-LEVEL ATTENTION DUMP v2 (post-hoc, exploratory; Hans via the label-free-mask thread 03:15Z "value 보정용 덤프도 같이 준비시킬까 그러면 토큰 단위의 질량도 필요하고"; forward only, W0, no write): e57_attdump2_run.py saves for chunks 1..279 of the one-cycle news stream T_max, T_key, T_sum, T_ent, T_sink, T_rec, T_cur, position-binned window mass POS (9 bins) and key value norms VN per layer/head/token, same definitions as the synthetic dump, in 40-chunk shards, read-only, about 7 GiB, output experiments/exp07_write_ladder/e57_attdump2/. NOT APPROVED yet (needs Hans job name + commit + time). ESTIMATE ONLY, unmeasured: model load 15 min + record of the natural K,V 5 min + 279 chunks x about 12-15 s (job 278 measured 12 s per chunk with the AL reduction) = 56-70 min, about 1.3-1.6 h in total. (Header of the copied job 278 follows)  = E57 NEWS ATTENTION DUMP (post-hoc, exploratory; Hans: put into the lab queue; forward only, W0, no write): e57_attdump_run.py saves, for chunks 1..279 of the one-cycle news stream, the per-layer/head sentence attention mass AL [24,32,nQ,nW] (+ AL_chunk, event axis, token ids; layout of the seed-61 synthetic dump) in 40-chunk shards, read-only, about 4.7 GiB, output experiments/exp07_write_ladder/e57_attdump/. NOT APPROVED yet. (Header of the copied job 273 follows) E57 WRITE-TIME NLL CHECK (post-hoc, exploratory; Hans 16:08Z "확인 정도는 해봐도좋지", 16:10Z "1주기만"): NAT one cycle re-run with the per-token NLL before the write saved for every chunk (e57_wnll_run.py), then the CPU analysis by answer sentences (class A / B / C, exposure rank, gold-span tokens; e57_wnll_analysis.py). NOT APPROVED yet. (Header of the copied job 265 follows) continuation of job 262: the arms stage of 262 finished (4 arms, 24 states in e57_main_mid/states); its reads stage failed at the start with a NameError (my change: ids undefined in stage_reads, not exercised by the CPU check-only). This job runs ONLY --stage reads and the report on the saved states. NOT APPROVED yet. (Header of 262 follows) E57 MID-CYCLE READS (post-hoc, exploratory; Hans 13:33Z "1주기 안 읽기가 중요하겠네"): the four arms (NAT, NATADAMH, ORC, RAND) re-run for ONE cycle (280 chunks) with reads of the frozen bank at the in-cycle chunks 7,15,32,65,131,279 (about 1,2,4,8,16,34 exposures per question on average, to be checked against n_k printed by the run) and canary at the same chunks; output dir e57_main_mid (the states of job 261 in e57_main are not touched). NOT APPROVED yet. Copy of job 261 (original header follows): E57 MAIN RUN R = $CYCLES cycles, default 6 (Hans 05:54Z: no canary monitoring, canary values are only recorded at W0 and at the end of every cycle; no canary limit (E57 section 50): the report gives NLL, rise and perplexity ratio per cycle end) (GPU): the natural news stream in date order for 2 cycles (280 chunks each, one inner update per chunk) with four arms: NAT (SGD), NATADAMH (Adam direction, 0.5 x NAT norm, no mask = the same-size update control), ORC (Adam 0.5 x NAT norm + ORACLE answer mask: window sentences holding a gold answer are hidden; labels used), RAND (same hidden tokens per chunk, random sentences). Reads of the FROZEN bank bank_v1.json (OUT neutral tail; IN natural window with present / absent) at W0 and at the end of every cycle (one read per cycle), read-only; canary at W0 and at every read position (the first chunk over +0.10 is recorded, runs continue). Report e57_main_report.md with all-42 and without may-person-01 / may-person-03 tables and paired differences to W0 and NAT. First observation: 42 questions, 10 events.
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
# FIFTH VERSION (08:3xZ, E57 section 53): the segmented prefix is the reference method; stage -1 is only the stop condition on the values already measured by job 259 (rank changes <= 4 of 84, canary differences <= 0.01). THIRD VERSION (07:33Z): the full-stream prefix output of 6 cycles needs 77 GiB in one call and failed even without the fp32 reader; the arms stage now computes it in segments (280 chunks + 192 chunks of context, cached for the other arms) and stage -1 checks the segmented against the one-call prefix on 560 chunks. SECOND VERSION after an out-of-memory failure of the first attempt (06:02Z): two stages. EXPECTED RUN TIME (Hans confirms before it runs): ESTIMATE ONLY, unmeasured: stage 1 = model load 15 min + one record of the natural K,V of 1680 chunks (about 8 min, about 400 GiB host RAM, 35 s per 128 chunks in job 238) + the arms (NAT 30-40 min, three Adam arms 95-100 min each) + canary at 6 cycle ends; stage 2 = model load + fp32 reader 15 min + scorers 10 min + W0 reads + 24 reads x about 70 s = 30 min; about 6.5-7.5 h in total plus the paired check stage 0 (about 30 min: model load 15 min, 280 updates 5 min, scorers 10 min) = about 7-8 h, plus about 20 min for stage -1 and about 7 min per arm for the segmented prefix (once, cached) = about 7.5-8.5 h (stage -1 now about 45 min instead of 20 min: 8-9 h in total). 24 state files of 1.4 GB (34 GB) are written to $OUT/e57_main/states. The earlier estimate follows: From the measured trial (job 256: 280 updates + 13 canary readings 7 min, reads 61 s per read position after the scorers exist, scorer build about 4 min per context) and job 238 (masked Adam arm about 3.5 s per update): W0 reads about 55 min (10 IN scorers + 1 OUT scorer built once, shared by all arms), NAT 560 updates about 12 min, NATADAMH and ORC and RAND about 25-35 min each, reads 20 positions x 61 s = 20 min per arm, model load 7-20 min once, CPU oracle masks a few min = about 4-5 h in total. Code: e57_main_run.py was NOT run on a GPU (CPU selftest and --check-only only); the arm engines are the proven ones of jobs 233 \/ 238 (SELC1 via SelMask, RSELC1 via UV, Adam via the ADAM table with the added NATADAMH entry) but their use on the news stream (dummy chunk table, 560 chunks) is new: a failure at the first arm is possible. Commit 6bf5250047fe95da4404071bd3325de1711bb12d. NOT APPROVED yet.
if [ -z "$NO_INSTALL" ]; then git cat-file -e 6bf5250047fe95da4404071bd3325de1711bb12d^{commit} || { echo "commit 6bf5250047fe95da4404071bd3325de1711bb12d missing (exit 4)"; exit 4; }; git checkout 6bf5250047fe95da4404071bd3325de1711bb12d -- $D/e57_main_run.py $D/e57_main_report.py $D/e57_attdump_run.py $D/e57_attdump.py $D/e57_mask_stage.py; git checkout 770742cec1978b0ed881db8d7ab5e34a5c0b2f38 -- $D/e57_attdump2.py $D/e57_attdump2_run.py; git checkout c9e5e6a6b60adad98bf917d00cd19678d8c4c1a6 -- $D/e57_reads.py; git checkout c9e5e6a6b60adad98bf917d00cd19678d8c4c1a6 -- $D/e57_trial_run.py; git checkout 57c50b5ce5dee7e3efea0ba036c79ab4eb6e6f02 -- $D/e57_canary_run.py; git checkout 5c835409e9a32b77829fdb20e4022a6547f06705 -- $D/e56/e56_text.py $D/e57/e57_common.py; git checkout c9e5e6a6b60adad98bf917d00cd19678d8c4c1a6 -- $D/e57/e57_bank.py; fi
for x in "8c0a4eaa44b2 e57_attdump2.py" "13841794e9b4 e57_attdump2_run.py" "91568fc3bb10 e57_attdump_run.py" "7bc1b396e2d5 e57_attdump.py" "cdc465c76bb0 e57_mask_stage.py" "d13a0ac9ee4a e57_main_run.py" "a268feab4d3f e57_main_report.py" "b19e986904ba e57_reads.py" "9ee4658ee168 e57_trial_run.py" "fe17f085f622 e57_canary_run.py" "88076d1f3dd1 e57/e57_bank.py" "a0590cd53539 e57/e57_common.py" "f09dcb7e69eb e56/e56_text.py"; do set -- $x; G=$(sha256sum $D/$2 | cut -c1-12); [ "$G" = "$1" ] || { echo "VERSION MISMATCH $2 (exit 5)"; exit 5; }; done
S=$D
$PY $D4/gen_facts.py --selftest | tail -1
$PY $S/e55_swap.py | tail -1
$PY $S/e57_reads.py --selftest
$PY $S/e57_main_run.py --selftest | tail -1
$PY $S/e57_attdump2_run.py --selftest | tail -1
$PY $S/e57_mask_stage.py --selftest | tail -1
BUNDLE=${BUNDLE:-$OUT/e57_news}; MOUT=${MOUT:-$OUT/e57_attdump2}
[ -f "$BUNDLE/e57_bundle/bundle_store.pkl" ] && [ -f "$BUNDLE/e57_bank/bank_v1.json" ] || { echo "bundle store or frozen bank missing (exit 7)"; exit 7; }
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu $PY $S/e57_attdump2_run.py --bundle "$BUNDLE" --out "$MOUT" --check-only
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
[ -f "$MOUT/e57_attdump2_manifest.json" ] && [ -z "$RERUN" ] && { echo "e57_attdump2_manifest.json exists (exit 8; RERUN=1 to redo)"; exit 8; }
$PY $S/e57_attdump2_run.py --bundle "$BUNDLE" --out "$MOUT"
chmod -R a+rX "$MOUT"
if [ -f "$OUT/e57_attdump/e57_attdump_manifest.json" ]; then $PY $S/e57_attdump2_run.py --check-278 --old "$OUT/e57_attdump" --out "$MOUT" --chunks 1,17,33,65,129,257,279; else echo "job-278 dump ($OUT/e57_attdump) not present here: cross-check skipped; run e57_attdump2_run.py --check-278 where both dumps exist"; fi

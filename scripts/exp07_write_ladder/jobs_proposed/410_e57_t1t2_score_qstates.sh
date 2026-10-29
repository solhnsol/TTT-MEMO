#!/bin/bash
# JOB 410 = E57 T1/T2 OUT READS OF THE QUERY-BAND STATES (post-hoc; Hans 08:11Z decision card "401 바로 뒤, 연구실 GPU": job 410_e57_t1t2_score_qstates): the method of jobs 276 / 277 (e57_t1t2_run.py, commit 15666eb; frozen t1t2_v1.json, content hash e3897ebd…, 150 frames, OUT reads of the 8192-token SQuAD tail, fp32 read-only reader) at W0 (control) and at chunk 279 of GAINQ512 p=.26 (job 292b), GAINQ512 p=.10 (293), RANDQ512 p=.10 (297); the states are the existing ones (hash checked), no new state is written. Output $OUT/e57_t1t2_qstates/E57_t1t2_scores.json + E57_t1t2_score_keys.json (~4 MB). ESTIMATE ONLY: load 15 min + 4551 scorer rows ~30-40 min + 4 x ~4.5 min = ~65-75 min.
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

D=scripts/exp07_write_ladder; S=$D
if [ -z "$NO_INSTALL" ]; then git cat-file -e c697f01039387f2d16c48cfcf1600b3d145dc66e^{commit} || { echo "commit c697f01039387f2d16c48cfcf1600b3d145dc66e missing (exit 4)"; exit 4; }; git checkout c697f01039387f2d16c48cfcf1600b3d145dc66e -- $D/e57_t1t2_qstates_run.py; git checkout 15666eb -- $D/e57_t1t2_run.py; fi
for x in "3419b588f180 e57_t1t2_run.py" "d13a0ac9ee4a e57_main_run.py" "b19e986904ba e57_reads.py" "9ee4658ee168 e57_trial_run.py" "88076d1f3dd1 e57/e57_bank.py" "a0590cd53539 e57/e57_common.py" "f09dcb7e69eb e56/e56_text.py"; do set -- $x; G=$(sha256sum $D/$2 | cut -c1-12); [ "$G" = "$1" ] || { echo "VERSION MISMATCH $2: have $G, expected $1 (exit 5)"; exit 5; }; done
$PY $D4/gen_facts.py --selftest | tail -1
$PY $S/e55_swap.py | tail -1
$PY $S/e57_reads.py --selftest
$PY $S/e57_main_run.py --selftest | tail -1
$PY $S/e57_t1t2_run.py --selftest
$PY $S/e57_t1t2_qstates_run.py --selftest | tail -1
BUNDLE=${BUNDLE:-$OUT/e57_news}; TOUT=${TOUT:-$OUT/e57_t1t2_qstates}; FROZEN=${FROZEN:-$BUNDLE/e57_t1t2/t1t2_v1.json}; MANIFEST=${MANIFEST:-$TOUT/e57_t1t2_states_manifest.json}; STATES_FROM=${STATES_FROM:-$OUT/e57_main}
[ -f "$BUNDLE/e57_bundle/bundle_store.pkl" ] && [ -f "$BUNDLE/e57_bank/bank_v1.json" ] && [ -f "$FROZEN" ] || { echo "bundle store, bank_v1.json or the frozen file missing (exit 7)"; exit 7; }
mkdir -p "$TOUT"
$PY - "$OUT" "$MANIFEST" <<'PYE'
import json,sys
o,mf=sys.argv[1:3];m={}
for arm,f in (('GAINQ512P26',o+'/e57_gainq_p0.26/E57_main_GAINQ512.arms.json'),('GAINQ512P10',o+'/e57_gainq_p0.10/E57_main_GAINQ512.arms.json'),('RANDQ512P10',o+'/e57_randq_p0.10/E57_main_RANDQ512.arms.json')):
    st=json.load(open(f))['states']['279'];m[arm]=dict(file=st['file'],hash=st['hash'])
json.dump(m,open(mf,'w'),indent=1);print('manifest',mf,{k:v['hash'][:12] for k,v in m.items()})
PYE
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu $PY $S/e57_t1t2_run.py --cpu-check --frozen "$FROZEN" --manifest "$MANIFEST" --out "$TOUT"
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
[ -f "$TOUT/E57_t1t2_scores.json" ] && [ -z "$RERUN" ] && { echo "E57_t1t2_scores.json exists (exit 8; RERUN=1 to redo)"; exit 8; }
$PY $S/e57_t1t2_qstates_run.py --bundle "$BUNDLE" --out "$TOUT" --bank "$BUNDLE/e57_bank/bank_v1.json" --cycles 1 --npos 1 --canary-npos 1 --arms GAINQ512P26,GAINQ512P10,RANDQ512P10 --stage reads --frozen "$FROZEN" --manifest "$MANIFEST" ${STATE_DIR:+--state-dir "$STATE_DIR"}
echo "outputs: $TOUT/E57_t1t2_scores.json $TOUT/E57_t1t2_score_keys.json (copy flat to the shared root as e57_t1t2_scores.json / e57_t1t2_score_keys.json)"

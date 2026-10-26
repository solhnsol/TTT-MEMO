#!/bin/bash
# JOB 257 = E55 32.13 WRITE EXPERIMENT WITH LEARNED HEAD-WEIGHTING MASKS on the HELD-OUT seeds 62, 63, 64 (GPU). Coefficients of the logistic regression on the per-head sentence mass were trained WITH LABELS on seed 61 only (job 242) and are FROZEN (never refit). Per seed: (1) mask stage e55_lmask_run.py (forward only, frozen prefix attention, top 0.33 of the window tokens hidden by the learned score), (2) write arms SELC1 = learned masks and RSELC1 = random sentences with the same hidden tokens per chunk (tag lsel; the world, norm target, references and metrics of jobs 233 / 238: out-of-window a = 16, 32, 64, 128, in-window, canary, DCLM). References that already exist and are NOT rerun: oracle SENTPQ-ADAM-0.5N and NAT-ADAM (tag q1b), candidate 1 (tag sel). Pre-set rule (same as jobs 233 / 234 / 238 / 239): R = g(arm) / g(SENTPQ-ADAM-0.5N) at a = 128; WORKS if R >= 0.7 and g(SELC1) - g(RSELC1) >= +0.10; FAILS if R <= 0.3. Reports: e55_lsel_report_s62to64.md (learned) and e55_sel_report_s62to64.md (candidate 1, same seeds, from the existing files).
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
# EXPECTED RUN TIME (Hans confirms before it runs): ESTIMATE ONLY, unmeasured, GPU: per seed the mask stage (record of the natural K,V about 6-10 min + 127 attention-sentence passes, guess 15-40 min; the selmasks stage of job 231 took 28 min, this pass keeps more, so up to about 1.5x) plus model load 7-20 min, then the two write arms (job 238 measured about 7.5 min per arm after loading) plus a second model load = about 1.0-1.7 h per seed, about 3-5 h for the three seeds. The mask stage (e55_lmask_run.py) was NOT run on a GPU (CPU selftest and --check-only only): a failure at its first GPU call is possible. Commit 251cbf60db0f813225b334ee18086463fbdce0db. NOT APPROVED yet.
if [ -z "$NO_INSTALL" ]; then git cat-file -e 251cbf60db0f813225b334ee18086463fbdce0db^{commit} || { echo "commit 251cbf60db0f813225b334ee18086463fbdce0db missing (exit 4)"; exit 4; }; git checkout 251cbf60db0f813225b334ee18086463fbdce0db -- $D/e55_lmask_run.py; git checkout 9b9364319dde87d5366ac7c725e645c326f1ba8b -- $D/e55_sentgrid_ext.py; fi
for x in "a298ad64127f e55_lmask_run.py" "7de0dd3b97f0 e55_sentgrid_ext.py"; do set -- $x; G=$(sha256sum $D/$2 | cut -c1-12); [ "$G" = "$1" ] || { echo "VERSION MISMATCH $2 (exit 5)"; exit 5; }; done
S=$D
SEEDS="${SEEDS:-62,63,64}"; FRAC=0.33
COEF=$OUT/e55_sentgrid_ext_s61_coef.npz; [ -f $COEF ] || { echo "frozen coefficients $COEF missing (job 242): exit 7"; exit 7; }
NORM="$OUT/E55_v3a_NAT_s{seed}.saved.json"
CANDJ=$OUT/e55_lmask_candidates.json
DCLM=data/dclm_val_subset/val_c0_200x8193.u32le.bin
$PY -c "import json,sys;json.dump([{'name':'LEARNED head weighting (frozen seed-61 coefficients NAT/sum/all/100, top %s of the window tokens)'%sys.argv[1],'learned':True,'frac':float(sys.argv[1])}],open(sys.argv[2],'w'))" $FRAC $CANDJ
DEC="--cell v3_16_1 --tag lsel --norm-ref-tag q1b --bg pool --out $OUT --seeds $SEEDS --arms SELC1,RSELC1 --norm-files $NORM --decomp-apps 16,32,64,128 --in-window --in-apps 128 --canary-all --sel-masks $OUT/e55_lmasks_s{seed}.npz --sel-cands $CANDJ"
$PY $D4/gen_facts.py --selftest | tail -1
$PY $S/e55_read.py --selftest | tail -1
$PY $S/e55_swap.py | tail -1
$PY $S/e55_metrics.py --selftest | tail -1
$PY $S/e55_ic_report.py --selftest | tail -1
$PY $S/e55_lmask_run.py --selftest | tail -1
for SD in $(echo $SEEDS | tr ',' ' '); do
  $PY $S/e55_gates.py --cell v3_16_1 --seed $SD --default-refs --require-refs --out $OUT/e55_gates_lsel_s$SD.md > $OUT/e55_gates_lsel_s$SD.log 2>&1 || { tail -30 $OUT/e55_gates_lsel_s$SD.log; echo "background gates failed for seed $SD (exit 7)"; exit 7; }
  tail -2 $OUT/e55_gates_lsel_s$SD.md
  for F in v3a_NAT q1b_SENTPQ q1b_SENTPQADAMH q1b_NATADAM sel_SELC1 sel_RSELC1; do [ -f $OUT/E55_${F}_s$SD.saved.json ] || { echo "file E55_${F}_s$SD.saved.json missing (exit 7)"; exit 7; }; done
  $PY $S/e55_lmask_run.py --seed $SD --out $OUT --coef $COEF --frac $FRAC --check-only | tail -1
done
$PY $S/e55_run.py decomp $DEC --check-only
[ -f "$DCLM" ] && echo "DCLM ids present: $(stat -c %s $DCLM) bytes" || echo "DCLM ids file absent here (needed on the GPU run)"
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
[ -f "$DCLM" ] || { echo "DCLM ids file $DCLM missing (exit 7)"; exit 7; }
REP=$OUT/e55_lsel_report_s62to64.md
if [ -f $REP ] && [ -z "$RERUN" ]; then echo "report $REP exists: not redone (exit 8; RERUN=1 to redo)"; exit 8; fi
for SD in $(echo $SEEDS | tr ',' ' '); do
  echo "=== mask stage seed $SD"; $PY $S/e55_lmask_run.py --seed $SD --out $OUT --coef $COEF --frac $FRAC
done
$PY $S/e55_run.py decomp $DEC --dclm-ids $DCLM --dclm-n 100
$PY $S/e55_ic_report.py --uv --dir $OUT --seeds $SEEDS --tag lsel --ref-tag q1b --sel-ranks 1 --out $REP
$PY $S/e55_ic_report.py --uv --dir $OUT --seeds $SEEDS --tag sel --ref-tag q1b --sel-ranks 1 --out $OUT/e55_sel_report_s62to64.md
tail -40 $REP

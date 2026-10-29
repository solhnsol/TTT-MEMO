#!/bin/bash
# JOB 460 = E57 PUBLIC QA STAGE 2 (E57 section 122-P, 20:50 addendum; prepared, NOT APPROVED to run, needs Hans's new approval line): the read-only W0 reads of job 440 (OUT / IN, same tail, same E49 scores, e57_squad_w0_reads.squad_reads unchanged) for 1,000 HotpotQA-distractor-dev items (items sha256 458177aa…, content hash c993393b) and 448 MuSiQue-Ans-dev items (bfaf4b2c…, 291351ba), 16 candidates each, IN = the supporting paragraphs joined at the end of the tail. No state written. Output $OUT/e57_multihop_w0/E57_squad_w0_reads.json + E57_squad_w0_keys.json (~22 MB; file names as in job 440, written by the same code). IN scorer: KV-reuse (job 447: 3.1 s instead of 24.4 s per item) if $OUT/e57_kvreuse_check/PASSED exists, else the reference path (the estimate below is for the reference path; with KV-reuse the IN part is ~1.3 h and the total ~3.7-4.5 h). ESTIMATE ONLY (scaled from the job 440 estimate by 23,168 / 16,000 rows): load 15 min + OUT 23,168 rows scorer prep ~3.2-3.9 h + OUT reads ~22 min + IN 1,448 contexts (~2-4 s each + 16 rows) ~1.2-2 h = ~4.9-6.5 h.
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
if [ -z "$NO_INSTALL" ]; then git cat-file -e e7caf2f273d53a86f159126ae4fb7e72db666c2a^{commit} && git cat-file -e f1b46a0e4d147059e3bd280173e2f2df4825d5bb^{commit} || { echo "commit e7caf2f or f1b46a0e4d147059e3bd280173e2f2df4825d5bb missing (exit 4)"; exit 4; }; git checkout e7caf2f273d53a86f159126ae4fb7e72db666c2a -- $D/e57_squad_w0_reads.py; git checkout f1b46a0e4d147059e3bd280173e2f2df4825d5bb -- $D/e57_multihop_w0_reads.py $D/e57_public_qa_build_multihop.py; git cat-file -e 27345beb69201270ef174975e9f9211215af514b^{commit} && git checkout 27345beb69201270ef174975e9f9211215af514b -- $D/e57_fastin_reads.py $D/e57_squad_w0_resume.py $D/e57_kvreuse.py; fi
for x in "3419b588f180 e57_t1t2_run.py" "d13a0ac9ee4a e57_main_run.py" "b19e986904ba e57_reads.py" "9ee4658ee168 e57_trial_run.py" "88076d1f3dd1 e57/e57_bank.py" "a0590cd53539 e57/e57_common.py" "f09dcb7e69eb e56/e56_text.py"; do set -- $x; G=$(sha256sum $D/$2 | cut -c1-12); [ "$G" = "$1" ] || { echo "VERSION MISMATCH $2: have $G, expected $1 (exit 5)"; exit 5; }; done
$PY $D4/gen_facts.py --selftest | tail -1
$PY $S/e55_swap.py | tail -1
$PY $S/e57_reads.py --selftest
$PY $S/e57_main_run.py --selftest | tail -1
$PY $S/e57_multihop_w0_reads.py --selftest | tail -1
$PY $S/e57_fastin_reads.py --selftest | tail -1
[ $(sha256sum $D/e57_public_qa_build_multihop.py | cut -c1-8) = d64ff653 ] || { echo 'multihop builder differs from the project-library copy d64ff653 (exit 5)'; exit 5; }
BUNDLE=${BUNDLE:-$OUT/e57_news}; TOUT=${TOUT:-$OUT/e57_multihop_w0}; IH_=$OUT/e57_public_qa/e57_public_qa_hotpot_items.json; IM_=$OUT/e57_public_qa/e57_public_qa_musique_items.json; ITEMS=${ITEMS:-$IH_,$IM_}
[ -f "$BUNDLE/e57_bundle/bundle_store.pkl" ] && [ -f "$BUNDLE/e57_bank/bank_v1.json" ] && [ -f "$IH_" ] && [ -f "$IM_" ] || { echo "bundle store, bank_v1.json or an items file missing (exit 7)"; exit 7; }
HH=$(sha256sum "$IH_" | cut -c1-64); MH=$(sha256sum "$IM_" | cut -c1-64); echo "hotpot items sha256 $HH"; echo "musique items sha256 $MH"
[ "$HH" = "458177aa3a2fb50dc36289ae4dccddf5738653cb97f9129753404049d2d66776" ] && [ "$MH" = "bfaf4b2cd9e1341bcb9e266e0a5803b0ca41c863579e7be596938bef2d6a8438" ] || { echo "items files differ from the frozen copies (exit 9)"; exit 9; }
mkdir -p "$TOUT"; CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu $PY $S/e57_main_run.py --bundle "$BUNDLE" --out "$TOUT" --cycles 1 --arms NAT --check-only
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu $PY -c "import sys;sys.path.insert(0,'$S');import e57_multihop_w0_reads as Q;it,h=Q.load_multihop('$ITEMS');print('items file ok',len(it),h[:12])"
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
[ -f "$TOUT/E57_squad_w0_reads.json" ] && [ -z "$RERUN" ] && { echo "E57_squad_w0_reads.json exists (exit 8; RERUN=1 to redo)"; exit 8; }
$PY $S/e57_fastin_reads.py --multihop --fast-if-exists "$OUT/e57_kvreuse_check/PASSED" --bundle "$BUNDLE" --out "$TOUT" --bank "$BUNDLE/e57_bank/bank_v1.json" --cycles 1 --npos 1 --canary-npos 1 --arms NAT --items-file "$ITEMS" ${NITEMS:+--n-items $NITEMS} --stage reads
echo "outputs: $TOUT/E57_squad_w0_reads.json $TOUT/E57_squad_w0_keys.json"


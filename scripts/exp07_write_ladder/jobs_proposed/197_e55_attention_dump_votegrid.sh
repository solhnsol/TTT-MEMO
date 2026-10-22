#!/bin/bash
# E55 31 attention dump for the voting-mask grid (FORWARD ONLY, no backward, no weight change by the dump): same setting as job 193 (world v3_16_1, NAT stream, seed 61, chunks 16, 32, 64, 127, weights BEFORE the write of the chunk, prefix layers read through the s3 mode "att"), but the raw material is SAVED for the CPU: per layer group (all, g1, g2, g3) the per-query top-8 window keys (index, attention value) of the head- and layer-averaged row and the attention mass per fact over all 1024 keys; per layer and head the per-query top-8 window keys; the per-key labels (fact id, global sentence id, name / place / occupation bits). Then, on the CPU, e55_votegrid.py computes the grid n in {8, 16, 32} x c in {2, 4} for the sources avg (design definition: every trigger votes for [j-n, j+n], hidden when votes >= c x the random level), heads and mass (box-filter approximation) and the layer groups g3, all, g1, g2: hidden share, fact-token coverage, answer-token coverage (name / place / occupation), precision, sentences >= 50% covered, same-fact earlier-sentence coverage, and the pass rule of design 31.3 (fixed before the results).
# Outputs (NEW names): $OUT/e55_attdump_s61.npz (+ e55_attdump_s61.json meta), $OUT/e55_votegrid_report.md; the run is skipped if the npz exists (RERUN=1 to redo); exit 8 if the grid report exists unless RERUN=1.
# FILE SIZE (estimate from the array shapes, 24 prefix layers, 32 heads assumed, about 130 fact query rows per chunk; compressed npz is smaller): per chunk about 9 MB per-head tops (layers x heads x rows x 8 x 4 bytes) + 2.4 MB masses = about 11 MB; 4 chunks about 45 MB uncompressed, 20-45 MB compressed.
# NEEDS an s3.py change (the att mode now also returns per-head top-m; the lab s3.py must be f261120d0f03, a16e6dd0d4c9 or already the new version, otherwise exit 6) and the files e55_diag.py (make_dump_fn), e55_votegrid.py.
# EXPECTED RUN TIME (Hans confirms before it runs): ESTIMATE ONLY, unmeasured. About 25-45 min (the same steps as job 193, plus the per-head top-8 (a device top_k per layer) and the host copies: CPU part 5-10 min; model load 7-20 min; the NAT stream of 128 chunks with the 30 GiB K,V recording about 6-10 min; 4 eager diagnostic forwards about 1-3 min each); the grid itself runs on the CPU in under a minute. The design estimate was 10-20 min of GPU; the load time is not included in it. No reads, no canary, no GPU unit check (U38 on the tiny model: the run with the dump == the plain run bitwise, mass and tops consistent with the 193 diagnostic rows, labels match the table).
# CPU-only dry run (validates the arguments through attdump --check-only, the real-tokenizer name / place / occupation span search through maskdump and the grid script selftest): git fetch origin claude/project-thread-rwhdac && git worktree add --detach /tmp/dry_dump FETCH_HEAD && ln -s "$(pwd)/data" /tmp/dry_dump/data && cd /tmp/dry_dump && PY=/abs/path/python CPU_ONLY=1 bash scripts/exp07_write_ladder/jobs_proposed/197_e55_attention_dump_votegrid.sh   (then git worktree remove --force /tmp/dry_dump)
# PY must be an ABSOLUTE path to the lab python. PINNED INSTALL, NO FETCH (exit 4 if the commit is absent, exit 5 on a sha256 mismatch). The lab ladder.py must be the patched W1 version (exit 6).
# Queue position: any time (needs the NAT file of job 181 seed 61 only for the gates check). NOT APPROVED yet.
set -eo pipefail
PY=$(realpath -s "$PY")   # the runner exports a relative PY (.venv/bin/python): make it absolute BEFORE the cd below, WITHOUT resolving symlinks (uv venv; readlink -f breaks it)
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder
D4=scripts/exp04_what_ttt_retains
COMMIT=87b73152ceffc28e6ba83215018541d8566ee72b
FILES="$D/e55_run.py $D/e55_core.py $D/e55_gpu.py $D/e55_swap.py $D/e55_read.py $D/e55_analyze.py $D/e55_metrics.py $D/e55_ic_report.py $D/e55_iso.py $D/e55_diag.py $D/e55_votegrid.py $D/s3.py $D4/gen_facts.py $D4/gen_world_ctl.py $D4/frame_bank.py $D/e55_gates.py $D4/frames300.txt"
LADDER_OK="714b08ea3b90"
S3_OLD="f261120d0f03"   # lab s3.py after job 189 (2-D mask version); the pre-189 version a16e6dd0d4c9 is accepted too
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
if [ -z "$NO_INSTALL" ]; then
  GL=$(sha256sum $D/ladder.py | cut -c1-12); [ "$GL" = "$LADDER_OK" ] || { echo "lab ladder.py ($GL) is not the patched W1 version ($LADDER_OK): exit 6"; exit 6; }
  git cat-file -e $COMMIT^{commit} || { echo "commit $COMMIT not in the local clone: git fetch origin claude/project-thread-rwhdac first (exit 4)"; exit 4; }
  S3NEW=$(git show $COMMIT:$D/s3.py | sha256sum | cut -c1-12); S3H=$(sha256sum $D/s3.py | cut -c1-12)
  [ "$S3H" = "$S3_OLD" ] || [ "$S3H" = "a16e6dd0d4c9" ] || [ "$S3H" = "$S3NEW" ] || { echo "lab s3.py ($S3H) is neither the committed old version ($S3_OLD) nor the new one ($S3NEW): not overwritten (exit 6)"; exit 6; }
  git checkout $COMMIT -- $FILES
fi
if [ -z "$SKIP_VERIFY" ]; then
  while read HH F; do
    [ -n "$HH" ] || continue
    G=$(sha256sum $D/$F | cut -c1-12)
    [ "$G" = "$HH" ] || { echo "VERSION MISMATCH $F: have $G, expected $HH (exit 5)"; exit 5; }
  done <<'MANIFEST'
13a1fbdedc2a e55_run.py
8422d0ae3278 e55_core.py
9827e34dcde6 e55_gpu.py
f240fac19f43 e55_swap.py
a6d506949071 e55_read.py
4a811650bcd8 e55_analyze.py
30769e241d5e e55_metrics.py
4c8679add232 e55_ic_report.py
c4c8a0d4e386 e55_iso.py
725d18d16b65 e55_diag.py
597294ca312b e55_votegrid.py
558d7974f051 s3.py
f14491a9ddf7 ../exp04_what_ttt_retains/gen_facts.py
1cf0caa82553 ../exp04_what_ttt_retains/gen_world_ctl.py
eeaaa9135907 ../exp04_what_ttt_retains/frame_bank.py
ee82bf19b0b3 e55_gates.py
866eb5989467 ../exp04_what_ttt_retains/frames300.txt
MANIFEST
  echo "installed code verified against the manifest"
fi
S=$D
$PY $D4/gen_facts.py --selftest | tail -1
$PY $S/e55_read.py --selftest | tail -1
$PY $S/e55_swap.py | tail -1
$PY $S/e55_metrics.py --selftest | tail -1
$PY $S/e55_ic_report.py --selftest | tail -1
$PY $S/e55_votegrid.py --selftest
for SD in 61; do
  $PY $S/e55_gates.py --cell v3_16_1 --seed $SD --default-refs --require-refs --out $OUT/e55_gates_ad_s$SD.md > $OUT/e55_gates_ad_s$SD.log 2>&1 || { tail -30 $OUT/e55_gates_ad_s$SD.log; echo "background gates failed for seed $SD (exit 7)"; exit 7; }
  tail -4 $OUT/e55_gates_ad_s$SD.md
  [ -f $OUT/E55_v3a_NAT_s$SD.saved.json ] || [ -n "$CPU_ONLY" ] || { echo "NAT file of job 181 missing for seed $SD (exit 7)"; exit 7; }
done
$PY $S/e55_run.py dry --cell v3_16_1 --seed 61 --bg pool --arms NAT | tail -3
$PY $S/e55_run.py attdump --cell v3_16_1 --seed 61 --bg pool --out $OUT --diag-chunks 16,32,64,127 --check-only
$PY $S/e55_run.py maskdump --cell v3_16_1 --seed 61 --bg pool > $OUT/e55_maskdump_dump.json
grep -A6 'answer_span_miss' $OUT/e55_maskdump_dump.json || true
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
REP=$OUT/e55_votegrid_report.md
if [ -f $REP ] && [ -z "$RERUN" ]; then echo "report $REP exists: not redone (exit 8; RERUN=1 to redo)"; exit 8; fi
$PY $S/e55_run.py attdump --cell v3_16_1 --seed 61 --bg pool --out $OUT --diag-chunks 16,32,64,127
ls -l $OUT/e55_attdump_s61.npz
$PY $S/e55_votegrid.py --npz $OUT/e55_attdump_s61.npz --out $REP
tail -60 $REP

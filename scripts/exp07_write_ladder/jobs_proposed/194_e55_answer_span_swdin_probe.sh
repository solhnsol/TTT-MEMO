#!/bin/bash
# E55 design 28.3 / 28.6 / 28.7 bundle (v3_16_1 = the 181 world: 16 facts, 128 chunks = 128 exposures, 300-frame bank, pooled background; seeds 61 62 63; window kept; per-query convention like SENTPQ; reads at a = 64 and 128).
# PART A (decomp, tag ans): arms that hide the earlier fact sentences ACROSS ALL facts only partly: OSPAN = name AND place tokens (O-ALL), PSPAN = place tokens only (P-ALL), NSPAN = name tokens only (N-ALL). Report: g out-of-window, R = g / g(SENTPQ of job 190), hidden keys as % of the window; contrasts O/P/N-ALL - SENTPQ and P+N versus O. References are read from the files of jobs 181 (NAT, SWD), 189 (SENT) and 190 (SENTPQ).
# PART A2 (E55 30.3 recall dose-response, same decomp call): OSPANR25 / OSPANR50 / OSPANR75 = O-ALL with only a token-wise random 25 / 50 / 75 % of the name+place answer tokens of all earlier fact sentences hidden (fixed draws, nested, same hidden set for every fact-sentence query; background queries see everything). The report adds the dose-response line (g and R versus the hidden share of the answer tokens). Drop them with ARMS=OSPAN,PSPAN,NSPAN.
# PART B (run, tag swdin, decoy namesin): SWD-IN = SWD of job 181 (fake window, same chunks) but the fake PLACES are drawn from the real candidate pool by a length-preserving derangement of the other facts' places; the fake names are unchanged. Report adds SWD-IN - SWD, SWD - SENT, SWD-IN - SENT.
# PART C (probe, seed 61): one-step binding probe. From the seed-61 NAT states before the write of the chunks 32, 64, 127 and for each window condition (NAT, KMASK, SENTO, SENT, SWD, O-ALL, P-ALL, N-ALL) the write step of that chunk is computed, rescaled to one fixed norm (sum of the last 8 NAT update norms), applied to a COPY (not committed) and the change of PM3 (gold - mean of the other 15 candidates) on unseen question frames is read: 24 reads. The verdict thresholds in the probe report are an analogue only (design thread owns the rule).
# EXPECTED RUN TIME (Hans confirms before it runs): ESTIMATE ONLY, unmeasured. About 3.1-4.2 h in total (2-2.7 h without the OSPAN-R arms): part A 3 arms x 3 seeds = 9 runs, each 128 chunks + 30 GiB K,V recording about 6-10 min plus 2 reads about 1-2 min = about 1.2-1.8 h; part A2 (OSPAN-R25/50/75) 9 more runs of the same kind = about 1.2-1.8 h (design estimate 1-1.2 h); part B 3 runs (SWD-IN, same cost as SWD, with the fake stream) about 20-25 min; part C the NAT stream of seed 61 about 6-10 min + 24 reads about 16-19 min; model loads 7-20 min per process (three processes). Not run on a GPU: reads, canary, probe, the 3B paths.
# Outputs (NEW names): $OUT/E55_ans_*, $OUT/E55_swdin_*, $OUT/e55_ans_report.md, $OUT/e55_probe_s61.json, $OUT/e55_probe_report.md; exit 8 if the report exists unless RERUN=1. Nothing is overwritten.
# CPU-only dry run (validates the arms through decomp --check-only, the namesin decoy through dry, the probe arguments through probe --check-only and the real-tokenizer answer-span search through maskdump): git fetch origin claude/project-thread-rwhdac && git worktree add --detach /tmp/dry_ans FETCH_HEAD && ln -s "$(pwd)/data" /tmp/dry_ans/data && cd /tmp/dry_ans && PY=/abs/path/python CPU_ONLY=1 bash scripts/exp07_write_ladder/jobs_proposed/194_e55_answer_span_swdin_probe.sh   (then git worktree remove --force /tmp/dry_ans)
# PY must be an ABSOLUTE path to the lab python. PINNED INSTALL, NO FETCH (exit 4 if the commit is absent, exit 5 on a sha256 mismatch). Lab ladder.py must be the patched W1 version (exit 6); lab s3.py must be f261120d0f03, a16e6dd0d4c9 or the new version (exit 6).
# Needs: NAT/SWD files of job 181 (tag v3a), SENT and SENTPQ files of jobs 189/190 (tag dc) for the report. NOT APPROVED yet.
set -eo pipefail
PY=$(realpath -s "$PY")   # the runner exports a relative PY (.venv/bin/python): make it absolute BEFORE the cd below, WITHOUT resolving symlinks (uv venv; readlink -f breaks it)
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder
D4=scripts/exp04_what_ttt_retains
COMMIT=7f24c6cb670eff626cc2ada0a47030f0bc4c8207
FILES="$D/e55_run.py $D/e55_core.py $D/e55_gpu.py $D/e55_swap.py $D/e55_read.py $D/e55_analyze.py $D/e55_metrics.py $D/e55_ic_report.py $D/e55_iso.py $D/e55_diag.py $D/s3.py $D4/gen_facts.py $D4/gen_world_ctl.py $D4/frame_bank.py $D/e55_gates.py $D4/frames300.txt"
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
35c0bf2ff28f e55_run.py
73cafbbf4177 e55_core.py
9827e34dcde6 e55_gpu.py
04067fefb630 e55_swap.py
a6d506949071 e55_read.py
4a811650bcd8 e55_analyze.py
30769e241d5e e55_metrics.py
e47bf9c370ca e55_ic_report.py
c4c8a0d4e386 e55_iso.py
a6ceb1237d78 e55_diag.py
8111393db79e s3.py
f14491a9ddf7 ../exp04_what_ttt_retains/gen_facts.py
1cf0caa82553 ../exp04_what_ttt_retains/gen_world_ctl.py
eeaaa9135907 ../exp04_what_ttt_retains/frame_bank.py
ee82bf19b0b3 e55_gates.py
866eb5989467 ../exp04_what_ttt_retains/frames300.txt
MANIFEST
  echo "installed code verified against the manifest"
fi
S=$D
SEEDS="${SEEDS:-61,62,63}"
ARMS="${ARMS:-OSPAN,PSPAN,NSPAN,OSPANR25,OSPANR50,OSPANR75}"
NORM="$OUT/E55_v3a_NAT_s{seed}.saved.json"
$PY $D4/gen_facts.py --selftest | tail -1
$PY $S/e55_read.py --selftest | tail -1
$PY $S/e55_swap.py | tail -1
$PY $S/e55_metrics.py --selftest | tail -1
$PY $S/e55_ic_report.py --selftest | tail -1
for SD in $(echo $SEEDS | tr ',' ' '); do
  $PY $S/e55_gates.py --cell v3_16_1 --seed $SD --default-refs --require-refs --out $OUT/e55_gates_ans_s$SD.md > $OUT/e55_gates_ans_s$SD.log 2>&1 || { tail -30 $OUT/e55_gates_ans_s$SD.log; echo "background gates failed for seed $SD (exit 7)"; exit 7; }
  tail -4 $OUT/e55_gates_ans_s$SD.md
  [ -f $OUT/E55_v3a_NAT_s$SD.saved.json ] || [ -n "$CPU_ONLY" ] || { echo "NAT file of job 181 missing for seed $SD (exit 7)"; exit 7; }
done
$PY $S/e55_run.py dry --cell v3_16_1 --seed 61 --bg pool --decoy namesin --arms NAT,SWD | tail -3
$PY $S/e55_run.py decomp --cell v3_16_1 --tag ans --bg pool --out $OUT --seeds $SEEDS --arms $ARMS --norm-files "$NORM" --check-only
$PY $S/e55_run.py probe --cell v3_16_1 --seed 61 --bg pool --out $OUT --diag-chunks 32,64,127 --check-only
$PY $S/e55_diag.py --selftest
$PY $S/e55_run.py maskdump --cell v3_16_1 --seed 61 --bg pool > $OUT/e55_maskdump_ans.json
grep -A16 'answer_span_miss' $OUT/e55_maskdump_ans.json || true
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
REP=$OUT/e55_ans_report.md
if [ -f $REP ] && [ -z "$RERUN" ]; then echo "report $REP exists: not redone (exit 8; RERUN=1 to redo)"; exit 8; fi
$PY $S/e55_run.py decomp --cell v3_16_1 --tag ans --bg pool --out $OUT --seeds $SEEDS --arms $ARMS --norm-files "$NORM"
$PY $S/e55_run.py run --cell v3_16_1 --bg pool --tag swdin --out $OUT --decoy namesin --save-apps 128 --canary-apps 128 --seeds $SEEDS --arms SWD --plan "61=64,128|128;62=64,128|128;63=64,128|128"
$PY $S/e55_ic_report.py --ans --dir $OUT --seeds $SEEDS --out $REP
tail -50 $REP
$PY $S/e55_run.py probe --cell v3_16_1 --seed 61 --bg pool --out $OUT --diag-chunks 32,64,127
tail -40 $OUT/e55_probe_report.md

#!/bin/bash
# E55 design v3 (section 13), cell 1: 16 facts, each fact once in EVERY chunk, 128 chunks (131k tokens), 300-frame bank (128 distinct frames per fact), non-repeating pooled background (four books, deduplicated, seed-specific order). Arms NAT and SWD (window slots replaced by the same chunks with fake names / places; the written chunk is real), new seeds 61, 62, 63.
# EXPECTED RUN TIME (Hans confirms before it runs): ESTIMATE ONLY, NOT MEASURED on this cell. About 1.7-2.7 h on one A100: CPU part (selftests, 3 background gate checks, dry run) 10-20 min; GPU unit checks 20-30 min (model load 13-23 min + 12 chunks); ONE run process for 3 seeds x 2 arms (model + fp32 reader load 20-35 min once, then per seed and arm 128 chunks at about 1.3 s (NAT, measured in 180a) to 2 s (SWD, guess) = 3-5 min plus the frozen-weight reads: W0, out-of-window after exposure 16, 32, 64, 128 and the in-window read at 128, 30-90 s each (assumed) = 5-8 min). Host memory about 30 GiB per stream x 2 streams (real, fake) = 60 GiB (estimate; the lab host has 1.2 TB).
# Reads (frozen weights, 16 candidates = gold + 15 places of other facts, chance mean rank 8.5; question frame = fact id mod 3; 4 control items): out-of-window (neutral 8192-token context) after exposure 16, 32, 64, 128 and in-window (the 8 real chunks that precede the question) at exposure 128. Verdict point = exposure 128, out-of-window; rules 11b-11h unchanged (threshold 0.15 u, every seed S > 0, hold ratio 1.15).
# The background pool is filtered before use: paragraphs with a repeated unit, a 13-gram already kept elsewhere, or a 13-gram of the SQuAD readout context / canary books / frame bank / question frames are dropped (then seed-specific shuffle). Gates before any GPU step (exit 7 on failure): background unit reuse 0, 13-gram sharing between units 0, 13-gram overlap with frame bank / question frames / SQuAD readout context / canary books pg11, pg1342 = 0 (a reference file that is missing = failure here), B-arm filler spare, need / available >= 1.5.
# Outputs: $OUT/E55_v3a_<ARM>_s<SEED>.saved.json (+ states/ at exposure 128), .read.json (+ .read.partial.json while running), e55_gates_v3a_s<SEED>.md, e55_report_v3a.md, e55_verdict_v3a.md, E55_unit_v3a.json, e55_dry_v3a.json. NO OVERWRITE: finished files are reused (restart-safe); exit 8 if the report exists unless RERUN=1.
# CPU-only dry run: git fetch origin claude/project-thread-rwhdac && git worktree add --detach /tmp/dry_e55v3 FETCH_HEAD && ln -s "$(pwd)/data" /tmp/dry_e55v3/data && cd /tmp/dry_e55v3 && PY=/abs/path/python CPU_ONLY=1 bash scripts/exp07_write_ladder/jobs_proposed/181_e55_v3_cell1_128chunks.sh   (then git worktree remove --force /tmp/dry_e55v3)
# PY must be an ABSOLUTE path to the lab python. PINNED INSTALL, NO FETCH: code checked out of the local clone at $COMMIT (exit 4 if absent), sha256 first 12 hex (exit 5). The lab ladder.py must be the patched W1 version (exit 6). s3.py is the lab's own.
# Queue position: after the running jobs (file name 181).
set -eo pipefail
PY=$(realpath -s "$PY")   # the runner exports a relative PY (.venv/bin/python): make it absolute BEFORE the cd below, WITHOUT resolving symlinks (uv venv; readlink -f breaks it)
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
case "$PY" in /*) ;; *) echo "PY must be an absolute path (got '$PY')"; exit 3;; esac
D=scripts/exp07_write_ladder
D4=scripts/exp04_what_ttt_retains
COMMIT=754fa1944065053a9c02ec304e4ce3901b83d4f1
FILES="$D/e55_run.py $D/e55_core.py $D/e55_gpu.py $D/e55_swap.py $D/e55_read.py $D/e55_analyze.py $D/e55_metrics.py $D4/gen_facts.py $D4/gen_world_ctl.py $D4/frame_bank.py $D/e55_gates.py $D4/frames300.txt"
LADDER_OK="714b08ea3b90"
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
if [ -z "$NO_INSTALL" ]; then
  GL=$(sha256sum $D/ladder.py | cut -c1-12); [ "$GL" = "$LADDER_OK" ] || { echo "lab ladder.py ($GL) is not the patched W1 version ($LADDER_OK): exit 6"; exit 6; }
  git cat-file -e $COMMIT^{commit} || { echo "commit $COMMIT not in the local clone: git fetch origin claude/project-thread-rwhdac first (exit 4)"; exit 4; }
  git checkout $COMMIT -- $FILES
fi
if [ -z "$SKIP_VERIFY" ]; then
  while read HH F; do
    [ -n "$HH" ] || continue
    G=$(sha256sum $D/$F | cut -c1-12)
    [ "$G" = "$HH" ] || { echo "VERSION MISMATCH $F: have $G, expected $HH (exit 5)"; exit 5; }
  done <<'MANIFEST'
04e4fdac18ff e55_run.py
210b40d2df40 e55_core.py
9827e34dcde6 e55_gpu.py
b0cd3acd36bb e55_swap.py
a6d506949071 e55_read.py
4a811650bcd8 e55_analyze.py
30769e241d5e e55_metrics.py
035d8779f83a ../exp04_what_ttt_retains/gen_facts.py
95401cdec50a ../exp04_what_ttt_retains/gen_world_ctl.py
eeaaa9135907 ../exp04_what_ttt_retains/frame_bank.py
866eb5989467 ../exp04_what_ttt_retains/frames300.txt
ee82bf19b0b3 e55_gates.py
MANIFEST
  echo "installed code verified against the manifest; lab s3.py sha12 $(sha256sum $D/s3.py | cut -c1-12) (repo copy a16e6dd0d4c9)"
fi
S=$D
$PY $D4/gen_facts.py --selftest | tail -1
$PY $S/e55_read.py --selftest | tail -1
$PY $S/e55_swap.py | tail -1
$PY $S/e55_metrics.py --selftest | tail -1
$PY $S/e55_analyze.py --selftest | tail -1
$PY $S/e55_gates.py --selftest | tail -1
for SD in 61 62 63; do
  $PY $S/e55_gates.py --cell v3_16_1 --seed $SD --default-refs --require-refs --out $OUT/e55_gates_v3a_s$SD.md > $OUT/e55_gates_v3a_s$SD.log 2>&1 || { cat $OUT/e55_gates_v3a_s$SD.log | tail -30; echo "background gates failed for seed $SD (exit 7)"; exit 7; }
  tail -14 $OUT/e55_gates_v3a_s$SD.md
done
$PY $S/e55_run.py dry --cell v3_16_1 --seed 61 --bg pool --arms NAT,SWD | tee $OUT/e55_dry_v3a.json | tail -3
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
REP=$OUT/e55_report_v3a.md
if [ -f $REP ] && [ -z "$RERUN" ]; then echo "report $REP exists: not redone (exit 8; RERUN=1 to redo)"; exit 8; fi
$PY $S/e55_run.py unit --cell v3_16_1 --seed 61 --bg pool --tag v3a --out $OUT --unit-chunks 12
$PY $S/e55_run.py run --cell v3_16_1 --bg pool --tag v3a --out $OUT --decoy names --save-apps 128 --canary-apps 128 --seeds 61,62,63 --arms NAT,SWD --plan "61=16,32,64,128|128;62=16,32,64,128|128;63=16,32,64,128|128"
files() { local out=""; for s in "$@"; do [ -f "$OUT/E55_v3a_${ARM}_s$s.read.json" ] && out="$out,$OUT/E55_v3a_${ARM}_s$s.read.json"; done; echo "${out#,}"; }
ARM=NAT; NATF=$(files 61 62 63); ARM=SWD; SWDF=$(files 61 62 63)
SAVED=$(ls $OUT/E55_v3a_*_s61.saved.json $OUT/E55_v3a_*_s62.saved.json $OUT/E55_v3a_*_s63.saved.json | tr '\n' ',' | sed 's/,$//')
$PY $S/e55_analyze.py --nat "$NATF" --swd "$SWDF" --saved "$SAVED" --a 128 --out $REP
$PY $S/e55_analyze.py --verdict --nat "$NATF" --swd "$SWDF" --saved "$SAVED" --unit $OUT/E55_unit_v3a.json --a 128 --out $OUT/e55_verdict_v3a.md
tail -60 $OUT/e55_verdict_v3a.md

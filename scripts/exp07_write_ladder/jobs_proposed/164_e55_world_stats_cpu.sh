#!/bin/bash
# EXPECTED RUN TIME (Hans confirms before it runs): CPU ONLY, NO GPU, no model: about 5-15 min (ESTIMATE, not measured: nine worlds of 16-144 chunks, each tokenized once for the real and once for the swapped world plus a decoy, plus the checks; the llama3 tokenizer on CPU encodes about 0.5-1 M tokens/s). Writes only small text files.
# E55 controlled world, CPU part: (1) token-length classes of the person / place name pools (the modal class must hold 4N + 8 names for the largest world N = 72 pairs: 296), (2) world statistics table for the nine cells of the design (base, p = 1 / 4 / 9, M = 4 / 16, s = 2 with M = 4, frame bank 12, frame bank disjoint) with every automatic check (chunk length 1024, M pairs and 2 M s sentences per chunk, appearance schedule, p = 9 distance >= 8193 tokens, no repeated frame inside a person while the bank is large enough, question frames vs frame bank 5-gram overlap 0, real / decoy name and place disjointness, background paragraph reuse 0 and real-decoy sharing 0), (3) a second build of every world with the same seed and a comparison of the sha256 of the real and swapped token streams (reproducibility), (4) 30 generated sample sentences of the base world, (5) the CPU unit tests (generator, window-swap slot plan, read plan, analysis).
# Output: $OUT/e55_world_stats.md, $OUT/e55_pools.json, $OUT/e55_samples.txt. NO OVERWRITE: exit 8 if e55_world_stats.md exists unless RERUN=1.
# PINNED INSTALL, NO FETCH: checks the files out of the local repository at commit $COMMIT (exit 4 if absent), verifies sha256 (first 12 hex; exit 5). Needs the lab data/tokenizer/llama3/tokenizer.json and data/narrative_gutenberg (the same files gen_world.py uses; exit 3 if missing). NO_INSTALL=1 skips the checkout, SKIP_VERIFY=1 the check.
# CPU-only dry run (this job IS CPU only; to see what it would do without writing): git fetch origin claude/project-thread-rwhdac && git worktree add --detach /tmp/dry_e55 FETCH_HEAD && ln -s "$(pwd)/data" /tmp/dry_e55/data && cd /tmp/dry_e55 && PY=python3 CPU_ONLY=1 NO_INSTALL=1 bash scripts/exp07_write_ladder/jobs_proposed/164_e55_world_stats_cpu.sh   (then: git worktree remove --force /tmp/dry_e55). CPU_ONLY=1 runs only the unit tests with the stub tokenizer and checks that the data files exist.
set -e
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder
D4=scripts/exp04_what_ttt_retains
COMMIT=d6f8d88496670a664ddb7a3e0988801b8b273667
FILES="$D4/gen_world_ctl.py $D/e55_swap.py $D/e55_read.py $D/e55_analyze.py $D4/frame_bank.py $D/e49_stats.py"
: "${PY:=python3}"; : "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"
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
bb05d7321713 scripts/exp04_what_ttt_retains/gen_world_ctl.py
4c3999626ec6 scripts/exp07_write_ladder/e55_swap.py
cf7eb61f1417 scripts/exp07_write_ladder/e55_read.py
9072c930880c scripts/exp07_write_ladder/e55_analyze.py
eeaaa9135907 scripts/exp04_what_ttt_retains/frame_bank.py
794ae88a9baf scripts/exp07_write_ladder/e49_stats.py

MANIFEST
  echo "installed code verified against the manifest"
fi
$PY $D4/gen_world_ctl.py --selftest | tail -1
$PY $D/e55_swap.py | tail -1
$PY $D/e55_read.py --selftest | tail -1
$PY $D/e55_analyze.py --selftest | tail -1
TOKF=data/tokenizer/llama3/tokenizer.json
[ -f "$TOKF" ] || { echo "$TOKF missing (exit 3)"; exit 3; }
[ -d data/narrative_gutenberg ] || { echo "data/narrative_gutenberg missing (exit 3)"; exit 3; }
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: would write $OUT/e55_pools.json, e55_world_stats.md, e55_samples.txt"; exit 0; }
mkdir -p "$OUT"
[ -f "$OUT/e55_world_stats.md" ] && [ -z "$RERUN" ] && { echo "$OUT/e55_world_stats.md exists: not overwritten (exit 8; RERUN=1)"; exit 8; }
$PY $D4/gen_world_ctl.py --pools --tokenizer-file $TOKF > "$OUT/e55_pools.json"; cat "$OUT/e55_pools.json"
$PY $D4/gen_world_ctl.py --stats all --check-repro --tokenizer-file $TOKF --out "$OUT/e55_world_stats.md"
$PY $D4/gen_world_ctl.py --sample 30 --tokenizer-file $TOKF > "$OUT/e55_samples.txt"; cat "$OUT/e55_samples.txt"
sha256sum $FILES "$OUT/e55_world_stats.md" "$OUT/e55_pools.json" "$OUT/e55_samples.txt" > "$OUT/e55_world_stats.sha256"

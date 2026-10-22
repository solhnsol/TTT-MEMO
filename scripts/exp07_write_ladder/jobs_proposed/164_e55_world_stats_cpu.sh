#!/bin/bash
# EXPECTED RUN TIME (Hans confirms before it runs): CPU ONLY, NO GPU, no model: about 5-15 min (ESTIMATE, not measured: ten worlds of 16-144 chunks, each tokenized once plus a decoy and once more for the reproducibility check; the llama3 tokenizer on CPU encodes about 0.5-1 M tokens/s). Writes only small text files.
# E55 pairless fact world (design section 10), CPU part: (1) token-length classes of the person / place name pools (the modal class must hold 2k + 4 persons and 2k places for the largest world k = 144: 292 / 288), (2) world statistics table for the ten cells (base k=32 p=2; p = 1 / 4 / 9 with k = 16 p; k = 16 / 64 at p = 2; n = 2 with k = 32 or 16; frame bank 12; frame bank disjoint) with every automatic check (chunk length 1024, k n / p sentences per chunk, appearance schedule, p = 9 distance >= 8193 tokens, no repeated frame inside a fact while the bank is large enough, question frames vs frame bank 5-gram overlap 0, real / decoy name and place disjointness, candidate structure of every item: 16 distinct places, 7 same-group + 8 other-group, control names never in the stream), (3) a second build of every world with the same seed and a comparison of the sha256 of the token streams (reproducibility), (4) 30 generated sample sentences of the base world, (5) the CPU unit tests (generator, window-swap slot plan, read plan, metrics). Background is cut into units of at most ~40 tokens, so a chunk always has several slots for the sentences (the earlier pair-world build stopped at 'too few background paragraphs per chunk').
# Output: $OUT/e55_world_stats.md, $OUT/e55_pools.json, $OUT/e55_samples.txt. NO OVERWRITE: exit 8 if e55_world_stats.md exists unless RERUN=1.
# PINNED INSTALL, NO FETCH: checks the files out of the local repository at commit $COMMIT (exit 4 if absent), verifies sha256 (first 12 hex; exit 5). Needs the lab data/tokenizer/llama3/tokenizer.json and data/narrative_gutenberg (the same files gen_world.py uses; exit 3 if missing). NO_INSTALL=1 skips the checkout, SKIP_VERIFY=1 the check.
# CPU-only dry run (this job IS CPU only; to see what it would do without writing): git fetch origin claude/project-thread-rwhdac && git worktree add --detach /tmp/dry_e55 FETCH_HEAD && ln -s "$(pwd)/data" /tmp/dry_e55/data && cd /tmp/dry_e55 && PY=python3 CPU_ONLY=1 bash scripts/exp07_write_ladder/jobs_proposed/164_e55_world_stats_cpu.sh   (then: git worktree remove --force /tmp/dry_e55). CPU_ONLY=1 runs only the unit tests with the stub tokenizer and checks that the data files exist.
set -eo pipefail
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder
D4=scripts/exp04_what_ttt_retains
COMMIT=50cf2de08f393bc48dcf7955cb202a43e435b5a4
FILES="$D4/gen_facts.py $D4/gen_world_ctl.py $D/e55_swap.py $D/e55_read.py $D/e55_analyze.py $D/e55_metrics.py $D4/frame_bank.py"
: "${PY:=python3}"; : "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"
if [ -z "$NO_INSTALL" ]; then
  git cat-file -e $COMMIT^{commit} || { echo "commit $COMMIT not in the local clone: git fetch origin claude/project-thread-rwhdac first (exit 4)"; exit 4; }
  git checkout $COMMIT -- $FILES
fi
if [ -z "$SKIP_VERIFY" ]; then
  while read HH F; do
    [ -n "$HH" ] || continue
    G=$(sha256sum $D/$F | cut -c1-12)
    [ "$G" = "$HH" ] || { echo "VERSION MISMATCH $F: have $G, expected $HH (exit 5)"; exit 5; }
  done <<'MANIFEST'
f51de80be2d3 e55_swap.py
a6d506949071 e55_read.py
a6ef4073a976 e55_analyze.py
1a28133d518d e55_metrics.py
73ac56e67bd7 ../exp04_what_ttt_retains/gen_facts.py
ada39e55ccc3 ../exp04_what_ttt_retains/gen_world_ctl.py
eeaaa9135907 ../exp04_what_ttt_retains/frame_bank.py
MANIFEST
  echo "installed code verified against the manifest"
fi
$PY $D4/gen_facts.py --selftest | tail -1
$PY $D/e55_swap.py | tail -1
$PY $D/e55_read.py --selftest | tail -1
$PY $D/e55_metrics.py --selftest | tail -1
TOKF=data/tokenizer/llama3/tokenizer.json
[ -f "$TOKF" ] || { echo "$TOKF missing (exit 3)"; exit 3; }
[ -d data/narrative_gutenberg ] || { echo "data/narrative_gutenberg missing (exit 3)"; exit 3; }
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: would write $OUT/e55_pools.json, e55_world_stats.md, e55_samples.txt"; exit 0; }
mkdir -p "$OUT"
[ -f "$OUT/e55_world_stats.md" ] && [ -z "$RERUN" ] && { echo "$OUT/e55_world_stats.md exists: not overwritten (exit 8; RERUN=1)"; exit 8; }
$PY $D4/gen_facts.py --pools --tokenizer-file $TOKF > "$OUT/e55_pools.json"; cat "$OUT/e55_pools.json"
$PY $D4/gen_facts.py --stats all --check-repro --tokenizer-file $TOKF --out "$OUT/e55_world_stats.md"
$PY $D4/gen_facts.py --sample 30 --tokenizer-file $TOKF > "$OUT/e55_samples.txt"; cat "$OUT/e55_samples.txt"
sha256sum $FILES "$OUT/e55_world_stats.md" "$OUT/e55_pools.json" "$OUT/e55_samples.txt" > "$OUT/e55_world_stats.sha256"

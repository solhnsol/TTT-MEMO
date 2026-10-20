#!/bin/bash
# EXPECTED RUN TIME (Hans confirms before it runs): about 2.6-3.8 h GPU in total [estimate, per-cell cost NOT calibrated: the masked write with a record of natural K,V per pass is new on natural text]. Breakdown (stream 9105 spread, worlds A and B, passes 1-3):
#   1. W0 score extraction + reuse check: model load ~5-10 min; attention order at W0 for the pass-1 stream of A and B (one capture per chunk, 63 chunks x 2): 0.3-0.6 h; native pass-1 repeat of the R call on both worlds (reuse check): ~0.3 h incl. its own load; if it DIFFERS the job stops here (exit 4) and nothing else runs.
#   2. four attention cells attn k=8, 16, 32, 64, each: write 2 worlds x 3 passes + 6 state readouts (fp32, W0 evaluated once and reused) = 0.5-0.7 h per cell, i.e. 2.0-2.8 h for the four (the design's 2.3-3.4 h = extraction 0.3-0.6 h + 4 x 0.5-0.7 h).
#   3. analysis (CPU, ~1 min); first-chunk NLL read of 24 attention states (last step, non-fatal, SKIP_NLL=1 skips it): ~0.3-0.5 h more. Without that step and the reuse check: 2.3-3.4 h.
# Resumable and splittable in 1-2 h pieces: every (cell, world) writes its own .saved.json and is skipped if it exists; ATTN_CELLS="attn8,attn16" runs only those cells (the order extraction and the reuse check are skipped when their files exist), then eval; rerun the job for the rest.
# E33 attention-mask cells (PROPOSED, NOT QUEUED; needs Hans's approval of the run time): natural-text stream 9105, layout spread, worlds A and B, 3 passes. Masked write (the S3 / chunk-alone write forward) where the masked set per chunk = union over the chunk's loss targets of the top-k
#   head-max attention keys read at FIXED W0 (suffix layers, layer-mean probability, candidates = 8K window before the chunk minus the first 4 stream positions and the last 128 tokens), k = 8, 16, 32, 64, all cut from ONE score read per world (nested sets).
#   Scores are read on the pass-1 stream of each world (A, B separately) and the SAME positional sets are used in passes 1-3 (pass-2/3 filler keeps every exposure unit at the same token position, drift <= 2). Only the top-256 key order is saved (nat_attn_sel_s9105_spread_{A,B}.npz), no score values.
#   Native-fs (L0fs) and chunk-alone (L0mwfs) are NOT rerun: they are the R run's cells (job 80, same $OUT). Reuse is only valid if (a) the filler streams are the same (CPU check below against the R run's saved exposure-unit token starts and stream lengths) and (b) a native pass-1 repeat of the R call reproduces the R pass-1 inner-weight hash bit for bit (step 1);
#   if either differs the job stops before the attention cells (FORCE=1 overrides at your own risk) and native-fs / chunk-alone must be added to this job.
#   No rand-dist comparison cell is included (design decision).
# Readouts and analysis as R: R1-alt (SQuAD tail) main, R1 (pg1342 tail) second, fp32 packed, canary gen / tail NLL, D(A-B) per person and change from W0 per world, 5000 pair-cluster bootstraps (13 pairs) + exact pair-level permutation; contrasts attn - native and attn - chunk-alone, recovered fraction of (chunk-alone - native);
#   per k next to the table: mask fraction of the whole window, mask fraction of the candidates, O recall inside the candidates, enrichment (recall / candidate mask fraction) (nat_attn_analyze.py, report nat_attn_report_9105.{md,json}); first-chunk NLL (e37_firstchunk_nll.py) for the attention states.
# Needs on the server (git fetch origin claude/project-thread-rwhdac && git checkout FETCH_HEAD -- <files>): scripts/exp07_write_ladder/{nat_attn,nat_attn_analyze,nat_signal,nat_ladder,nat_dryrun}.py (nat_ladder.py: hooks only, the R behaviour is unchanged when no hook is installed) plus the R job's files, and the server's own ladder.py, s3.py, s3_stage1.py, s3_attn.py, s3_sets.py.
#   Needs in $OUT the R run's outputs: L0fs_nat_s9105_{A,B}.saved.json (+ .json), L0mwfs_nat_s9105_{A,B}.json, meta_nat_nat_s9105.json. The attention job does not overwrite the R meta (it asserts that the item order is identical).
# First GPU run of the attention cell code: if a cell stops at an assert (e.g. pass stream vs its realized-world encoding), fix the code, do not relax the assert.
set -e
cd "$(dirname "$0")/../../.." 2>/dev/null || true
NAT_DIR=${NAT_DIR:-data/natural-text/worlds}
ATTN_CELLS=${ATTN_CELLS:-attn8,attn16,attn32,attn64}
TOK=${TOKENIZER_JSON:-data/tokenizer/llama3/tokenizer.json}
COMMON="--nat 9105:spread --nat-dir $NAT_DIR --passes 3 --worlds AB --tag nat"
$PY $S/nat_attn.py --selftest
$PY $S/nat_attn_analyze.py --selftest
$PY $S/nat_signal.py --selftest
$PY $S/nat_filler.py --selftest
$PY $S/nat_pass.py --selftest
# CPU: real-tokenizer rebuild of the pass-2/3 streams; must equal the R run's exposure-unit token starts and stream lengths (reuse of the R cells), audit, pool >= 1.2x. Run it BEFORE queueing: the same line.
$PY $S/nat_dryrun.py --nat-dir $NAT_DIR --stream 9105 --passes 3 --require-real --tokenizer-file $TOK --compare-pass-info "$OUT/L0fs_nat_s9105_{w}.saved.json"
$PY $S/nat_filler.py --nat-dir $NAT_DIR --stream 9105 --passes 3 --budget-only
if [ -z "$SKIP_REPRO" ] && [ ! -f "$OUT/nat_attn_repro_s9105_B.json" ]; then
  $PY $S/nat_attn.py $COMMON --cells sel,nativerep --log-clip --mode save
else
  $PY $S/nat_attn.py $COMMON --cells sel --mode save
fi
if [ -z "$SKIP_REPRO" ]; then
  for w in A B; do
    $PY - <<PYEOF || { echo "REUSE CHECK FAILED for world $w: the native pass-1 repeat is not bit-identical to the R run (or missing); stopping before the attention cells (FORCE=1 to continue anyway)"; [ -n "$FORCE" ] || exit 4; }
import json,sys
d=json.load(open('$OUT/nat_attn_repro_s9105_$w.json'));print('world $w reuse check identical:',d['identical']);sys.exit(0 if d['identical'] else 4)
PYEOF
  done
fi
$PY $S/nat_attn.py $COMMON --cells $ATTN_CELLS --log-clip --mode save
$PY $S/nat_attn.py $COMMON --cells $ATTN_CELLS --mode eval --keep
$PY $S/nat_attn_analyze.py --dir $OUT --stream 9105 --tag nat
IDS=data/dclm_val_subset/val_c0_200x8193.u32le.bin
if [ -z "$SKIP_NLL" ] && [ -f "$IDS" ]; then
  ARGS=""
  for c in $(echo $ATTN_CELLS | tr ',' ' '); do for w in A B; do for k in 1 2 3; do ARGS="$ARGS --state ${c}_${w}_p${k}=$OUT/states/${c}_nat_s9105/$w/state_00${k}.npz"; done; done; done
  $PY $S/e37_firstchunk_nll.py --ids $IDS --seed 101 --tag e33attn $ARGS || echo "first-chunk NLL read failed (the margin and canary results stand)"
else
  echo "first-chunk NLL skipped (SKIP_NLL set or $IDS missing)"
fi

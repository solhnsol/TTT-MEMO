#!/bin/bash
# EXPECTED RUN TIME (Hans confirms before it runs): about 1.1-1.5 h GPU [estimate, evaluation cost per state NOT calibrated for natural text]. Breakdown (worlds A and B, passes 1-3, NO L1):
#   native-fs (L0fs) writes 2 worlds x 3 passes: ~18 min with --log-clip; masked-fs (L0mwfs) writes 2 x 3 passes: ~15-18 min;
#   evaluation: 13 distinct states (W0 once, native 3x2, masked 3x2) at 1.5-3 min each = 20-39 min; model load + readout setup for save and eval ~10 min; CPU filler audit ~3 min;
#   first-chunk NLL read of 12 states (last step, non-fatal, SKIP_NLL=1 skips it): ~10-15 min. Without NLL and the clip log: ~1.0-1.3 h.
# E33 main comparison, STAGE 1 (PROPOSED, NOT QUEUED; needs Hans's approval of the run time): natural-text stream 9103, layout clustered, worlds A and B; native-fs (L0fs) vs chunk-alone masked-fs (L0mwfs, block 1024, x1), passes 1-3.
#   Same R-type cells as the 9105 spread run (job 80) minus the offline L1 rows (L1 is not part of the stage 1 comparison; add --cells L0fs,L0mwfs,L1 --epochs 8 --l1-eval-epochs 1,2,3,8 if the design wants it).
#   Pass-2/3 filler = background of the OTHER natural streams, CPU-audited, cut to each slot's token length (same exposure positions in every pass; asserted). The pool must be >= 1.2x the slot text per pass (--budget-only, exit 3 otherwise);
#   proxy-tokenizer dry run on the clustered layout: pool ratio 1.66 (pass 2) / 2.42 (pass 3), pass 2 consumes ~98% of the pool paragraphs (tight); the real-tokenizer dry run below decides.
#   Stage 2 cells (attention k32/k64, recency count-matched) are NOT in this job.
# Needs on the server: scripts/exp07_write_ladder/{nat_world,nat_filler,nat_pass,nat_ladder,nat_analyze,nat_dryrun}.py and the server's own ladder.py. Data as job 80 (nat_9101..9106_clustered.json under $NAT_DIR for 9103 and the other streams' backgrounds). No real-name map is read.
# Output: $OUT/{L0fs,L0mwfs}_nat_s9103_{A,B}.json, meta_nat_nat_s9103.json, nat_filler_audit_9103.{md,json}, nat_r_report_9103.{md,json}, nat_person_margins_9103.json, e37_firstchunk_e33s1.json.
# Aborts before any GPU work if the filler audit fails (exit 3). If save mode stops at an assert, fix the code, do not relax the assert.
set -e
cd "$(dirname "$0")/../../.." 2>/dev/null || true
NAT_DIR=${NAT_DIR:-data/natural-text/worlds}
COMMON="--nat 9103:clustered --nat-dir $NAT_DIR --cells L0fs,L0mwfs --passes 3 --worlds AB --tag nat"
$PY $S/nat_filler.py --selftest
$PY $S/nat_pass.py --selftest
# CPU dry run of the whole filler assembly with the REAL tokenizer (both worlds, passes 2-3, drift <= 2, strata equal, no pool paragraph twice, audit); needs the `tokenizers` package and the llama3 tokenizer.json (baseline_config.TOKENIZER_PATH or TOKENIZER_JSON); no GPU. Run it BEFORE queueing: $PY $S/nat_dryrun.py --nat-dir $NAT_DIR --stream 9103 --layout clustered --passes 3 --require-real
$PY $S/nat_dryrun.py --nat-dir $NAT_DIR --stream 9103 --layout clustered --passes 3 --require-real --tokenizer-file ${TOKENIZER_JSON:-data/tokenizer/llama3/tokenizer.json}
$PY $S/nat_filler.py --nat-dir $NAT_DIR --stream 9103 --layout clustered --passes 3 --budget-only   # CPU dry run of the pool assembly: pool chars >= 1.2 x slot chars per pass (exit 3 otherwise)
$PY $S/nat_filler.py --nat-dir $NAT_DIR --stream 9103 --layout clustered --passes 3 --out-dir $OUT
$PY $S/nat_ladder.py $COMMON --log-clip --mode save
$PY $S/nat_ladder.py $COMMON --log-clip --mode eval --keep
$PY $S/nat_analyze.py --dir $OUT --stream 9103 --tag nat
IDS=data/dclm_val_subset/val_c0_200x8193.u32le.bin
if [ -z "$SKIP_NLL" ] && [ -f "$IDS" ]; then
  ARGS=""
  for c in L0fs L0mwfs; do for w in A B; do for k in 1 2 3; do ARGS="$ARGS --state ${c}_${w}_p${k}=$OUT/states/${c}_nat_s9103/$w/state_00${k}.npz"; done; done; done
  $PY $S/e37_firstchunk_nll.py --ids $IDS --seed 101 --tag e33s1 $ARGS || echo "first-chunk NLL read failed (the margin and canary results stand)"
else
  echo "first-chunk NLL skipped (SKIP_NLL set or $IDS missing)"
fi

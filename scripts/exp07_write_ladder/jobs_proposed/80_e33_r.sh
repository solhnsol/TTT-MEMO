#!/bin/bash
# EXPECTED RUN TIME (Hans confirms before it runs): about 1.6-2.1 h GPU in total [estimate, evaluation cost per state NOT calibrated for natural text]. Breakdown (worlds A and B, passes 1-3, L1 8 epochs):
#   native-fs (L0fs) writes 2 worlds x 3 passes: ~18 min with --log-clip (job 35: 97-147 s/pass, +1 forward/backward per chunk for the clip log); masked-fs (L0mwfs) writes 2 x 3 passes: ~15-18 min;
#   L1 doc rows: ~15 packed 1024-token steps per epoch (152 rows, ~14k tokens: NOT the synthetic 10 steps/epoch; logged as n_chunks), 8 epochs x 2 worlds: ~8-10 min;
#   evaluation: 21 distinct states (W0 once, evaluated once and reused by hash; native 3x2, masked 3x2, L1 epochs 1,2,3,8 x2) at 1.5-3 min each = 32-63 min (3 min/state is backed out of E39's 52 min [inference]);
#   model load + readout setup for save and for eval ~10 min; CPU filler audit ~3 min; first-chunk NLL read of 12 states (step 6, non-fatal, skip with SKIP_NLL=1): ~10-15 min.
#   The coordinator's/E33's 1.1-1.5 h is the lower end if eval costs ~1.5 min/state and the clip log / NLL read are dropped. Cheaper variants: drop --log-clip from the native save command (-6 min), SKIP_NLL=1 (-10-15 min).
# E33 R (PROPOSED, NOT QUEUED; Hans approved the experiment, the run time above still needs his confirmation): natural-text stream 9105, layout spread, worlds A and B; native-fs vs masked-fs (chunk alone, block 1024, x1), passes 1-3,
#   pass-2/3 filler = background of the OTHER natural streams (CPU-audited nat_filler.py), cut to each slot's token length so every exposure unit sits at the SAME token position in every pass (asserted: s_cross/s_same/n_out equal pass 1, drift <= 2 tokens, audit repeated on the cut texts);
#   offline L1 on the exposure units (doc rows, plain SGD lr x1 clip 1.0 as the synthetic L1), evaluated at W0 and epochs 1,2,3,8.
# Readouts: R1-alt (SQuAD tail) and R1 (pg1342 tail), fp32 packed, all items of every family per person (meta_nat_nat_s9105.json 'items' = order of the margin lists), canary gen/tail NLL per state, per-chunk grad norm / clip coefficient per pass ('clip').
# Analysis (nat_analyze.py): E33 2.11 D(A-B) per person (un-halved; analyze_ladder.py D is half) with pair-bootstrap CI, masked-native, R*, S4 exposure-group table, AND C_w (change from W0) per world; person-level json nat_person_margins_9105.json.
# Hash reproducibility: the eval step asserts that every loaded state's inner-weight hash equals the streamed one (as ladder.py); a second identical save run for a run-to-run check is NOT included (it would double the writes).
# Needs on the server (git fetch origin claude/project-thread-rwhdac && git checkout FETCH_HEAD -- <files>): scripts/exp07_write_ladder/{nat_world,nat_filler,nat_pass,nat_ladder,nat_analyze}.py (new files only, nothing existing changed) and the server's own ladder.py.
#   Data: $NAT_DIR = data/natural-text/worlds with nat_9101..9106_spread.json (9105 + the five other streams' backgrounds); optional $NAT_DIR/../strata.json; data/narrative_gutenberg/{pg11,pg1342}.txt (job aborts if missing); data/squad/dev-v1.1.json; data/dclm_val_subset/val_c0_200x8193.u32le.bin (step 6 only).
#   No real-name map is read. Output: $OUT/{L0fs,L0mwfs,L1}_nat_s9105_{A,B}.json, meta_nat_nat_s9105.json, nat_filler_audit_9105.{md,json}, nat_r_report_9105.{md,json}, nat_person_margins_9105.json, e37_firstchunk_e33r.json.
# Aborts before any GPU work if the filler audit fails (exit 3). First GPU run of the natural loader: if save mode stops at an assert (stream length, exposure position drift, SQuAD-tail length, readout tail check), fix the code, do not relax the assert.
set -e
cd "$(dirname "$0")/../../.." 2>/dev/null || true
NAT_DIR=${NAT_DIR:-data/natural-text/worlds}
COMMON="--nat 9105:spread --nat-dir $NAT_DIR --cells L0fs,L0mwfs,L1 --passes 3 --epochs 8 --l1-eval-epochs 1,2,3,8 --worlds AB --tag nat"
$PY $S/nat_filler.py --selftest
$PY $S/nat_pass.py --selftest
$PY $S/nat_filler.py --nat-dir $NAT_DIR --stream 9105 --passes 3 --budget-only   # CPU dry run of the pool assembly: pool chars >= 1.2 x slot chars per pass (exit 3 otherwise)
$PY $S/nat_filler.py --nat-dir $NAT_DIR --stream 9105 --passes 3 --out-dir $OUT
$PY $S/nat_ladder.py $COMMON --log-clip --mode save
$PY $S/nat_ladder.py $COMMON --log-clip --mode eval --keep
$PY $S/nat_analyze.py --dir $OUT --stream 9105 --tag nat
IDS=data/dclm_val_subset/val_c0_200x8193.u32le.bin
if [ -z "$SKIP_NLL" ] && [ -f "$IDS" ]; then
  ARGS=""
  for c in L0fs L0mwfs; do for w in A B; do for k in 1 2 3; do ARGS="$ARGS --state ${c}_${w}_p${k}=$OUT/states/${c}_nat_s9105/$w/state_00${k}.npz"; done; done; done
  $PY $S/e37_firstchunk_nll.py --ids $IDS --seed 101 --tag e33r $ARGS || echo "first-chunk NLL read failed (the margin and canary results stand)"
else
  echo "first-chunk NLL skipped (SKIP_NLL set or $IDS missing)"
fi

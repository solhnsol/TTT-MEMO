#!/bin/bash
# PROPOSED, NOT QUEUED (no approval yet). Follow-ups of job 77 / 83 (window context gain). Pick the step with STEP=split|states|nat (default split). Expected GPU time per step (estimates, not calibrated):
#   split  : CPU only (server CPU, ~1 min; needs gen_world, suppress, the tokenizer and the job 77 json). No GPU.
#   states : the same reads on the saved K6 / K6' states (passes 1-4, world A, seed 101): per cell one process, model load ~5 min + W0 and 4 states x 3-4 min = ~20-25 min; K6 and K6' ~45 min in total (world B doubles it).
#   nat    : natural-text world (E33), stream 9105 spread, worlds A and B, W0: load ~5 min + 2 x 3-4 min = ~12-15 min; plus a second stream (9103 clustered) +~8 min. States of the E33 runs (states/L0fs_nat_s9105/...) can be added with STATES=1 (+~4 min per state and world).
set -e
STEP=${STEP:-split}
JSON=${JSON:-$OUT/ctx_gain_ctx.json}
case $STEP in
  split)
    $PY $S/ctx_gain_split.py --selftest
    $PY $S/ctx_gain_split.py $JSON --out-dir ${SPLIT_DIR:-/tmp/ctx_gain_split} --tokenizer-file ${TOKENIZER_JSON:-data/tokenizer/llama3/tokenizer.json} ;;
  states)
    # saved K6 / K6' states: tags K6TAG / K6PTAG are the --tag of the K6 and K6' jobs (files states/S3_<tag>_s101/A/state_00<p>.npz, p = 1..4 = after pass p)
    $PY $S/ctx_gain.py --selftest
    for TAG in ${K6PTAG:-k6p} ${K6TAG:-k6}; do
      $PY $S/ctx_gain.py --seeds 101 --worlds ${WORLDS:-A} --state-template "states/S3_${TAG}_s{seed}/{w}/state_{p:03d}.npz" --state-passes 1,2,3,4 --tag ctx_states_$TAG
    done ;;
  nat)
    NATS=${NATS:-9105:spread}
    for N in $NATS; do
      $PY $S/ctx_gain.py --selftest
      # CPU dry run first (labels only; real tokenizer): O chunks / non-O chunks after the first O chunk per world
      $PY $S/ctx_gain.py --dry --nat $N --nat-dir ${NAT_DIR:-data/natural-text/worlds} --worlds AB --tokenizer-file ${TOKENIZER_JSON:-data/tokenizer/llama3/tokenizer.json}
      EXTRA=""
      [ -n "$STATES" ] && EXTRA="--state-template states/L0fs_nat_s${N%%:*}/{w}/state_{p:03d}.npz --state-passes 1,2,3"
      $PY $S/ctx_gain.py --nat $N --nat-dir ${NAT_DIR:-data/natural-text/worlds} --worlds AB $EXTRA --tag ctx_nat_${N%%:*}
    done ;;
esac

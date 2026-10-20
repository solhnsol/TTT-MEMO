#!/bin/bash
# EXPECTED RUN TIME (Hans confirms before it runs): about 15-25 min GPU for W0 on seeds 101, 102, 103 world A [estimate, NOT calibrated]: model load ~5 min; per seed one native read of 63 chunks (prefix over the 64K stream once, ~1.5-2 min) + 63 no-context reads (~1-2 min)
#   = ~3-4 min per (seed, weights); three seeds ~10-12 min. Each extra weights set (saved state) adds ~3-4 min per seed; world B doubles it. The default below is W0 only.
# Label-free chunk-selection signal, exact: window context gain per chunk = mean NLL with the 8K window context - mean NLL with no context (same CH-1 targets), then the AUC of O chunks (oracle set non-empty) vs non-O chunks after the first O chunk (score = -delta, 0.5 = no separation).
# PROPOSED, NOT QUEUED (Hans has not approved). No weights are written, nothing is trained. Output: $OUT/ctx_gain_ctx.json (per-chunk nll_ctx / nll_noctx / delta / |O| per run) and ctx_gain_ctx.md (summary table).
# Optional (set before running): STATES=1 adds the saved native-fs states of each seed (states/L0fs_fskeep_s{seed}/A/state_00{p}.npz, passes 1-4), +~13 min per seed; WORLDS=AB adds world B.
set -e
cd "$(dirname "$0")/../../.." 2>/dev/null || true
$PY $S/ctx_gain.py --selftest
# CPU part (no model, no GPU; needs the `tokenizers` package and the llama3 tokenizer.json): builds the three worlds and the O / non-O chunk labels, asserts >= 3 chunks of each class. Run it BEFORE queueing:
$PY $S/ctx_gain.py --dry --seeds 101,102,103 --worlds ${WORLDS:-A} --tokenizer-file ${TOKENIZER_JSON:-data/tokenizer/llama3/tokenizer.json}
EXTRA=""
[ -n "$STATES" ] && EXTRA="--state-template states/L0fs_fskeep_s{seed}/{w}/state_{p:03d}.npz --state-passes 1,2,3,4"
$PY $S/ctx_gain.py --seeds 101,102,103 --worlds ${WORLDS:-A} $EXTRA --tag ctx

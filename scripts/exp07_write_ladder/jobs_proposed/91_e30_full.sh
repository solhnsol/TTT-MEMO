#!/bin/bash
# EXPECTED RUN TIME (Hans confirms before it runs): about 1.2-1.6 h GPU on top of job 90 [estimate, unmeasured; same inputs as job 90]. Group 1 (F128, F512, F2048): ~13 min write + 12 states eval ~8 min + setup ~5 min = ~25 min;
#   group 2 (P512, R, Na, Nb; P8192 is omitted: with a real 8192-token context and the chunk at position 8192 it is the same computation as F8192): ~75 s/chunk (three 8192-token context preparations per world) = ~33 min write + 16 states eval ~10 min + setup ~5 min = ~50 min.
# E30 step 2, s101, the remaining cells of E30 2.0 after job 90 (PROPOSED, NOT QUEUED): F128/F512/F2048 (dose-response), P512 (position-shift control: context of ceil(L/1024)*1024 = 1024 real tokens, chunk at position 1024, nearest 512 visible),
#   R (F0 with the first 128 loss tokens excluded), Na (F8192 on a second independent background-style text), Nb (F8192 on raw natural book text). Run only if job 90 passed its selfcheck and the F0-vs-M gate in its report is not failed.
# Uses the context files and the same code as job 90 (ctx-dir $OUT); the analysis merges the three tags (identical item order asserted).
set -e
cd "$(dirname "$0")/../../.." 2>/dev/null || true
[ -f $OUT/e30_ctx_bg_s101.json ] || { echo "context files missing: run job 90 first"; exit 5; }
$PY $S/e30_dose.py --seed 101 --ctx-dir $OUT --cells F128,F512,F2048 --scales 1,4 --mode save --tag e30g1
$PY $S/e30_dose.py --seed 101 --ctx-dir $OUT --cells F128,F512,F2048 --scales 1,4 --mode eval --tag e30g1
$PY $S/e30_dose.py --seed 101 --ctx-dir $OUT --cells P512,R,Na,Nb --scales 1,4 --mode save --tag e30g2
$PY $S/e30_dose.py --seed 101 --ctx-dir $OUT --cells P512,R,Na,Nb --scales 1,4 --mode eval --tag e30g2
$PY $S/analyze_e30.py --dir $OUT --seed 101 --tags e30core,e30g1,e30g2

#!/bin/bash
# EXPECTED RUN TIME (Hans confirms before it runs): about 40-60 min GPU [estimate, nothing here was measured: the context preparation (10K-token prefix + 8-chunk suffix read per chunk and world, ~9 s guessed), the gradient (~2 s/cell/chunk,
#   E36: 1.4 s/chunk) and the eval (35-40 s/state, from E35/E39 logs) are the inputs]. Breakdown: CPU context files + audit 2-3 min; selfcheck (3 chunks, model load + compile) ~5 min; save core (A, M, F0, F8192; 26 c2 chunks x 2 worlds, ~35 s/chunk) ~15 min
#   + writing 16 state files (scales 1 and 4) ~3 min; eval 17 states (16 + W0) ~10 min + fp32 setup ~3 min; analysis <1 min. E30 2.0's 15-30 min core estimate is the lower end.
# E30 step 1, core cells, seed s101 only (PROPOSED, NOT QUEUED; needs Hans's approval of the run time): native A, chunk alone M, fact-free context with nothing visible F0 (the gate: must equal M), fact-free context fully visible F8192.
#   One aggregated step W0 -> W per cell, world and scale (1, 4), readout R1-alt D1 per pair (analyze_e30.py). The full dose-response (F128/512/2048, P, R, N) is job 91 and runs only after this core job's gate.
# Needs on the server (fetch from claude/project-thread-rwhdac): scripts/exp07_write_ladder/{e30_core,e30_ctx,e30_dose,analyze_e30}.py (new files only; the server's s3.py / ladder.py / s3_sets.py are imported unchanged: Ladder3.prefix_masked/record/grad_nat/grad_mask/read_state).
#   Data: data/narrative_gutenberg/pg*.txt (>= ~12 books usable for the three context kinds, canary pg11/pg1342), data/squad (readout), stage-1 verified fp32 readout log as for every ladder job.
# SELFCHECK DECIDES (E30 design 10/20, 2nd revision): gate_realctx_vs_windowref (cos>=0.999 and CE diff<0.01; rel>0.05 is a warning only) and gate_F0_vs_M_cos_ge_0.999; info_* keys (native cos 0.99, F0 vs M rel 1e-3) are descriptive. Check without GPU: python3 e30_core.py --selftest (the verdict function, which keys decide). E30_GATES=old restores the original gates.
# Gates, in this order, each aborts the job (set -e): context audit (exit 3), selfcheck (exit 4: real-window context pipeline vs native gradient cosine >= 0.99 and CE within 0.05; F0 vs M gradient relative difference <= 1e-3), then the F0-vs-M D1 gate is printed in the report (not an abort).
# Memory: accumulators = (cells x 2 worlds x inner-parameter fp32 pytree); the script refuses to start if they exceed 50% of the device memory (split --cells then). Core = 4 cells x 2 worlds ~ 19 GiB if the inner parameters are the three suffix MLP matrices.
set -e
cd "$(dirname "$0")/../../.." 2>/dev/null || true
$PY $S/e30_core.py --selftest
$PY $S/e30_ctx.py --selftest
$PY $S/e30_ctx.py --world-seed 101 --out-dir $OUT --n-segments 26
$PY $S/e30_dose.py --seed 101 --ctx-dir $OUT --selfcheck
$PY $S/e30_dose.py --seed 101 --ctx-dir $OUT --cells core --scales 1,4 --mode save --tag e30core
$PY $S/e30_dose.py --seed 101 --ctx-dir $OUT --cells core --scales 1,4 --mode eval --tag e30core
$PY $S/analyze_e30.py --dir $OUT --seed 101 --tags e30core

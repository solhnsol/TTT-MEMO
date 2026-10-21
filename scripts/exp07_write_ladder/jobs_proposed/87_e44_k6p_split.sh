#!/bin/bash
# EXPECTED RUN TIME (Hans confirms before it runs): about 50-55 min GPU per cell [estimate from the K6' job 64 (53.9 min); the masked-token count differs per cell]. Approved cells only: sent and noo (two cells, ~1.7-1.8 h in total). bg and unif70 are NOT approved: they are in 88_e44_bg_unif70.sh.
# E44 K6' factor split (PROPOSED, NOT QUEUED; Hans approved the experiment 2026-10-21 10:22 KST in the core thread, the run time above still needs his confirmation). s101, worlds A+B, passes 1-4, --keep, save then eval per cell. All cells mask only chunks WITH an oracle set (same range as K6'), nothing elsewhere; masked positions are in the window before the chunk (O keeps its in-chunk part as the oracle definition).
#   sent  (SENT)   = O + every token of every exposure sentence in the window (B, P, X); background stays visible.
#   noo   (NOO)    = the whole window minus the O tokens (only the answer tokens stay visible).
#   (bg and unif70 are coded but not approved: see 88_e44_bg_unif70.sh)
# Code: s3_sets.py (Sets.sentmask / bgmask / noomask / type_counts; sentence_table 'toks' from E43), s3.py (cells sent bg noo; the chunk log gets n_O, has_O and types = masked tokens by type O/B/P/X/G/chunk). unif70 = the existing oraclefill with --fill-size-from bg (no new code).
# Unit tests (CPU): e44_sets_test.py (O <= K3 <= SENT; BG and NOO as defined; types add up; UNIF70 size; existing cells bit-identical to the stored hash). Per-chunk masked counts on the real table (CPU, server): k3_counts.py (K3, K3b, SENT, BG, NOO and the type counts per chunk) -> /tmp/k3_counts/k3_counts.md. Run both BEFORE queueing.
set -e
$PY $S/e43_sets_test.py
$PY $S/e44_sets_test.py
$PY $S/k3_counts.py --selftest
$PY $S/k3_counts.py --seeds 101 --passes 4 --worlds AB --out-dir ${K3_COUNTS_DIR:-/tmp/k3_counts}
for C in sent noo; do
  for M in save eval; do
    $PY $S/s3.py --cell $C --world-seed 101 --passes 4 --worlds AB --keep --tag $C --mode $M
  done
done

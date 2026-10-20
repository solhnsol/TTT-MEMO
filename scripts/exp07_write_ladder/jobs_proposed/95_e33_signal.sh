#!/bin/bash
# EXPECTED RUN TIME (Hans confirms before it runs): about 0.6-1.2 GPU-h for both streams together [estimate from the E33 2.13 design: ~0.3-0.6 h per stream = one attention capture per chunk (63) + per eligible chunk 3 baselines x 7 k masked forwards]; CPU dry run and selftests ~2 min before.
# E33 2.13 k-resolved attention signal check on natural text (W0, no training, world A): masked fraction, O recall/precision/enrichment, background share, epsilon with rand-dist and recency, K* check. PROPOSED, NOT QUEUED.
# Main stream 9103 clustered, check stream 9105 spread (STREAMS env: "9103:clustered 9105:spread"). Needs commits >= the nat_signal.py commit; server-only: s3_stage1.py.
# CPU step first (no GPU): python3 nat_signal.py --dry --nat 9103:clustered --nat-dir $NAT_DIR --require-real   (prints per chunk O sizes, answer tokens, eligible chunks)
set -e
cd "$(dirname "$0")/../../.." 2>/dev/null || true
NAT_DIR=${NAT_DIR:-data/natural-text/worlds}
STREAMS=${STREAMS:-"9103:clustered 9105:spread"}
$PY $S/nat_signal.py --selftest
$PY $S/nat_signal_analyze.py --selftest
for SL in $STREAMS; do
  $PY $S/nat_signal.py --dry --nat $SL --nat-dir $NAT_DIR --require-real | tail -3
done
FILES=""
for SL in $STREAMS; do
  ST=${SL%%:*}; LA=${SL##*:}
  $PY $S/nat_signal.py --nat $SL --nat-dir $NAT_DIR --worlds A --tag sig
  F=$OUT/nat_signal_sig_s${ST}_${LA}_A.json
  $PY $S/nat_signal_analyze.py $F
  FILES="${FILES:+$FILES,}$F"
done
# main stream first; pooled table (all eligible chunks of both streams, CI only) when two streams ran
case "$FILES" in *,*) $PY $S/nat_signal_analyze.py $FILES --out $OUT/nat_signal_sig_pooled.md;; esac

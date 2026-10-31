#!/bin/bash
# JOB 590 = E57 T1/T2 v2 BAND-512 WRITES ON STREAM B20 (prepared; NOT QUEUED until Hans confirms job name + commit; Hans 19:56 KST 'as recommended'): band-512 W0 scoring pass (forward only, natural + band-512 NLL per row) on B20, then GAINQ512 p=.26 (top 26% of the eligible rows per chunk by nll_band512 - nll_win) and RANDQ512 p=.10 (random 10% of the eligible rows, seed 586) written for one cycle on the frozen v2 setup of 570 / 575 / 585 (SELC1 engine, Adam direction, 0.5 x the NAT norm of the 570 B20 NAT arm), end canary in the arms.json trajectory. Stream A needs no write: the states GAINQ512 p.26 (hash e3dc1ac6...), RANDQ512 p.10 (8dafe402...) and ALLQ512 (454c1658..., job 560) exist (read in job 595). New file e57_t1t2v2_b512_run.py (reuses the stages of e57_t1t2v2_gainq_run / e57_t1t2v2_lowp_run); no existing e57_*.py edited, no git checkout. Output $TOUT: E57_v2B20_score_W0.npz, e57_v2B20_lowp_masks.npz (arms GAINQ512P26 / RANDQ512P10), E57_v2B20_GAINQ512P26.arms.json, E57_v2B20_RANDQ512P10.arms.json, states/. ESTIMATE ONLY, unmeasured: load 15 min + B20 record 5 min + score pass ~10 min + 2 writes x ~7 min = ~45 min.
set -eo pipefail
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder; S=$D
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
for f in e57_t1t2v2_b512_run.py e57_t1t2v2_lowp_run.py e57_t1t2v2_gainq_run.py e57_t1t2v2_run.py e57_qband_run.py e57_closure_masks.py e57_main_run.py e55_core.py ladder.py; do echo "sha256 $(sha256sum $D/$f | cut -c1-12) $f"; done
$PY $S/e57_t1t2v2_b512_run.py --selftest | tail -1
IN=${IN:-$OUT/e57_t1t2v2_in}; TOUT=${TOUT:-$OUT/e57_t1t2v2_b512}; P570=${P570:-$OUT/e57_t1t2v2}
chk(){ [ "$(sha256sum "$1" | cut -c1-64)" = "$2" ] || { echo "$1 differs from the frozen copy (exit 9)"; exit 9; }; }
[ -f "$IN/streamB20_stream_sentences.jsonl" ] && [ -f "$P570/streamB20/E57_v2B20_NAT.arms.json" ] || { echo "input missing (exit 7)"; exit 7; }
chk $IN/streamB20_stream_sentences.jsonl ee7b1dc00622a55f395e63360a96095faf397dfc87081b9563022c759e055c85
mkdir -p "$TOUT"
[ -f "$TOUT/E57_v2B20_GAINQ512P26.arms.json" ] && [ -z "$RERUN" ] && { echo "arms.json exists (exit 8; RERUN=1 to redo)"; exit 8; }
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
[ -f "$TOUT/E57_v2B20_score_W0.npz" ] || $PY $S/e57_t1t2v2_b512_run.py --stage score --sentences $IN/streamB20_stream_sentences.jsonl --out $TOUT
$PY $S/e57_t1t2v2_b512_run.py --stage mask --sentences $IN/streamB20_stream_sentences.jsonl --out $TOUT
$PY $S/e57_t1t2v2_b512_run.py --stage write --sentences $IN/streamB20_stream_sentences.jsonl --out $TOUT --nat-ref $P570/streamB20/E57_v2B20_NAT.arms.json
echo "outputs: $TOUT/E57_v2B20_GAINQ512P26.arms.json $TOUT/E57_v2B20_RANDQ512P10.arms.json $TOUT/E57_v2B20_score_W0.npz $TOUT/e57_v2B20_lowp_masks.npz"

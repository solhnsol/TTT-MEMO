#!/bin/bash
# JOB 677 = E57 NEWS T1/T2 PER-TOKEN RE-READ (comparison thread card 'synthetic + news T1/T2', Hans 22:11 KST 'go'; read only, no write): the 150 frozen T1/T2 v1 frames (100 overlap-free) and the 98 frozen reverse v2 questions (196 frames) are read again with the OUT reader of the original jobs on the end-of-cycle-1 states (stream A: NAT, NATQ, ALLQ128, ALLQ512, GAINQ512 p.26 / p.10, RANDQ512 p.10, SGD x2 / x4; stream B20: NAT, NATQ, ALLQ128, GAINQ512 p.26, RANDQ512 p.10) and W0, token log-probs stored; branch-token / first-token / full-sum ranks; summed log-probs compared with the stored score files. New file e57_news_tokread.py (uses e57_syn_tokread.py); no existing e57_*.py edited, no git checkout.
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
case "$PY" in /*) ;; *) echo "PY not absolute: $PY" >&2; exit 3;; esac
cd "$(dirname "$0")/../../.." 2>/dev/null || true
D=scripts/exp07_write_ladder; S=$D
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"; mkdir -p "$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
for f in e57_news_tokread.py e57_syn_tokread.py e57_t1t2_run.py e57_t1t2v2_run.py e57_reads.py e55_gpu.py ladder.py; do echo "sha256 $(sha256sum $D/$f | cut -c1-12) $f"; done
$PY $S/e57_news_tokread.py --selftest | tail -1
TOUT=${TOUT:-$OUT/e57_news_tokread}; mkdir -p "$TOUT"
for f in $OUT/e57_news/e57_t1t2/t1t2_v1.json $OUT/e57_news/e57_t1t2/e57_t1t2_overlap.json $OUT/e57_t1t2v2_in/t1t2_v2_frozen.json $OUT/e57_t1t2v2_in/e57_t1t2v2d3_candidates.json $OUT/e57_news/e57_bank/bank_v1.json; do [ -f "$f" ] || { echo "input missing: $f (exit 7)"; exit 7; }; done
$PY - <<'PYE'
import sys,os,json
sys.path.insert(0,'scripts/exp07_write_ladder');sys.argv=['x']
import e57_news_tokread as N
miss=[(k,l) for k,d in N.registry().items() for l,(f,h) in d.items() if not (os.path.exists(f) or os.path.exists(N.ROOT/f))];print('state files missing:',miss or 'none');sys.exit(1 if miss else 0)
PYE
[ -n "$CPU_ONLY" ] && { echo "CPU_ONLY: done before any GPU step"; exit 0; }
[ -f "$TOUT/e57_news_tokread_keys.json" ] && [ -z "$RERUN" ] && { echo "e57_news_tokread_keys.json exists (exit 8; RERUN=1 to redo)"; exit 8; }
[ -f "$TOUT/e57_news_tokread.json" ] || $PY $S/e57_news_tokread.py --stage read --out "$TOUT"
$PY $S/e57_news_tokread.py --stage analyze --out "$TOUT"
echo "outputs: $TOUT/e57_news_tokread_keys.json $TOUT/e57_news_tokread.json"

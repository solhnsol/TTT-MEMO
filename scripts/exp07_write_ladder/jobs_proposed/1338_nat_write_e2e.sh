#!/bin/bash
# JOB 1338 = NATURAL-DOCUMENT WRITE EXPERIMENT, E2E 3B (Hans 11/22 '자연 문서 쓰기 만들어'; code e59_nat_write.py + e59_nat_write_analyze.py (new), engine of e59_srv_run.py; data = TTT-reg branch claude/natural-recur-stage0 cloned read-only to /home/solhnsol/workspace/TTT-reg-stage0): the first $BOOKS (default 12; BOOKS=24 for all) PG-19 books are streamed as 1024-token chunks and WRITTEN with the original rule (clipped SGD lr 1 clip 1, loss = mean NLL of all tokens), weights restart from W0 for every book. Arms NAT / HIDE_PAIR (event-sentence rows lose the keys of the earlier occurrences of the same pair) / HIDE_EV (event-sentence rows lose all earlier event sentences) / MASK1024 (band 1024 on every row), update-path masks in all 32 layers. Read = prequential: before each write step the chunk is forwarded once more with the NATURAL 8192 window (masked arms: extra forward-only pass) and the NLL of the m2 tokens of the selected events (all k >= 1 and 2000 random k = 0 events, selection seed 0) is stored. Main measurement: events with prev > 8192 (earlier mention outside the natural window), (arm - NAT), book-clustered, minus the k = 0 control; the k = 1 far events are a placebo-like class for HIDE_PAIR / HIDE_EV (their previous occurrence was written unmasked). Books are the unit of variation (no write randomness). EXPLORATORY; NOT queued until Hans' sentence; est. ~6 h for 12 books (~12 h for 24; unmeasured: NAT 1.9 s / chunk + masked arms ~3.9 s / chunk (write 3.2 s + natural-window forward 0.7 s) from earlier measurements; smoke ~15 min).
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
cd "$(dirname "$0")/../../.." 2>/dev/null || true
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 26"; exit 26; }
D=scripts/exp07_write_ladder;O=$OUT/nat_write;DATA=/home/solhnsol/workspace/TTT-reg-stage0/data/natural-text/recur;BOOKS=${BOOKS:-12}
mkdir -p $O/smoke
sha256sum $D/e59_nat_write.py $D/e59_nat_write_analyze.py $D/e59_stage0_run.py $D/e55_core.py $D/e57_natmask_run.py $D/e57_qband_run.py $DATA/sel24_events.json $DATA/sel24_books.jsonl
$PY $D/e59_nat_write.py --selftest --data $DATA | tail -1 | cut -c1-200
$PY $D/e59_nat_write.py --data $DATA --out $O/smoke --books 1 --max-chunks 24 --arms NAT,HIDE_EV,MASK1024 --n-k0 50
$PY - $O/smoke <<'PYE'
import json,sys,math,numpy as np
r=json.load(open(sys.argv[1]+'/nat_write.json'));b=r['books'][0];A=b['arms']
assert b['chunks']==24 and all(math.isfinite(A[a]['mean_nat_ce']) and 1.0<A[a]['mean_nat_ce']<6.0 for a in A),A
assert A['HIDE_EV']['masked_pairs']>0 and A['MASK1024']['masked_pairs']==0 and A['NAT']['masked_pairs']==0,'HIDE_EV must hide something in the first 24 chunks'
for a in ('HIDE_EV','MASK1024'):assert abs(A[a]['nat_ce_chunk'][0]-A['NAT']['nat_ce_chunk'][0])<2e-2,a
assert max(abs(A[a]['nat_ce_chunk'][k]-A['NAT']['nat_ce_chunk'][k]) for a in ('HIDE_EV','MASK1024') for k in range(8,24))>1e-4,'the arms must write different weights'
print('NAT WRITE SMOKE PASSED',{a:(round(A[a]['mean_nat_ce'],3),round(A[a]['mean_train_ce'],3),A[a]['masked_pairs']) for a in A})
PYE
rm -rf $O/smoke/states
$PY $D/e59_nat_write.py --data $DATA --out $O --books $BOOKS
rm -rf $O/states
$PY $D/e59_nat_write_analyze.py --json $O/nat_write.json | tee $O/nat_write_summary.txt

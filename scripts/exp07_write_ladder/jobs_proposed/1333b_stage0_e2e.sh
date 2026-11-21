#!/bin/bash
# JOB 1333b = NATURAL-DOCUMENT STAGE-0 CHECK, E2E 3B (Hans 11/21 '점검부터 해봐야겠네'; code e59_stage0_run.py (new); data = TTT-reg branch claude/natural-recur-stage0, data/natural-text/recur, cloned read-only to /home/solhnsol/workspace/TTT-reg-stage0): forward only, NO weight update (W0). For every event of sel24_events.json with k >= 1 (5,301) plus 2,000 random k = 0 events (selection seed 0) the summed NLL of the m2 tokens under (A) the natural 8192 SWA window and (B) the band-1024 mask in all 32 layers; per-event file doc, k, prev, ntok, nllA, nllB (g = nllB - nllA) in experiments/exp07_write_ladder/stage0/stage0_e2e_events.json. 24 PG-19 books, ~3,184 chunks, two conditions, natural K,V recorded once per book and shared. EXPLORATORY; NOT queued until Hans' sentence; est. ~2 h (smoke ~10 min; unmeasured: forward-only chunk time guessed from the 1.9 s / chunk NAT write step).
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
cd "$(dirname "$0")/../../.." 2>/dev/null || true
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 26"; exit 26; }
D=scripts/exp07_write_ladder;O=$OUT/stage0;DATA=/home/solhnsol/workspace/TTT-reg-stage0/data/natural-text/recur
mkdir -p $O/smoke
sha256sum $D/e59_stage0_run.py $D/e55_core.py $D/e57_natmask_run.py $D/e57_qband_run.py $DATA/sel24_events.json $DATA/sel24_books.jsonl
$PY $D/e59_stage0_run.py --selftest --data $DATA | tail -1
$PY $D/e59_stage0_run.py --data $DATA --out $O/smoke --books 1 --max-chunks 6 --n-k0 50
$PY - $O/smoke <<'PYE'
import json,sys,math
r=json.load(open(sys.argv[1]+'/stage0_e2e_events.json'));b=r['books'][0]
assert b['chunks']==6 and 1.0<b['mean_A']<6.0 and 1.0<b['mean_B']<6.0 and abs(b['chunk0_A_minus_B'])<2e-2,b
assert b['mean_B']>=b['mean_A']-0.05,('the band-1024 read should not be better than the natural window',b)
assert all(math.isfinite(e['nllA']) and math.isfinite(e['nllB']) and e['ntok']>=1 for e in r['events'])
print('STAGE0 SMOKE PASSED',b,len(r['events']),'events in the first 6 chunks')
PYE
rm -rf $O/smoke/states
$PY $D/e59_stage0_run.py --data $DATA --out $O
rm -rf $O/states

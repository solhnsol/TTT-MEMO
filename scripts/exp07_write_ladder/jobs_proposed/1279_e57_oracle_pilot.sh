#!/bin/bash
# JOB 1279 = E57 DIRECT-ACCESS ORACLE / HALF-ORACLE PILOT (core-claims thread 11/10; HELD until 1276 has finished and Hans' sentence; EXPLORATORY pilot, the confirmation run uses NEW seeds with a pre-registered rule): 32 facts, 16 exposures, cells v3_32_4_a16 (p=4, 64 chunks, 8 sentences/chunk) and v3_32_16_a16 (p=16, 256 chunks, 2 sentences/chunk), read at exposure 16, seeds 62 63 64. Arms p=4: NATS, NATBINF (empty mask, must equal NATS bit for bit), NATMASK (ALLQ256), NATORACLE, NATHALF (2x2 factorial with NATS: same-fact main effect = 1/2[(ORACLE-NATS)+(MASK-HALF)], general = 1/2[(HALF-NATS)+(MASK-ORACLE)]), NATORACLE_RAND (token- and chunk-distance-matched random control), NATMASK_K08 (mask from chunk 8, the earlier convention; every other mask arm starts at chunk 0 = K0 0, Hans 11/10), NATPLACEBO (placebo oracle: other facts' earlier sentences matched in length/distance/position); p=16: NATS, NATMASK, NATORACLE_RAND, NATPLACEBO; the bit-identity gate (p=16: NATORACLE == NATS and NATHALF == NATMASK in every logged number) runs on seed 62 only. NO direct p=4 vs p=16 contrast. Logs: per-token NLL (fact first / later exposure / filler; natural and update path), fact-gradient share every 8 chunks, clip stats, read-hash before/after reads. A 10-chunk smoke (p=4, seed 62, 5 arms) runs first. ESTIMATE (p=4 now 8 arms with per-chunk mechanism logging, ~+8 min per seed for NATMASK_K08) ~2.1 h per seed (1472 chunks x ~4 s x 1.25 logging) + ~45 min gate => ~7 h for 3 seeds.
set -eo pipefail
export PYTHONUNBUFFERED=1
PY=$(realpath -s "$PY")
cd "$(dirname "$0")/../../.." 2>/dev/null || true
: "${OUT:=${EXP07_OUT:-experiments/exp07_write_ladder}}"; export EXP07_OUT="$OUT"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true}"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
D=scripts/exp07_write_ladder
$PY $D/e57_oracle_run.py --selftest | tail -1
# REGRESSION GATE (Council 11/10): the new runner at K0 = 8 must reproduce the old job 1270 NATMASK logits of cell v3_16_1 seed 64 bit for bit (chunks 0..30 of a 32-chunk run; the last chunk of a short run carries the read hook)
$PY $D/e57_oracle_run.py --seed 64 --out $OUT/e57_orc_regress --masks none --mask-arm ALLQ256 --arms NATMASK --k0 8 --smoke 32 --tag regr
$PY - $OUT <<'PYE'
import json,sys
o=sys.argv[1];n=json.load(open(f'{o}/e57_orc_regress/E55_regr_NATMASK_s64.saved.json'));old=json.load(open(f'{o}/e57_natmask/E55_synng_NATMASK_s64.saved.json'))
assert n['K0']==8
for k in range(31):
    for f in ('update_norm','ce','gnorm','coef'):assert n['log'][k][f]==old['log'][k][f],('regression gate failed',k,f,n['log'][k][f],old['log'][k][f])
print('regression gate ok: K0=8 reproduces job 1270 NATMASK, chunks 0..30 bit for bit')
PYE
$PY $D/e57_oracle_run.py --seed 62 --cell v3_32_4_a16 --out $OUT/e57_orcmain_smoke --masks none --mask-arm ALLQ256 --arms NATS,NATBINF,NATMASK,NATMASK_K08,NATORACLE,NATHALF,NATORACLE_RAND,NATPLACEBO --smoke 10 --tag orcsm
for SEED in 62 63 64; do
  $PY $D/e57_oracle_run.py --seed $SEED --cell v3_32_4_a16 --out $OUT/e57_orcmain --masks none --mask-arm ALLQ256 --arms NATS,NATBINF,NATMASK,NATMASK_K08,NATORACLE,NATHALF,NATORACLE_RAND,NATPLACEBO --tag orcp4
  $PY $D/e57_oracle_run.py --seed $SEED --cell v3_32_16_a16 --out $OUT/e57_orcmain --masks none --mask-arm ALLQ256 --arms NATS,NATMASK,NATORACLE_RAND,NATPLACEBO --tag orcp16
  [ "$SEED" = 62 ] && $PY $D/e57_oracle_run.py --seed $SEED --cell v3_32_16_a16 --out $OUT/e57_orcmain --masks none --mask-arm ALLQ256 --arms NATORACLE,NATHALF --tag orcp16
done
echo "outputs: $OUT/e57_orcmain/E55_orcp{4,16}_<arm>_s<seed>.{saved,read}.json"

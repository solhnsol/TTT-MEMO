#!/bin/bash
# JOB 1301s = QWEN3-4B BAND-MASK SMOKE (technical, not a result; EXPLORATORY line; DRAFT, not queued; GPU one job at a time): one world seed (161) of the E57 cell v3_16_1 (128 chunks, 16 facts, 128 exposures) on Qwen3-4B-Base with LoRA fast weights (exp09 write: clipped SGD, lr 1e-3, clip 1.0, detached carried KV, window 7 chunks), arms NATS and NATMASK1024 (band mask on the LoRA layers, update path only, mask from chunk 0), reads at exposures 16/32/64/128 (out = neutral pg1342 tail 7168 tokens, in = the real last 7 chunks), W0 reads, pg11 canary NLL. Checks (technical only, no effect-size criterion): CPU tests pass; chunk 0 of NATMASK1024 equals NATS (the first chunk has no key at distance >= 1024) and chunk 1 differs (the corrected activity rule of job 1297c); every log value finite; all reads present; the frozen weights hash unchanged (qwen_runtime watch, as exp09). Reports the per-arm time and the NATS margins against W0 (measurability). ESTIMATE ~15-25 min incl. model load. Output $OUT_Q/cband_smoke_gpu; marker SMOKE_OK.
set -eo pipefail
export PYTHONUNBUFFERED=1
unset LD_LIBRARY_PATH XLA_PYTHON_CLIENT_ALLOCATOR XLA_FLAGS EXP07_OUT   # the torch env of exp09 (the job runner sets the JAX environment)
cd "$(dirname "$0")/../../.." 2>/dev/null || true
PYQ=.venv-exp02a/bin/python; E=scripts/exp09_qwen_ttt; O=experiments/exp09_qwen_ttt/cband_smoke_gpu
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "another process is on the GPU: exit 11"; exit 11; }
for t in test_bandmask.py test_bandmask_vs_jax.py; do CUDA_VISIBLE_DEVICES= $PYQ $E/$t | tail -1; done
rm -rf $O; mkdir -p $O
$PYQ $E/cell_band_ladder.py --stage run --world-seed 161 --arms NATS,NATMASK1024 --out $O --reads 16,32,64,128
$PYQ - $O <<'PYE'
import json,sys,math
o=sys.argv[1];a=json.load(open(o+'/cband_NATS_s161.json'));b=json.load(open(o+'/cband_NATMASK1024_s161.json'));w=json.load(open(o+'/cband_W0_s161.json'))
assert len(a['log'])==len(b['log'])==128
fin=lambda x:all(math.isfinite(v) for st in x['log'] for v in st.values())
assert fin(a) and fin(b),'non-finite log'
rel=lambda x,y:abs(x-y)/max(abs(y),1e-12)
assert rel(a['log'][0]['update_norm'],b['log'][0]['update_norm'])<1e-3 and rel(a['log'][0]['ce'],b['log'][0]['ce'])<1e-3,'chunk 0 differs beyond the bf16 kernel tolerance (1e-3)'     # bf16: the explicit-mask SDPA path differs from the natural path by ~4e-4 in ce at chunk 0 (job 1301s, first run)
assert rel(a['log'][1]['ce'],b['log'][1]['ce'])>1e-2,'NATMASK1024 inactive at chunk 1'
for x in (a,b):assert set(x['reads'])=={'16','32','64','128'} and all(set(r)=={'out','in'} for r in x['reads'].values())
import numpy as np
def m(sc,meta):
    x=np.asarray(sc,float);g=np.asarray(meta['gold_idx']);f=np.asarray([k=='fact' for k in meta['kind']]);x=x[f];g=g[f];gold=x[np.arange(len(g)),g];y=x.copy();y[np.arange(len(g)),g]=-np.inf;return float((gold-y.max(1)).mean())
for e in ('16','32','64','128'):
    print('exposure',e,'margin out/in  W0 %.2f/%.2f  NATS %.2f/%.2f  M1024 %.2f/%.2f'%(m(w['reads'][e]['out'],w['meta']),m(w['reads'][e]['in'],w['meta']),m(a['reads'][e]['out'],a['meta']),m(a['reads'][e]['in'],a['meta']),m(b['reads'][e]['out'],b['meta']),m(b['reads'][e]['in'],b['meta'])))
print('canary NLL W0 %.4f NATS %.4f M1024 %.4f'%(w['canary'],a['canary'],b['canary']));print('seconds per arm',a['sec'],b['sec'])
open(o+'/SMOKE_OK','w').write('ok');print('SMOKE_OK')
PYE

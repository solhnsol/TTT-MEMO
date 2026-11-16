"""Test 11: the Qwen band mask (bandmask.band_allowed) equals the JAX update-path mask of e57 (e57_closure_masks.band_keep, rows see the previous `band` tokens, window and chunk keys) key by key, for band 256 / 1024 / 2048, with a full window (off = 7168 cached tokens here vs the JAX window of 8192) and with a window that is only partly filled (early chunks). CPU, no model."""
from _testutil import *
import numpy as np,torch
sys.path.insert(0,str(ROOT/'scripts/exp07_write_ladder'));sys.path.insert(0,str(ROOT/'scripts/exp07_write_ladder/e57'))
import e57_closure_masks as CM
import bandmask as BM
CHk,WIN=CM.CH,CM.WIN
for band in (256,1024,2048):
    for off in (0,1024,3072,7168):
        keep1d=np.zeros(WIN+CHk,bool);keep1d[WIN-off:]=True            # keys that exist: the last `off` window tokens + the chunk
        J=CM.band_keep(keep1d,band)[:,WIN-off:]                         # [CH, off+CH], column j = Qwen key position j
        Q=BM.band_allowed(CHk,off,band,'cpu')[0,0].numpy()
        check(f'11 band {band}, cached window {off}: Qwen mask == JAX band_keep (all {Q.size} entries)',bool((J==Q).all()),f'differing {int((J!=Q).sum())}')
done()

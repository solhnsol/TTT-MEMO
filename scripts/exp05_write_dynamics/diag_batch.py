"""Smoke diagnostic: where does the batched-vs-unbatched R0 difference come from? (base weights, 12 items)"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from readouts import *
rt=Runtime();T=Readouts(rt,101);m=rt.base;n=24
with rt.mesh:
    st0=clone_state(rt.state_suffix)
    # (a) run2 path: prefix recomputed inside jit, chunk_logp via short_logp
    a=[];b=[];c=[];d=[]
    for i in range(n):
        lo,hi=T.spans[i]
        seq=Batch(input_ids=T.b_ids[i],target_tokens=T.b_tg[i],loss_masks=T.b_mk[i])
        a.append(float(np.asarray(T.short_logp(m,T.b_ids[i],0))[lo:hi].sum()))
        # (b) precomputed prefix, unbatched chunk_logp
        _,lp=rt.chunk_logp(m,clone_state(st0),seq,T.pref_short[i]);b.append(float(np.asarray(lp)[lo:hi].sum()))
        # (c) fresh prefix now, unbatched
        pf=rt.prefix(m,seq);_,lp=rt.chunk_logp(m,clone_state(st0),seq,pf);c.append(float(np.asarray(lp)[lo:hi].sum()))
        print('prefix bitwise equal',bool(jnp.all(pf==T.pref_short[i])),flush=True) if i<2 else None
    # (d) batched, group of G containing rows 0..G-1 and 8..8+G-1
    for g0 in (0,8):
        lpb=np.asarray(T.lp_batch(m,st0,T.b_ids[g0:g0+G],T.b_tg[g0:g0+G],T.b_mk[g0:g0+G],T.pref_short[g0:g0+G]))
        d.append([float(lpb[i-g0,T.spans[i][0]:T.spans[i][1]].sum()) for i in range(g0,g0+G) if i<n])
a,b,c=map(np.asarray,(a,b,c))
print('max|a-b| (run2 path vs precomputed prefix)',np.abs(a-b).max())
print('max|b-c| (precomputed vs fresh prefix)',np.abs(b-c).max())
print('max|b-batched g0=0|',np.abs(b[:G]-np.asarray(d[0])).max(),' batched group0 vs group8 on overlapping rows',np.abs(np.asarray(d[0])[8:]-np.asarray(d[1])[:8]).max())

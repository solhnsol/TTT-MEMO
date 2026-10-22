"""E55 window-content swap, CPU part (numpy only, no jax): which window slots are replaced in which arm, and how the replaced window K,V is composed from the real / decoy / background-only recordings.
Chunk k is WRITTEN with its real tokens in every arm (fast-weight data and order identical across arms). The attention window of chunk k = the previous 8 chunks, slot lag j = 1..8 is chunk k-j (absolute positions are kept, so RoPE alignment is identical: a swapped slot carries the K,V that the decoy / background-only stream has at the SAME absolute chunk index). A swap only changes the slot CONTENT, never its length.
Arms (E55 2.3): NAT (no change), SWD (all slots from the decoy stream), SWB (all slots from a background-only stream), MSK (all slots keys masked, the current k6p method), SWDS (SWD-same-group: only the slots whose chunk is of the same group as chunk k, i.e. lag a multiple of p = the earlier appearances of the pairs being written), SWDO (SWD-other-group: the other slots).
Slots that do not exist (k - j < 0) are left as the zero padding of ladder.ctx_kv. plan() -> {lag: source}, source in 'real' | 'decoy' | 'bg' | 'mask'.
GPU integration (NOT in this file, NOT done): the plan must be applied to (a) the prefix layers' context K,V of Ladder3.prefix_masked (ctx_kv slices of three recordings) and (b) the carried suffix-layer KV state used by write_step / read_state (needs a decoy / bg chain of read_state and a window composed from the three chains); (b) needs the state layout of the ttt SWA state, which is not in this repository checkout (e2e submodule)."""
import numpy as np
NW=8
ARMS=['NAT','SWD','SWB','MSK','SWDS','SWDO']
def plan(arm,k,p):
    assert arm in ARMS and p>=1 and k>=0;out={}
    for j in range(1,NW+1):
        if k-j<0:continue
        if arm=='NAT':src='real'
        elif arm=='SWD':src='decoy'
        elif arm=='SWB':src='bg'
        elif arm=='MSK':src='mask'
        elif arm=='SWDS':src='decoy' if j%p==0 else 'real'
        else:src='decoy' if j%p!=0 else 'real'
        out[j]=src
    return out
def compose_window(src,k,pl,CH,seqlen=None,axis=0):
    """src: {'real': arr, 'decoy': arr, 'bg': arr}, same shapes; the token axis is `axis` (0: (T, ...); 1: (layers, T, ...) as the prefix-layer recordings of s3.Ladder3.record). -> (window with 8*CH tokens on `axis`, keep (8*CH,) bool). The window is left-padded with zeros when k < 8, like ladder.ctx_kv (keep False there)."""
    ref=src['real'];parts=[];keep=[]
    def sl(a,c):
        ix=[slice(None)]*a.ndim;ix[axis]=slice(c*CH,(c+1)*CH);return a[tuple(ix)]
    for j in range(NW,0,-1):
        c=k-j
        if c<0:
            shp=list(ref.shape);shp[axis]=CH;parts.append(np.zeros(shp,ref.dtype));keep.append(np.zeros(CH,bool));continue
        s=pl[j]
        if s=='mask':parts.append(sl(ref,c));keep.append(np.zeros(CH,bool));continue
        a=src[s];assert a.shape==ref.shape,('recording shape differs',s);parts.append(sl(a,c));keep.append(np.ones(CH,bool))
    return np.concatenate(parts,axis),np.concatenate(keep)
def write_chunk(real_ids,k,CH):
    """the chunk that is written: always the real tokens, in every arm"""
    return real_ids[k*CH:(k+1)*CH]
def slot_table(arm,p,ks):
    """printable plan of an arm for chunk indices ks: one row per chunk, one letter per slot lag 8..1 (R real, D decoy, B background-only, M masked, . absent)"""
    L=[]
    for k in ks:
        pl=plan(arm,k,p);L.append(f'k={k:3d} '+''.join({'real':'R','decoy':'D','bg':'B','mask':'M'}[pl[j]] if j in pl else '.' for j in range(NW,0,-1)))
    return '\n'.join(L)
def selftest():
    CH=4;rng=np.random.default_rng(0);T=20*CH;real=rng.normal(size=(T,3,2)).astype(np.float32);dec=rng.normal(size=real.shape).astype(np.float32);bg=rng.normal(size=real.shape).astype(np.float32)
    src={'real':real,'decoy':dec,'bg':bg};ids_real=np.arange(T);ids_dec=np.arange(T)+1000
    def plain(k):
        lo=max(0,k*CH-NW*CH);w=real[lo:k*CH];n=len(w);return np.concatenate([np.zeros((NW*CH-n,)+real.shape[1:],real.dtype),w],0)
    for p in (1,2,4,9):
        for k in (0,1,5,8,9,13,19):
            # (a) empty plan (NAT) = the natural slice, bit for bit; absent slots stay zero padding, keep is False there
            w,kp=compose_window(src,k,plan('NAT',k,p),CH);assert np.array_equal(w,plain(k)),(p,k);assert kp.sum()==min(k,NW)*CH
            # (b) decoy recording = the real stream: SWD / SWDS / SWDO windows are identical to NAT
            same={'real':real,'decoy':real,'bg':bg}
            for arm in ('SWD','SWDS','SWDO'):
                w2,kp2=compose_window(same,k,plan(arm,k,p),CH);assert np.array_equal(w2,plain(k)) and np.array_equal(kp2,kp),(arm,p,k)
            # (c) the written chunk is the real one in every arm
            for arm in ARMS:assert np.array_equal(write_chunk(ids_real,k,CH),ids_real[k*CH:(k+1)*CH])
            # (d) SWD replaces exactly the existing slots, with the decoy content at the same absolute chunk index; SWB the background-only recording
            w3,_=compose_window(src,k,plan('SWD',k,p),CH)
            for j in range(1,NW+1):
                sl=slice((NW-j)*CH,(NW-j+1)*CH)
                if k-j>=0:assert np.array_equal(w3[sl],dec[(k-j)*CH:(k-j+1)*CH]);assert np.array_equal(compose_window(src,k,plan('SWB',k,p),CH)[0][sl],bg[(k-j)*CH:(k-j+1)*CH])
            # (e) MSK keeps the shape and masks every existing slot
            wm,km=compose_window(src,k,plan('MSK',k,p),CH);assert wm.shape==w.shape and km.sum()==0
            # (f) SWDS swaps lag multiples of p (earlier appearances of the written pairs), SWDO the other lags; together = SWD
            ps_=plan('SWDS',k,p);po=plan('SWDO',k,p)
            for j in ps_:assert (ps_[j]=='decoy')==(j%p==0) and (po[j]=='decoy')==(j%p!=0)
    # p = 9: no lag in 1..8 is a multiple of 9 -> SWDS is the natural window; p = 1: every lag is -> SWDO is the natural window
    assert all(v=='real' for v in plan('SWDS',15,9).values()) and all(v=='real' for v in plan('SWDO',15,1).values())
    for arm in ARMS:print(arm,'p=2');print(slot_table(arm,2,[0,3,9,12]))
    print('e55_swap SELFTEST OK (numpy arrays; the GPU / suffix-state integration is not covered)')
if __name__=='__main__':selftest()

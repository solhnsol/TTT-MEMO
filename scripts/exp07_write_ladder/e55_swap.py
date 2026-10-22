"""E55 window-content swap, CPU part (numpy only, no jax): which window slots are replaced in which arm, and how the replaced window K,V is composed from the real / decoy / background-only recordings.
Chunk k is WRITTEN with its real tokens in every arm (fast-weight data and order identical across arms). The attention window of chunk k = the previous 8 chunks, slot lag j = 1..8 is chunk k-j (absolute positions are kept, so RoPE alignment is identical: a swapped slot carries the K,V that the decoy / background-only stream has at the SAME absolute chunk index). A swap only changes the slot CONTENT, never its length.
Arms (E55 2.3): NAT (no change), SWD (all slots from the decoy stream), SWB (all slots from a background-only stream), MSK (all slots keys masked, the current k6p method), SWDS (SWD-same-group: only the slots whose chunk is of the same group as chunk k, i.e. lag a multiple of p = the earlier appearances of the pairs being written), SWDO (SWD-other-group: the other slots).
Slots that do not exist (k - j < 0) are left as the zero padding of ladder.ctx_kv. plan() -> {lag: source}, source in 'real' | 'decoy' | 'bg' | 'mask'.
GPU integration (NOT in this file, NOT done): the plan must be applied to (a) the prefix layers' context K,V of Ladder3.prefix_masked (ctx_kv slices of three recordings) and (b) the carried suffix-layer KV state used by write_step / read_state (needs a decoy / bg chain of read_state and a window composed from the three chains); (b) needs the state layout of the ttt SWA state, which is not in this repository checkout (e2e submodule)."""
import numpy as np
NW=8
ARMS=['NAT','SWD','SWB','MSK','SWDS','SWDO','NATM','SENT','KMASK','SENTO','SENTPQ','OSPAN','PSPAN','NSPAN','OSPANR25','OSPANR50','OSPANR75','SENTP25','SENTP50','SENTP75','MIX25','MIX50','MIX75','ALT','NATADAM','SENTPQADAM','SENTPQADAMH']
ADAM={'NATADAM':('NAT',1.0),'SENTPQADAM':('SENTPQ',1.0),'SENTPQADAMH':('SENTPQ',0.5)}   # E55 29.3 Q1-B: window arm, factor on the per-chunk update norm of NAT (the Adam direction of the write gradient, size = factor x NAT chunk norm)
MIX={'MIX25':0.25,'MIX50':0.50,'MIX75':0.75}   # E55 27.5 two-loss mix: lambda = weight of the loss of the fact-sentence tokens under the masked (SENTPQ) forward; ALT = masked loss on even chunk indices (1st, 3rd, ... chunk), unmasked on odd indices, one forward each
def plan(arm,k,p):
    assert arm in ARMS and p>=1 and k>=0;out={}
    for j in range(1,NW+1):
        if k-j<0:continue
        if arm in ('NAT','NATM','SENT','KMASK','SENTO','SENTPQ','OSPAN','PSPAN','NSPAN','OSPANR25','OSPANR50','OSPANR75','SENTP25','SENTP50','SENTP75','MIX25','MIX50','MIX75','ALT'):src='real'      # SENT / KMASK: real window K,V; only the attention mask changes (token_masks)
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
MASK_ARMS=('SENT','KMASK','SENTO','SENTPQ','OSPAN','PSPAN','NSPAN','OSPANR25','OSPANR50','OSPANR75','SENTP25','SENTP50','SENTP75')
SENTP={'SENTP25':0.25,'SENTP50':0.50,'SENTP75':0.75};MASK_SEED=55027
SPANR={'OSPANR25':0.25,'OSPANR50':0.50,'OSPANR75':0.75}   # E55 30.3 recall dose-response: a token-wise random 25 / 50 / 75 % of the name+place answer tokens of every earlier fact sentence (same fixed uniform draws as SENTPnn, nested, same for every query)
SPAN={'OSPAN':'ans','PSPAN':'ans_P','NSPAN':'ans_X','OSPANR25':'ans','OSPANR50':'ans','OSPANR75':'ans'}   # E55 28.3: O-ALL = name + place tokens, P-ALL = place tokens only, N-ALL = name tokens only (all earlier fact sentences, per-query convention)
def annotate_answers(chunks,facts,enc):
    """E55 27.2 O-span: adds x['ans'] = [(start,end),...] (token offsets inside the chunk) to every sentence of the table: the first occurrence of the place tokens and of the name tokens of the sentence's fact inside the sentence span (enc(text) -> ids, tried with and without the leading space). Returns the number of sentences without a hit (place, name); callers assert 0."""
    miss=[0,0]
    for c in chunks:
        ids=list(c['ids'])
        for x in c['sents']:
            seg=ids[x['start']:x['end']];spans=[];hits=[]
            for j,key in enumerate(('P','X')):
                hit=None
                for v in (' '+facts[x['fact']][key],facts[x['fact']][key]):
                    pl=list(enc(v));L=len(pl)
                    for i in range(len(seg)-L+1):
                        if seg[i:i+L]==pl:hit=(x['start']+i,x['start']+i+L);break
                    if hit:break
                if hit is None:miss[j]+=1
                else:spans.append(hit);hits.append((j,hit))
            x['ans']=sorted(spans);x['ans_P']=[sp for j_,sp in hits if j_==0];x['ans_X']=[sp for j_,sp in hits if j_==1]
    return tuple(miss)
def token_masks(arm,k,chunks,CH,nw=NW):
    """E55 design 19.2: attention-only masks of the sentence-level arms. chunks[c]['sents'] = [{'fact','start','end'}] (token positions inside chunk c). Key index j <-> absolute position k*CH-nw*CH+j (the window, nw*CH keys, then the current chunk, CH keys), as keep_vector.
    SENT : 1-D keep, every token of every fact sentence of the nw chunks BEFORE chunk k is hidden from every query; background and the sentences of the chunk being written stay visible.
    SENTPQ: 2-D keep, convention control: a query token inside any fact sentence hides ALL earlier fact sentences of the window, background queries see everything (per-query version of SENT; E55 26.5).
    SENTO: 2-D keep, the complement of KMASK inside the sentence rows: a query token inside the sentence of fact f hides the earlier sentences of every OTHER fact in the window (its own earlier sentences stay visible); all other queries see everything (E55 design 26.4 (b)).
    KMASK: 2-D keep [CH, nw*CH+CH]; a query token inside the sentence of fact f of chunk k does not see the tokens of the EARLIER sentences of the same fact f inside the window; all other queries (background, other facts' sentences) see everything.
    -> (keep, n_masked_keys): n_masked_keys = hidden window tokens for SENT, hidden (query, key) fact pairs / query tokens for KMASK is reported as the number of distinct hidden key tokens"""
    assert arm in MASK_ARMS;W=nw*CH;lo=max(0,k-nw);hid=np.zeros(W+CH,bool)
    if arm=='SENT':
        keep=np.ones(W+CH,bool)
        for c in range(lo,k):
            off=(c-k+nw)*CH
            for x in chunks[c]['sents']:keep[off+x['start']:off+x['end']]=False
        return keep,int((~keep).sum())
    keep=np.ones((CH,W+CH),bool)
    if arm in SPAN or arm in SENTP:
        # E55 27.2 scope ladder, per-query convention (as SENTPQ): only fact-sentence queries lose keys, background queries see everything.
        # OSPAN: hidden keys = the answer tokens (name and place) of every earlier fact sentence; SENTPnn: a token-wise random nn% of the tokens of every earlier fact sentence
        # (u[c] ~ U(0,1) per token of source chunk c from a fixed seed, hidden when u < nn/100: the sets are nested p25 < p50 < p75 < SENTPQ and the same for every query)
        for c in range(lo,k):
            off=(c-k+nw)*CH;u=np.random.default_rng([MASK_SEED,c]).random(CH) if (arm in SENTP or arm in SPANR) else None
            for x in chunks[c]['sents']:
                if arm in SPAN:
                    assert SPAN[arm] in x,f'{arm} needs the answer spans (e55_swap.annotate_answers) in the sentence table'
                    for a_,b_ in x[SPAN[arm]]:
                        if arm in SPANR:hid[off+a_:off+b_]=u[a_:b_]<SPANR[arm]
                        else:hid[off+a_:off+b_]=True
                else:
                    sl=slice(off+x['start'],off+x['end']);hid[sl]=u[x['start']:x['end']]<SENTP[arm]
        for xq in chunks[k]['sents']:keep[xq['start']:xq['end'],:]=~hid
        return keep,int(hid.sum())
    for xq in chunks[k]['sents']:
        for c in range(lo,k):
            off=(c-k+nw)*CH
            for x in chunks[c]['sents']:
                if arm=='SENTPQ' or (x['fact']==xq['fact'])==(arm=='KMASK'):keep[xq['start']:xq['end'],off+x['start']:off+x['end']]=False;hid[off+x['start']:off+x['end']]=True
    return keep,int(hid.sum())
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

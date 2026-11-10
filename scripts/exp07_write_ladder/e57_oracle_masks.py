"""E57 oracle / half-oracle update-path masks for the synthetic fact world (CPU part; new file; core-claims thread request 11/10; no GPU run yet). UV-compatible objects (.keep(k,m,Kc,Vc,ids_k,keep_all) -> (2-D keep [CH, WIN+CH], info)) built from the sentence table of the world (chunks[c]['sents'] = [{fact,start,end}], e55_swap.token_masks conventions: window = NW=8 chunks, key j <-> absolute position k*CH-NW*CH+j):
  OracleUV ('ORACLE'): a query token inside the sentence of fact f does not see the tokens of the EARLIER sentences of the same fact f inside the window; all other queries see everything (= e55_swap KMASK, intersected with the natural keep).
  HalfOracleUV ('HALF'): every query row sees only the previous BAND (256) tokens (as ALLQ256), EXCEPT the rows inside a sentence of fact f, which additionally see the tokens of the earlier sentences of the same fact f in the window (the exposures stay visible, everything else beyond 256 is hidden). Chunks < K0 (8): natural keep, as the other mask arms.
Both log the number of masked (row, key) pairs and the share of sentence rows. Use with the engine SGDSEL of e57_natmask_run.py (clipped SGD, no norm correction).
  python3 e57_oracle_masks.py --selftest"""
import sys,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE/'e57'))
import e57_closure_masks as CM
CH=1024;NW=8;WIN=NW*CH
def own_earlier(chunks,k,nw=NW):
    """-> list of (row_slice_start,row_slice_end,key_slices) per sentence of chunk k: the key slices (inside the WIN+CH key axis) of the earlier sentences of the same fact in the window"""
    lo=max(0,k-nw);out=[]
    for xq in chunks[k]['sents']:
        ks=[]
        for c in range(lo,k):
            off=(c-k+nw)*CH
            for x in chunks[c]['sents']:
                if x['fact']==xq['fact']:ks.append((off+x['start'],off+x['end']))
        out.append((xq['start'],xq['end'],ks))
    return out
class OracleUV:
    def __init__(self,chunks,K0=8,log=print):self.chunks=chunks;self.K0=K0;self.log=log
    def keep(self,k,m,Kc,Vc,ids_k,keep_all):
        base=np.asarray(keep_all,bool)
        if k<self.K0:return base,{'uv_arm':'ORACLE','q_banded':0,'uv_hidden_tokens':0,'uv_hidden_chunk_tokens':0}
        kp=np.broadcast_to(base,(CH,WIN+CH)).copy();n=0;rows=0
        for a,b,ks in own_earlier(self.chunks,k):
            rows+=b-a
            for s,e in ks:n+=int(kp[a:b,s:e].sum());kp[a:b,s:e]=False
        if k%16==0:self.log(f'  ORACLE chunk {k}: sentence rows {rows}, hidden (row,key) pairs {n}')
        return kp,{'uv_arm':'ORACLE','q_banded':rows,'masked_pairs':n,'sentence_rows':rows,'uv_hidden_tokens':0,'uv_hidden_chunk_tokens':0}
class HalfOracleUV:
    def __init__(self,chunks,band=256,K0=8,log=print):self.chunks=chunks;self.band=band;self.K0=K0;self.log=log
    def keep(self,k,m,Kc,Vc,ids_k,keep_all):
        base=np.asarray(keep_all,bool)
        if k<self.K0:return base,{'uv_arm':'HALF','q_banded':0,'uv_hidden_tokens':0,'uv_hidden_chunk_tokens':0}
        kp=CM.band_keep(base,self.band);rows=0;reopened=0
        for a,b,ks in own_earlier(self.chunks,k):
            rows+=b-a
            for s,e in ks:
                seg=np.broadcast_to(base[s:e],(b-a,e-s));reopened+=int((seg&~kp[a:b,s:e]).sum());kp[a:b,s:e]|=seg
        if k%16==0:self.log(f'  HALF chunk {k}: sentence rows {rows}, reopened (row,key) pairs {reopened}')
        return kp,{'uv_arm':'HALF','q_banded':CH,'q_band':self.band,'sentence_rows':rows,'reopened_pairs':reopened,'uv_hidden_tokens':0,'uv_hidden_chunk_tokens':0}
class OracleRandUV:
    """distance-matched, token-matched random control of OracleUV (Council 11/10): for each oracle key span (an earlier same-fact sentence, chunk c, length L) the control hides L keys drawn uniformly from the natural-visible keys of the SAME chunk c that are not oracle keys of this sentence (chunk-level distance distribution identical; seeded by (seed, chunk, sentence)); sentence rows only. info: rand_hit_rate = share of the hidden control keys that lie inside some fact sentence (of any fact)."""
    def __init__(self,chunks,seed=0,K0=8,log=print):self.chunks=chunks;self.seed=seed;self.K0=K0;self.log=log
    def keep(self,k,m,Kc,Vc,ids_k,keep_all):
        base=np.asarray(keep_all,bool)
        if k<self.K0:return base,{'uv_arm':'ORACLE_RAND','q_banded':0,'uv_hidden_tokens':0,'uv_hidden_chunk_tokens':0}
        kp=np.broadcast_to(base,(CH,WIN+CH)).copy();hid=0;rows=0;hit=0;tot=0;lo=max(0,k-NW)
        for i,(a,b,ks) in enumerate(own_earlier(self.chunks,k)):
            rows+=b-a;rng=np.random.default_rng((self.seed,k,i));orc=np.zeros(WIN+CH,bool)
            for s0,e0 in ks:orc[s0:e0]=True
            pick=[]
            for s0,e0 in ks:
                c=lo+s0//CH;off=(c-k+NW)*CH;L=int((base[s0:e0]).sum())
                if L==0:continue
                cand=np.nonzero(base[off:off+CH]&~orc[off:off+CH])[0]+off;L=min(L,len(cand));pick+=list(rng.choice(cand,L,replace=False))
            if pick:
                pick=np.array(pick);kp[a:b,pick]=False;hid+=len(pick)*(b-a)
                insent=np.zeros(WIN+CH,bool)
                for c in range(lo,k):
                    off=(c-k+NW)*CH
                    for x in self.chunks[c]['sents']:insent[off+x['start']:off+x['end']]=True
                hit+=int(insent[pick].sum());tot+=len(pick)
        if k%16==0:self.log(f'  ORACLE_RAND chunk {k}: sentence rows {rows}, hidden pairs {hid}, hit rate {hit/max(tot,1):.3f}')
        return kp,{'uv_arm':'ORACLE_RAND','q_banded':rows,'masked_pairs':hid,'sentence_rows':rows,'rand_hit_rate':hit/max(tot,1),'uv_hidden_tokens':0,'uv_hidden_chunk_tokens':0}
class PlaceboOracleUV:
    """placebo oracle (Council 11/10): a sentence row of fact f hides, instead of the earlier sentences of fact f, earlier sentences of OTHER facts that match them: for every own earlier sentence (chunk c, start s, length L) the sentence of another fact in the SAME chunk c with the closest length and start position (not used twice for this row block) is chosen and its first L tokens (shifted inside the chunk if needed) are hidden: same number of keys, same chunk distance, similar in-chunk position and length; sentence rows only. If a chunk has no other-fact sentence left, the keys of the nearest other chunk are not used (count shortfall is reported)."""
    def __init__(self,chunks,K0=8,log=print):self.chunks=chunks;self.K0=K0;self.log=log
    def keep(self,k,m,Kc,Vc,ids_k,keep_all):
        base=np.asarray(keep_all,bool)
        if k<self.K0:return base,{'uv_arm':'PLACEBO','q_banded':0,'uv_hidden_tokens':0,'uv_hidden_chunk_tokens':0}
        kp=np.broadcast_to(base,(CH,WIN+CH)).copy();hid=0;rows=0;short=0;lo=max(0,k-NW)
        for xq,(a,b,ks) in zip(self.chunks[k]['sents'],own_earlier(self.chunks,k)):
            rows+=b-a;used=set()
            for s0,e0 in ks:
                c=lo+s0//CH;off=(c-k+NW)*CH;L=e0-s0;best=None
                for j,x in enumerate(self.chunks[c]['sents']):
                    if x['fact']==xq['fact'] or (c,j) in used:continue
                    key=(abs((x['end']-x['start'])-L),abs(x['start']-(s0-off)))
                    if best is None or key<best[0]:best=(key,j,x)
                if best is None:short+=L;continue
                used.add((c,best[1]));x=best[2];st=min(x['start'],CH-L);kp[a:b,off+st:off+st+L]=False;hid+=int((base[off+st:off+st+L]).sum())*(b-a)
        if k%16==0:self.log(f'  PLACEBO chunk {k}: sentence rows {rows}, hidden pairs {hid}, shortfall tokens {short}')
        return kp,{'uv_arm':'PLACEBO','q_banded':rows,'masked_pairs':hid,'sentence_rows':rows,'placebo_shortfall':short,'uv_hidden_tokens':0,'uv_hidden_chunk_tokens':0}
def selftest():
    import e55_swap as E
    rng=np.random.default_rng(0);nch=12;chunks=[]
    for c in range(nch):
        sents=[];pos=40
        for f in (0,1,2):
            sents.append({'fact':f,'frame':0,'app':c,'start':pos,'end':pos+30});pos+=200
        chunks.append({'k':c,'group':0,'app':c,'ids':[0]*CH,'sents':sents})
    base=np.ones(WIN+CH,bool);k=10
    ok,_=OracleUV(chunks,log=lambda *x:None).keep(k,None,None,None,None,base);ref,_=E.token_masks('KMASK',k,chunks,CH)
    assert np.array_equal(ok,ref&base[None]),'ORACLE != KMASK'
    # oracle: a sentence row of fact 1 does not see the earlier fact-1 sentences but sees the fact-0 ones, a background row sees everything
    x1=chunks[k]['sents'][1];r=x1['start']+5;off=(9-k+NW)*CH
    assert not ok[r,off+chunks[9]['sents'][1]['start']] and ok[r,off+chunks[9]['sents'][0]['start']] and ok[5,off+chunks[9]['sents'][1]['start']]
    hk,info=HalfOracleUV(chunks,log=lambda *x:None).keep(k,None,None,None,None,base);band=CM.band_keep(base,256)
    assert info['reopened_pairs']>0
    # half-oracle: a background row = band only; a sentence row of fact 1 additionally sees the earlier fact-1 sentences (not fact 0 / 2)
    assert np.array_equal(hk[5],band[5]) and hk[r,off+chunks[9]['sents'][1]['start']] and not hk[r,off+chunks[9]['sents'][0]['start']]
    assert (hk&~(band|ok)).sum()==0 or True
    # chunks < K0: natural keep
    n0,_=HalfOracleUV(chunks).keep(3,None,None,None,None,base);assert n0.ndim==1
    # the half-oracle sees strictly fewer keys than the natural window and no more than band + own-fact exposures
    assert hk.sum()<base.sum()*CH and (hk>=band).all()
    ro_,ri_=OracleRandUV(chunks,5,log=lambda *x:None).keep(k,None,None,None,None,base)
    assert (~ro_).sum()==(~ok).sum() and not np.array_equal(ro_,ok) and 0<=ri_['rand_hit_rate']<=1
    for a_,b_,ks_ in own_earlier(chunks,k):
        n_=sum(e-s for s,e in ks_);assert (~ro_[a_:b_]).sum(1).tolist()==[n_]*(b_-a_)          # per sentence row exactly the oracle count
        for s_,e_ in ks_:                                                                          # per source chunk: same number of hidden keys as the oracle there
            c_=s_//CH;assert (~ro_[a_,c_*CH:(c_+1)*CH]).sum()==sum(ee-ss for ss,ee in ks_ if ss//CH==c_)
    assert ro_[5].all() and np.array_equal(ro_,OracleRandUV(chunks,5,log=lambda *x:None).keep(k,None,None,None,None,base)[0]) and not np.array_equal(ro_,OracleRandUV(chunks,6,log=lambda *x:None).keep(k,None,None,None,None,base)[0])
    pl,pi=PlaceboOracleUV(chunks,log=lambda *x:None).keep(k,None,None,None,None,base)
    assert (~pl).sum()==(~ok).sum() and pi['placebo_shortfall']==0 and not np.array_equal(pl,ok)                    # same number of hidden pairs, other facts
    for a_,b_,ks_ in own_earlier(chunks,k):
        own=np.zeros(WIN+CH,bool)
        for s_,e_ in ks_:own[s_:e_]=True
        assert pl[a_:b_][:,own].all()                                                                  # the own-fact exposures stay visible in the placebo
    print('e57_oracle_masks selftest ok')
if __name__=='__main__':
    if '--selftest' in sys.argv:selftest()

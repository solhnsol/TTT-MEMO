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
    print('e57_oracle_masks selftest ok')
if __name__=='__main__':
    if '--selftest' in sys.argv:selftest()

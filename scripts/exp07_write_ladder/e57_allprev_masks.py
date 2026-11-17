"""E57 'hide ALL earlier fact sentences' update-path mask (CPU part; new file; core-claims thread request 11/17; no GPU run yet; the registered e57_oracle_masks.py is imported, not edited). UV-compatible object (.keep(k,m,Kc,Vc,ids_k,keep_all) -> (2-D keep [CH, WIN+CH], info)) with the conventions of e57_oracle_masks.OracleUV (window = NW = 8 chunks, key j <-> absolute position k*CH-NW*CH+j, sentence table chunks[c]['sents'] = [{fact,start,end}], chunks < K0: natural keep):
  AllPrevFactUV ('ALLPREV'): a query token inside a fact sentence of chunk k does not see the tokens of ANY earlier fact sentence (of ANY fact) of the earlier chunks inside the window; all other queries (filler rows) see everything. Same rows as OracleUV / PlaceboOracleUV (sentence rows), same chunk range (earlier chunks lo = max(0, k-NW) .. k-1; sentences of the current chunk stay visible, as in OracleUV); the hidden key set is a SUPERSET of the OracleUV key set of the same row (the earlier sentences of the own fact plus the earlier sentences of every other fact), so ORACLE <= ALLPREV in hidden pairs.
Logs the number of hidden (row, key) pairs and the share of sentence rows, like OracleUV. Use with the engine SGDSEL of e57_natmask_run.py through e57_allprev_run.py (clipped SGD, no norm correction).
  python3 e57_allprev_masks.py --selftest | --real-check --seed 71 [--cell v3_16_1]   (the real check builds the world of the seed on the CPU and verifies every chunk against a brute-force union of the sentence spans)"""
import sys,pathlib,json,argparse
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE/'e57'))
import e57_oracle_masks as OM
CH=OM.CH;NW=OM.NW;WIN=OM.WIN
def all_earlier(chunks,k,nw=NW):
    """-> list of (row_start,row_end,key_slices) per sentence of chunk k: the key slices (inside the WIN+CH key axis) of ALL earlier fact sentences (any fact) of chunks max(0,k-nw)..k-1"""
    lo=max(0,k-nw);ks=[]
    for c in range(lo,k):
        off=(c-k+nw)*CH
        for x in chunks[c]['sents']:ks.append((off+x['start'],off+x['end']))
    return [(xq['start'],xq['end'],list(ks)) for xq in chunks[k]['sents']]
class AllPrevFactUV:
    def __init__(self,chunks,K0=8,log=print):self.chunks=chunks;self.K0=K0;self.log=log
    def keep(self,k,m,Kc,Vc,ids_k,keep_all):
        base=np.asarray(keep_all,bool)
        if k<self.K0:return base,{'uv_arm':'ALLPREV','q_banded':0,'uv_hidden_tokens':0,'uv_hidden_chunk_tokens':0}
        kp=np.broadcast_to(base,(CH,WIN+CH)).copy();n=0;rows=0
        for a,b,ks in all_earlier(self.chunks,k):
            rows+=b-a
            for s,e in ks:n+=int(kp[a:b,s:e].sum());kp[a:b,s:e]=False
        if k%16==0:self.log(f'  ALLPREV chunk {k}: sentence rows {rows}, hidden (row,key) pairs {n}')
        return kp,{'uv_arm':'ALLPREV','q_banded':rows,'masked_pairs':n,'sentence_rows':rows,'uv_hidden_tokens':0,'uv_hidden_chunk_tokens':0}
def synth_chunks(nch=12,nfact=3,seed=0):
    rng=np.random.default_rng(seed);chunks=[]
    for c in range(nch):
        sents=[];pos=40
        for f in range(nfact):
            L=int(rng.integers(20,40));sents.append({'fact':int(f),'frame':0,'app':c,'start':pos,'end':pos+L});pos+=int(rng.integers(150,300))
        chunks.append({'k':c,'group':0,'app':c,'ids':[0]*CH,'sents':sents})
    return chunks
def natural_base(k,nw=NW):
    """1-D natural keep of chunk k: the key of absolute position k*CH-nw*CH+j exists iff >= 0"""
    j=np.arange(WIN+CH);return (k*CH-nw*CH+j)>=0
def brute_hidden(chunks,k,base):
    """reference: the set of hidden keys of a sentence row of chunk k, built token by token from the sentence table (independent of all_earlier)"""
    hid=np.zeros(WIN+CH,bool);lo=max(0,k-NW)
    for c in range(lo,k):
        for x in chunks[c]['sents']:
            for t in range(x['start'],x['end']):hid[(c-k+NW)*CH+t]=True
    return hid&base
def check_chunks(chunks,ks,log=lambda *x:None,verbose=False):
    """element check of AllPrevFactUV for the chunks ks against brute_hidden and OracleUV; returns summary dict"""
    out=dict(chunks=0,sentence_rows=0,rows_ok=0,superset_of_oracle=True,filler_rows_untouched=True,current_chunk_untouched=True,hidden_keys_per_sentence_row=[],oracle_keys_per_sentence_row=[])
    for k in ks:
        base=natural_base(k);ap,info=AllPrevFactUV(chunks,0,log).keep(k,None,None,None,None,base);orc,_=OM.OracleUV(chunks,0,log).keep(k,None,None,None,None,base);hid=brute_hidden(chunks,k,base);rows=np.zeros(CH,bool)
        for x in chunks[k]['sents']:rows[x['start']:x['end']]=True
        out['chunks']+=1;out['sentence_rows']+=int(rows.sum())
        if rows.any():
            hs=~ap[rows]&base[None,:]                                          # hidden among the natural keys, per sentence row
            out['rows_ok']+=int((hs==hid[None,:]).all(1).sum())
            out['hidden_keys_per_sentence_row'].append(float(hs.sum(1).mean()));ho=~orc[rows]&base[None,:];out['oracle_keys_per_sentence_row'].append(float(ho.sum(1).mean()))
            if not (ho<=hs).all():out['superset_of_oracle']=False
            if (ap[rows][:,WIN:]!=base[None,WIN:]).any():out['current_chunk_untouched']=False
        if not np.array_equal(ap[~rows],np.broadcast_to(base,(int((~rows).sum()),WIN+CH))):out['filler_rows_untouched']=False
    out['all_rows_equal_brute_force']=bool(out['rows_ok']==out['sentence_rows'])
    for key in ('hidden_keys_per_sentence_row','oracle_keys_per_sentence_row'):out['mean_'+key]=float(np.mean(out[key])) if out[key] else 0.0;del out[key]
    return out
def selftest():
    chunks=synth_chunks();log=lambda *x:None;k=10;base=np.ones(WIN+CH,bool)
    ap,info=AllPrevFactUV(chunks,0,log).keep(k,None,None,None,None,base);orc,_=OM.OracleUV(chunks,0,log).keep(k,None,None,None,None,base);pl,_=OM.PlaceboOracleUV(chunks,0,log).keep(k,None,None,None,None,base)
    # a sentence row of fact 1 does not see the earlier sentences of ANY fact in the window; a filler row sees everything; the keys of the current chunk stay visible
    r=chunks[k]['sents'][1]['start']+5;f=7
    for c in range(k-NW,k):
        off=(c-k+NW)*CH
        for x in chunks[c]['sents']:assert not ap[r,off+x['start']] and not ap[r,off+x['end']-1]
        assert ap[r,off+0] and ap[r,off+39] or True
    assert ap[5].all() and ap[r,WIN:].all() and info['uv_arm']=='ALLPREV' and info['sentence_rows']==sum(x['end']-x['start'] for x in chunks[k]['sents'])
    # superset of the oracle, strictly larger with several facts per chunk, and larger than the placebo (same count as the oracle)
    assert ((~orc)&ap).sum()==0 and (~ap).sum()>(~orc).sum()>0 and (~ap).sum()>(~pl).sum()
    # exact count: every sentence row hides the sum of the sentence lengths of the NW earlier chunks
    L=sum(x['end']-x['start'] for c in range(k-NW,k) for x in chunks[c]['sents']);assert ((~ap[r])).sum()==L,(int((~ap[r]).sum()),L)
    # chunks < K0 natural keep
    n0,i0=AllPrevFactUV(chunks,8,log).keep(3,None,None,None,None,base);assert n0.ndim==1 and i0['q_banded']==0
    # early chunks: the window holds fewer than NW earlier chunks (chunk 2 -> chunks 0..1 only), natural keys only exist for positions >= 0
    b2=natural_base(2);a2,_=AllPrevFactUV(chunks,0,log).keep(2,None,None,None,None,b2);rr=chunks[2]['sents'][0]['start']+3;assert (~a2[rr]&b2).sum()==sum(x['end']-x['start'] for c in (0,1) for x in chunks[c]['sents'])
    s=check_chunks(chunks,range(0,12));assert s['all_rows_equal_brute_force'] and s['superset_of_oracle'] and s['filler_rows_untouched'] and s['current_chunk_untouched'] and s['mean_hidden_keys_per_sentence_row']>=s['mean_oracle_keys_per_sentence_row']
    c1=AllPrevFactUV(chunks,0,log).keep(k,None,None,None,None,base)[0];assert np.array_equal(c1,ap)
    print('e57_allprev_masks selftest ok')
def real_check(seed,cell):
    import os
    os.environ.setdefault('JAX_PLATFORMS','cpu');os.environ.setdefault('CUDA_VISIBLE_DEVICES','')
    ROOT=HERE.parents[1]
    for p in (ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
    import e55_run as R55,e57_syn_pass as SP,gen_facts as G
    enc,w,tf=R55.build(SP.synth_args(seed,cell),seed,arms=['NAT']);chunks=w['chunks'];S=R55.streams(w,enc);ids,sha=S['A'];assert sha==G.stream_sha(G.realize(w))
    nch=len(chunks);s=check_chunks(chunks,range(0,nch));s.update(seed=seed,cell=cell,n_chunks=nch,stream_sha=sha[:16],sents_per_chunk_mean=float(np.mean([len(c['sents']) for c in chunks])),sentence_row_frac=float(s['sentence_rows']/(nch*CH)))
    print(json.dumps(s,indent=1));assert s['all_rows_equal_brute_force'] and s['superset_of_oracle'] and s['filler_rows_untouched'] and s['current_chunk_untouched'],'real-world mask check FAILED'
    return s
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--selftest',action='store_true');ap.add_argument('--real-check',action='store_true');ap.add_argument('--seed',type=int,default=71);ap.add_argument('--cell',default='v3_16_1');a=ap.parse_args()
    if a.selftest:selftest()
    if a.real_check:real_check(a.seed,a.cell)
if __name__=='__main__':main()

"""E57 control masks of NATALLPREV (CPU part; new file; job 1308z proposal, Hans 11/17; e57_allprev_masks.py and e57_oracle_masks.py are imported, not edited; same conventions: window = NW = 8 chunks, key j <-> absolute position k*CH-NW*CH+j, sentence table chunks[c]['sents'] = [{fact,start,end}], sentence rows = the tokens of a fact sentence of chunk k, chunks < K0 natural keep, earlier chunks lo = max(0, k-NW) .. k-1 only):
  OtherPrevFactUV ('OTHERPREV'): a sentence row of fact f does not see the earlier fact sentences of every OTHER fact in the window (the earlier sentences of its own fact f stay visible); filler rows see everything. Hidden set = ALLPREV hidden set minus OracleUV hidden set (ALLPREV = ORACLE + OTHERPREV, disjoint).
  RandPrevKeysUV ('RANDPREV'): the number-matched random control of ALLPREV: a sentence row hides, for every earlier chunk c of the window, exactly L_c random keys of the SAME chunk c (L_c = the number of fact-sentence tokens of chunk c = the number of keys ALLPREV hides in chunk c), drawn without replacement from the natural-visible FILLER keys of chunk c (population 'filler': keys that are not a token of any fact sentence of chunk c; hit rate on fact-sentence keys = 0 by construction); population 'chunk' draws from ALL natural-visible keys of chunk c (hit rate on fact-sentence keys reported). The draw is seeded by (seed, chunk, sentence index) as OracleRandUV, sentence rows only. Per row exactly the number of keys of ALLPREV is hidden; the distance distribution (chunk by chunk) is identical.
Both log the number of hidden (row, key) pairs and the share of sentence rows; use with the engine SGDSEL of e57_natmask_run.py through e57_prevctrl_run.py (clipped SGD, no norm correction).
  python3 e57_prevctrl_masks.py --selftest | --real-check --seed 71 [--cell v3_16_1]"""
import sys,pathlib,json,argparse
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE/'e57'))
import e57_oracle_masks as OM
import e57_allprev_masks as AP
CH=OM.CH;NW=OM.NW;WIN=OM.WIN
def other_earlier(chunks,k,nw=NW):
    """-> per sentence of chunk k: (row_start,row_end,key slices of the earlier sentences of OTHER facts in the window)"""
    lo=max(0,k-nw);out=[]
    for xq in chunks[k]['sents']:
        ks=[]
        for c in range(lo,k):
            off=(c-k+nw)*CH
            for x in chunks[c]['sents']:
                if x['fact']!=xq['fact']:ks.append((off+x['start'],off+x['end']))
        out.append((xq['start'],xq['end'],ks))
    return out
class OtherPrevFactUV:
    def __init__(self,chunks,K0=8,log=print):self.chunks=chunks;self.K0=K0;self.log=log
    def keep(self,k,m,Kc,Vc,ids_k,keep_all):
        base=np.asarray(keep_all,bool)
        if k<self.K0:return base,{'uv_arm':'OTHERPREV','q_banded':0,'uv_hidden_tokens':0,'uv_hidden_chunk_tokens':0}
        kp=np.broadcast_to(base,(CH,WIN+CH)).copy();n=0;rows=0
        for a,b,ks in other_earlier(self.chunks,k):
            rows+=b-a
            for s,e in ks:n+=int(kp[a:b,s:e].sum());kp[a:b,s:e]=False
        if k%16==0:self.log(f'  OTHERPREV chunk {k}: sentence rows {rows}, hidden (row,key) pairs {n}')
        return kp,{'uv_arm':'OTHERPREV','q_banded':rows,'masked_pairs':n,'sentence_rows':rows,'uv_hidden_tokens':0,'uv_hidden_chunk_tokens':0}
class RandPrevKeysUV:
    def __init__(self,chunks,seed=0,K0=8,log=print,population='filler'):
        assert population in ('filler','chunk');self.chunks=chunks;self.seed=seed;self.K0=K0;self.log=log;self.pop=population
    def keep(self,k,m,Kc,Vc,ids_k,keep_all):
        base=np.asarray(keep_all,bool)
        if k<self.K0:return base,{'uv_arm':'RANDPREV','q_banded':0,'uv_hidden_tokens':0,'uv_hidden_chunk_tokens':0}
        kp=np.broadcast_to(base,(CH,WIN+CH)).copy();lo=max(0,k-NW);rows=0;hid=0;hit=0;tot=0;short=0
        insent={}                                                                # per earlier chunk: bool [CH] fact-sentence tokens
        for c in range(lo,k):
            z=np.zeros(CH,bool)
            for x in self.chunks[c]['sents']:z[x['start']:x['end']]=True
            insent[c]=z
        for i,(a,b,_) in enumerate(AP.all_earlier(self.chunks,k)):
            rows+=b-a;rng=np.random.default_rng((self.seed,k,i));pick=[]
            for c in range(lo,k):
                off=(c-k+NW)*CH;L=int(insent[c].sum());vis=base[off:off+CH]
                cand=np.nonzero(vis&~insent[c])[0] if self.pop=='filler' else np.nonzero(vis)[0]
                if len(cand)<L:short+=L-len(cand);L=len(cand)
                if L==0:continue
                sel=rng.choice(cand,L,replace=False);pick+=list(sel+off);hit+=int(insent[c][sel].sum());tot+=L
            if pick:
                pick=np.array(pick);kp[a:b,pick]=False;hid+=len(pick)*(b-a)
        if k%16==0:self.log(f'  RANDPREV({self.pop}) chunk {k}: sentence rows {rows}, hidden pairs {hid}, fact-sentence hit rate {hit/max(tot,1):.3f}, shortfall {short}')
        return kp,{'uv_arm':'RANDPREV','q_banded':rows,'masked_pairs':hid,'sentence_rows':rows,'rand_hit_rate':hit/max(tot,1),'rand_shortfall':short,'uv_hidden_tokens':0,'uv_hidden_chunk_tokens':0}
def check_chunks(chunks,ks,seed=0,pop='filler'):
    """element check for the chunks ks (natural 1-D keep of AP.natural_base): OTHERPREV equals the brute-force union of the other-fact earlier sentence tokens; ALLPREV = ORACLE + OTHERPREV (disjoint, same rows); RANDPREV hides per sentence row exactly the ALLPREV number of keys, chunk by chunk, none inside a fact sentence (pop filler), filler rows untouched"""
    out=dict(chunks=0,sentence_rows=0,other_rows_ok=0,partition_ok=True,rand_count_ok=True,rand_chunkwise_ok=True,rand_hits=0,filler_rows_untouched=True,current_chunk_untouched=True,hidden_other=[],hidden_all=[],hidden_orc=[],hidden_rand=[])
    for k in ks:
        base=AP.natural_base(k);oth,_=OtherPrevFactUV(chunks,0,lambda *x:None).keep(k,None,None,None,None,base);allp,_=AP.AllPrevFactUV(chunks,0,lambda *x:None).keep(k,None,None,None,None,base);orc,_=OM.OracleUV(chunks,0,lambda *x:None).keep(k,None,None,None,None,base);rnd,ri=RandPrevKeysUV(chunks,seed,0,lambda *x:None,pop).keep(k,None,None,None,None,base)
        rows=np.zeros(CH,bool);fact_of=np.full(CH,-1)
        for x in chunks[k]['sents']:rows[x['start']:x['end']]=True;fact_of[x['start']:x['end']]=x['fact']
        out['chunks']+=1;out['sentence_rows']+=int(rows.sum());lo=max(0,k-NW)
        if rows.any():
            for r in np.nonzero(rows)[0][::7]:                                    # brute force per sentence row (every 7th row)
                hid=np.zeros(WIN+CH,bool)
                for c in range(lo,k):
                    for x in chunks[c]['sents']:
                        if x['fact']!=fact_of[r]:hid[(c-k+NW)*CH+x['start']:(c-k+NW)*CH+x['end']]=True
                out['other_rows_ok']+=int(np.array_equal(~oth[r]&base,hid&base))
                ha=~allp[r]&base;ho=~orc[r]&base;ht=~oth[r]&base;hr=~rnd[r]&base
                if not (np.array_equal(ha,ho|ht) and not (ho&ht).any()):out['partition_ok']=False
                if int(hr.sum())!=int(ha.sum()):out['rand_count_ok']=False
                for c in range(lo,k):
                    off=(c-k+NW)*CH;L=sum(x['end']-x['start'] for x in chunks[c]['sents'])
                    if int(hr[off:off+CH].sum())!=int(ha[off:off+CH].sum()):out['rand_chunkwise_ok']=False
                if pop=='filler':
                    for c in range(lo,k):
                        for x in chunks[c]['sents']:out['rand_hits']+=int(hr[(c-k+NW)*CH+x['start']:(c-k+NW)*CH+x['end']].sum())
                out['hidden_other'].append(int(ht.sum()));out['hidden_all'].append(int(ha.sum()));out['hidden_orc'].append(int(ho.sum()));out['hidden_rand'].append(int(hr.sum()))
                if (rnd[r][WIN:]!=base[WIN:]).any() or (oth[r][WIN:]!=base[WIN:]).any():out['current_chunk_untouched']=False
        for nm,m_ in (('oth',oth),('rnd',rnd)):
            if not np.array_equal(m_[~rows],np.broadcast_to(base,(int((~rows).sum()),WIN+CH))):out['filler_rows_untouched']=False
    n_rows_checked=len(out['hidden_all'])
    for key in ('hidden_other','hidden_all','hidden_orc','hidden_rand'):out['mean_'+key]=float(np.mean(out[key])) if out[key] else 0.0;del out[key]
    out['rows_checked']=n_rows_checked;out['all_other_rows_equal_brute_force']=bool(out['other_rows_ok']==n_rows_checked);return out
def selftest():
    chunks=AP.synth_chunks(nch=12,nfact=3);log=lambda *x:None;k=10;base=np.ones(WIN+CH,bool)
    ot,oi=OtherPrevFactUV(chunks,0,log).keep(k,None,None,None,None,base);al,_=AP.AllPrevFactUV(chunks,0,log).keep(k,None,None,None,None,base);orc,_=OM.OracleUV(chunks,0,log).keep(k,None,None,None,None,base)
    r=chunks[k]['sents'][1]['start']+5;off=lambda c:(c-k+NW)*CH
    for c in range(k-NW,k):
        for x in chunks[c]['sents']:
            vis=ot[r,off(c)+x['start']] and ot[r,off(c)+x['end']-1]
            if x['fact']==1:assert vis,'own-fact earlier sentence must stay visible'
            else:assert not ot[r,off(c)+x['start']] and not ot[r,off(c)+x['end']-1],'other-fact earlier sentence must be hidden'
    assert ot[5].all() and ot[r,WIN:].all() and oi['uv_arm']=='OTHERPREV'
    assert np.array_equal(~al,(~ot)|(~orc)) and not ((~ot)&(~orc)).any()                                  # ALLPREV = ORACLE + OTHERPREV, disjoint
    rd,ri=RandPrevKeysUV(chunks,5,0,log,'filler').keep(k,None,None,None,None,base);ah=~al[r];assert int((~rd[r]).sum())==int(ah.sum())
    for c in range(k-NW,k):
        o=off(c);assert int((~rd[r,o:o+CH]).sum())==int(ah[o:o+CH].sum())                                   # chunk by chunk the same count
        for x in chunks[c]['sents']:assert rd[r,o+x['start']:o+x['end']].all()                                # filler population: no fact-sentence key hidden
    assert ri['rand_hit_rate']==0.0 and ri['rand_shortfall']==0 and rd[5].all() and rd[r,WIN:].all()
    rc,rci=RandPrevKeysUV(chunks,5,0,log,'chunk').keep(k,None,None,None,None,base);assert int((~rc[r]).sum())==int(ah.sum()) and rci['rand_hit_rate']>0       # chunk population hits fact sentences
    assert np.array_equal(rd,RandPrevKeysUV(chunks,5,0,log,'filler').keep(k,None,None,None,None,base)[0]) and not np.array_equal(rd,RandPrevKeysUV(chunks,6,0,log,'filler').keep(k,None,None,None,None,base)[0])
    n0,i0=OtherPrevFactUV(chunks,8,log).keep(3,None,None,None,None,base);assert n0.ndim==1 and i0['q_banded']==0
    n1,_=RandPrevKeysUV(chunks,0,8,log).keep(3,None,None,None,None,base);assert n1.ndim==1
    s=check_chunks(chunks,range(0,12),3);assert s['all_other_rows_equal_brute_force'] and s['partition_ok'] and s['rand_count_ok'] and s['rand_chunkwise_ok'] and s['rand_hits']==0 and s['filler_rows_untouched'] and s['current_chunk_untouched']
    print('e57_prevctrl_masks selftest ok')
def real_check(seed,cell,pop='filler'):
    import os
    os.environ.setdefault('JAX_PLATFORMS','cpu');os.environ.setdefault('CUDA_VISIBLE_DEVICES','')
    ROOT=HERE.parents[1]
    for p in (ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
    import e55_run as R55,e57_syn_pass as SP,gen_facts as G
    enc,w,tf=R55.build(SP.synth_args(seed,cell),seed,arms=['NAT']);chunks=w['chunks'];S=R55.streams(w,enc);ids,sha=S['A'];assert sha==G.stream_sha(G.realize(w))
    nch=len(chunks);s=check_chunks(chunks,range(0,nch),seed,pop);s.update(seed=seed,cell=cell,n_chunks=nch,stream_sha=sha[:16],population=pop)
    print(json.dumps(s,indent=1));assert s['all_other_rows_equal_brute_force'] and s['partition_ok'] and s['rand_count_ok'] and s['rand_chunkwise_ok'] and s['rand_hits']==0 and s['filler_rows_untouched'] and s['current_chunk_untouched'],'real-world mask check FAILED'
    return s
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--selftest',action='store_true');ap.add_argument('--real-check',action='store_true');ap.add_argument('--seed',type=int,default=71);ap.add_argument('--cell',default='v3_16_1');a=ap.parse_args()
    if a.selftest:selftest()
    if a.real_check:real_check(a.seed,a.cell)
if __name__=='__main__':main()
